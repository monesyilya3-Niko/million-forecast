"""Automatic match result updater.

Checks for finished matches and updates results from available sources:
1. API-Football (if API key set)
2. ESPN (free, for international matches)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from football_model.data import LocalDatabase
from football_model.data.adapters.espn import ESPNAdapter
from football_model.data.adapters.lineups import ApiFootballAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateResult:
    checked: int
    updated: int
    errors: int
    message: str


class ResultUpdaterService:
    """Update match results after games finish."""

    def __init__(self, database: LocalDatabase, adapter: ApiFootballAdapter | None = None) -> None:
        self.database = database
        self.adapter = adapter or ApiFootballAdapter()
        self.espn = ESPNAdapter()

    def update_pending_results(self) -> UpdateResult:
        """Check for matches that should have results and try to fetch them."""
        # Find matches that are past kickoff but have no result
        with self.database.connection(read_only=True) as conn:
            pending = conn.execute("""
                SELECT s.match_id, s.home_team, s.away_team, s.kickoff, s.league_name
                FROM sporttery_matches s
                LEFT JOIN match_results r ON s.match_id = r.match_id
                WHERE r.match_id IS NULL
                  AND s.kickoff < CURRENT_TIMESTAMP
                ORDER BY s.kickoff DESC
            """).fetchall()

        if not pending:
            return UpdateResult(0, 0, 0, "无待更新比赛")

        checked = 0
        updated = 0
        errors = 0

        for match in pending:
            match_id, home_team, away_team, kickoff, league_name = match
            checked += 1

            try:
                # Try ESPN first (free, works for international matches)
                result = self._fetch_from_espn(match_id, home_team, away_team, league_name, kickoff)
                if result:
                    updated += 1
                    continue

                # Try API-Football
                if self.adapter.available:
                    result = self._fetch_from_api_football(match_id, home_team, away_team, kickoff)
                    if result:
                        updated += 1
                        continue

            except Exception as e:
                logger.warning("Failed to update result for %s: %s", match_id, e)
                errors += 1

        msg = f"检查{checked}场，更新{updated}场，失败{errors}场"
        return UpdateResult(checked, updated, errors, msg)

    def _fetch_from_espn(self, match_id: str, home_team: str, away_team: str, league_name: str, kickoff: object) -> bool:
        """Try to fetch result from ESPN."""
        import pandas as pd

        # Map league to ESPN league ID
        from football_model.data.adapters.espn import ESPN_LEAGUES, TEAM_SEARCH_MAP

        espn_league = ESPN_LEAGUES.get(league_name)
        if not espn_league:
            return False

        en_home = TEAM_SEARCH_MAP.get(home_team, home_team)
        en_away = TEAM_SEARCH_MAP.get(away_team, away_team)

        # Search for the match in recent events
        kickoff_dt = pd.to_datetime(kickoff)
        start_date = (kickoff_dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
        end_date = (kickoff_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")

        events = self.espn._get_events(espn_league, start_date, end_date)
        if not events:
            return False

        for event in events:
            event_name = event.get("name", "")
            # Check if this is our match
            if (en_home.lower() in event_name.lower() and en_away.lower() in event_name.lower()) or \
               (en_away.lower() in event_name.lower() and en_home.lower() in event_name.lower()):

                comp = event.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {}).get("name", "")

                if "FULL_TIME" not in status and "FINAL" not in status and "POST" not in status:
                    return False  # Match not finished

                competitors = comp.get("competitors", [])
                home_goals = None
                away_goals = None

                for c in competitors:
                    score = c.get("score")
                    home_away = c.get("homeAway", "")

                    if score is not None:
                        if home_away == "home":
                            home_goals = int(score)
                        elif home_away == "away":
                            away_goals = int(score)

                if home_goals is not None and away_goals is not None:
                    # Get halftime if available
                    ht_home = None
                    ht_away = None
                    halftime = comp.get("score", {}).get("halftime", {})
                    if halftime:
                        ht_home = halftime.get("home")
                        ht_away = halftime.get("away")

                    with self.database.connection() as conn:
                        conn.execute(
                            """INSERT OR REPLACE INTO match_results
                            VALUES (?, 'FT', ?, ?, ?, ?, 'espn', CURRENT_TIMESTAMP)""",
                            [match_id, home_goals, away_goals, ht_home, ht_away]
                        )

                    logger.info("ESPN: Updated %s %d-%d %s", home_team, home_goals, away_goals, away_team)
                    return True

        return False

    def _fetch_from_api_football(self, match_id: str, home_team: str, away_team: str, kickoff: object) -> bool:
        """Try to fetch result from API-Football."""
        if not self.adapter.available:
            return False

        # Check if we have a provider fixture ID
        with self.database.connection(read_only=True) as conn:
            fixture = conn.execute(
                "SELECT provider_fixture_id FROM provider_fixtures WHERE match_id=?",
                [match_id]
            ).fetchone()

        if not fixture:
            return False

        fixture_id = int(fixture[0])
        data = self.adapter._request("/fixtures", {"id": fixture_id})

        if not data or not data.get("response"):
            return False

        fixture_data = data["response"][0]
        status = fixture_data["fixture"]["status"]["short"]

        if status not in ("FT", "AET", "PEN"):
            return False

        goals = fixture_data.get("goals", {})
        halftime = fixture_data.get("score", {}).get("halftime", {})

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        if home_goals is None or away_goals is None:
            return False

        # Save result
        with self.database.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO match_results
                VALUES (?, ?, ?, ?, ?, ?, 'api-football', CURRENT_TIMESTAMP)""",
                [match_id, status, home_goals, away_goals,
                 halftime.get("home"), halftime.get("away")]
            )

        logger.info("API-Football: Updated %s %d-%d %s", home_team, home_goals, away_goals, away_team)
        return True

    def sync_results_from_sporttery(self) -> UpdateResult:
        """Refresh sporttery data and check for status updates."""
        from football_model.data.adapters import SportteryAdapter
        from football_model.data.repositories import SportteryRepository

        try:
            adapter = SportteryAdapter()
            snapshot = adapter.fetch()
            repo = SportteryRepository(self.database)
            matches, odds = repo.upsert(snapshot.matches, snapshot.odds)

            return UpdateResult(matches, 0, 0, f"同步{matches}场比赛数据")

        except Exception as e:
            logger.warning("Sporttery sync failed: %s", e)
            return UpdateResult(0, 0, 1, f"同步失败: {e}")


def auto_update_results(database: LocalDatabase) -> str:
    """Convenience function to run result updates."""
    updater = ResultUpdaterService(database)

    # First sync latest sporttery data
    sync_result = updater.sync_results_from_sporttery()

    # Then try to update results
    update_result = updater.update_pending_results()

    return f"{sync_result.message}；{update_result.message}"
