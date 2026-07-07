"""Automated data update service.

Fetches lineups, injuries, match results, standings, and team profiles
from free data sources (ESPN, Football-Data.org, API-Football).
Runs on schedule to keep data fresh.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from football_model.data import LocalDatabase
from football_model.data.adapters.espn import ESPNAdapter, ESPN_LEAGUES
from football_model.data.adapters.football_data_org import FootballDataOrgAdapter
from football_model.data.adapters.api_football import APIFootballAdapter

logger = logging.getLogger(__name__)


class DataUpdateService:
    """Automated data update from free sources."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database
        self.espn = ESPNAdapter()
        self.fdorg = FootballDataOrgAdapter()
        self.api_football = APIFootballAdapter()

    def update_all(self) -> dict[str, int]:
        """Run all updates. Returns counts of updated items."""
        results = {
            "lineups_updated": 0,
            "results_updated": 0,
            "standings_updated": 0,
            "profiles_updated": 0,
            "injuries_updated": 0,
        }

        try:
            results["lineups_updated"] = self.update_lineups()
        except Exception as e:
            logger.warning("Lineup update failed: %s", e)

        try:
            results["results_updated"] = self.update_match_results()
        except Exception as e:
            logger.warning("Results update failed: %s", e)

        try:
            results["standings_updated"] = self.update_standings()
        except Exception as e:
            logger.warning("Standings update failed: %s", e)

        try:
            results["profiles_updated"] = self.update_team_profiles()
        except Exception as e:
            logger.warning("Profile update failed: %s", e)

        try:
            results["injuries_updated"] = self.update_injuries()
        except Exception as e:
            logger.warning("Injury update failed: %s", e)

        return results

    def update_lineups(self) -> int:
        """Fetch and store lineups for upcoming/recent matches."""
        count = 0
        with self.database.connection(read_only=True) as conn:
            # Get recent sporttery matches that need lineups
            matches = conn.execute("""
                SELECT match_id, home_team, away_team, league_name, kickoff
                FROM sporttery_matches
                WHERE kickoff >= CURRENT_DATE - INTERVAL '7' DAY
                ORDER BY kickoff DESC
                LIMIT 20
            """).fetchall()

        for match_row in matches:
            match_id, home_team, away_team, league_name, kickoff = match_row
            try:
                home_players = self.espn.get_team_roster(league_name or "", home_team)
                away_players = self.espn.get_team_roster(league_name or "", away_team)

                if home_players and len(home_players) >= 11:
                    self._save_lineup(match_id, "home", home_players, kickoff)
                    count += 1

                if away_players and len(away_players) >= 11:
                    self._save_lineup(match_id, "away", away_players, kickoff)
                    count += 1

            except Exception as e:
                logger.debug("Lineup fetch failed for %s: %s", match_id, e)

        return count

    def update_match_results(self) -> int:
        """Fetch and store match results from ESPN."""
        count = 0
        today = datetime.now()
        start = (today - timedelta(days=7)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")

        for league_name, espn_league in ESPN_LEAGUES.items():
            try:
                events = self.espn._get_events(espn_league, start, end)
                for event in events:
                    comp = event.get("competitions", [{}])[0]
                    status = comp.get("status", {}).get("type", {}).get("name", "")
                    if status not in ("FULL_TIME", "FINAL", "POST"):
                        continue

                    competitors = comp.get("competitors", [])
                    if len(competitors) != 2:
                        continue

                    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

                    home_goals = int(home.get("score", "0"))
                    away_goals = int(away.get("score", "0"))

                    # Generate match_id
                    match_id = f"espn:{event.get('id', '')}"

                    with self.database.connection() as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO match_results
                            (match_id, status, home_goals, away_goals, provider, updated_at)
                            VALUES (?, 'completed', ?, ?, 'espn', CURRENT_TIMESTAMP)
                        """, [match_id, home_goals, away_goals])
                    count += 1

            except Exception as e:
                logger.debug("Results fetch failed for %s: %s", league_name, e)

        return count

    def update_team_profiles(self) -> int:
        """Build team profiles from historical match data."""
        count = 0
        with self.database.connection(read_only=True) as conn:
            teams = conn.execute("""
                SELECT home_team as team, competition,
                       COUNT(*) as matches,
                       SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) as draws,
                       SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END) as losses,
                       AVG(home_goals) as avg_goals_for,
                       AVG(away_goals) as avg_goals_against
                FROM matches
                WHERE home_goals IS NOT NULL
                GROUP BY home_team, competition
                HAVING COUNT(*) >= 10
            """).fetchall()

        with self.database.connection() as conn:
            for row in teams:
                team, comp, matches, wins, draws, losses, gf, ga = row
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO team_profiles
                        (team_name, league, ranking, points, goals_for, goals_against,
                         xg_for, xg_against, home_strength, away_strength,
                         form_last_5, form_last_10, elo_rating,
                         last_update)
                        VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, CURRENT_TIMESTAMP)
                    """, [team, comp, wins * 3 + draws, int(gf * matches), int(ga * matches),
                          float(gf), float(ga), float(gf), float(ga)])
                    count += 1
                except Exception as e:
                    logger.debug("Profile save failed for %s: %s", team, e)

        return count

    def _save_lineup(self, match_id: str, team_side: str, players: list, captured_at) -> None:
        """Save lineup to database."""
        players_json = str([
            {"name": p.name, "position": p.position, "jersey": p.jersey, "starter": p.starter}
            for p in players
        ])

        with self.database.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lineup_snapshots
                (match_id, provider_fixture_id, team_side, is_current, formation,
                 confirmed, players_json, captured_at)
                VALUES (?, 0, ?, TRUE, NULL, TRUE, ?, ?)
            """, [match_id, team_side, players_json, captured_at])

    def update_standings(self) -> int:
        """Fetch and store league standings from Football-Data.org."""
        count = 0
        for league_name in ["英格兰超级联赛", "西班牙甲级联赛", "意大利甲级联赛", "德国甲级联赛", "法国甲级联赛"]:
            try:
                standings = self.fdorg.get_standings(league_name)
                for s in standings:
                    with self.database.connection() as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO team_profiles
                            (team_name, league, ranking, points, goals_for, goals_against,
                             xg_for, xg_against, home_strength, away_strength,
                             form_last_5, form_last_10, elo_rating, last_update)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, CURRENT_TIMESTAMP)
                        """, [
                            s.team_name, league_name, s.position, s.points,
                            s.goals_for, s.goals_against,
                            s.goals_for / max(s.played, 1), s.goals_against / max(s.played, 1),
                            s.won / max(s.played, 1), s.lost / max(s.played, 1),
                            s.form[-5:] if s.form else None,
                            s.form[-10:] if s.form else None,
                        ])
                    count += 1
            except Exception as e:
                logger.debug("Standings update failed for %s: %s", league_name, e)
        return count

    def update_injuries(self) -> int:
        """Fetch and store injury data from API-Football."""
        if not self.api_football.enabled:
            return 0

        count = 0
        with self.database.connection(read_only=True) as conn:
            teams = conn.execute("""
                SELECT DISTINCT home_team FROM sporttery_matches
                WHERE kickoff >= CURRENT_DATE - INTERVAL '7' DAY
                LIMIT 20
            """).fetchall()

        for (team_name,) in teams:
            try:
                injuries = self.api_football.get_injuries(team_name)
                for inj in injuries:
                    with self.database.connection() as conn:
                        conn.execute("""
                            INSERT INTO injury_snapshots
                            (match_id, team_side, captured_at, injuries_json)
                            VALUES (?, 'unknown', CURRENT_TIMESTAMP, ?)
                        """, [team_name, str([{
                            "player": inj.player_name,
                            "type": inj.injury_type,
                            "reason": inj.reason,
                            "status": inj.status,
                        }])])
                    count += 1
            except Exception as e:
                logger.debug("Injury update failed for %s: %s", team_name, e)
        return count
