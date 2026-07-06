from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from football_model.core import map_team_name
from football_model.data import LocalDatabase
from football_model.data.adapters.lineups import ApiFootballAdapter


@dataclass(frozen=True)
class ContextSyncResult:
    available: bool
    fixture_resolved: bool = False
    current_lineup: bool = False
    previous_lineup: bool = False
    injuries: bool = False
    result_updated: bool = False
    message: str = ""


class LiveContextService:
    def __init__(self, database: LocalDatabase, adapter: ApiFootballAdapter | None = None) -> None:
        self.database = database
        self.adapter = adapter or ApiFootballAdapter()

    def sync_match(self, match: pd.Series, competition: str) -> ContextSyncResult:
        if not self.adapter.available:
            return ContextSyncResult(False, message="未配置API_FOOTBALL_KEY")
        kickoff = pd.Timestamp(match["kickoff"])
        with self.database.connection(read_only=True) as connection:
            cached = connection.execute(
                """SELECT provider_fixture_id, home_provider_team_id, away_provider_team_id
                FROM provider_fixtures WHERE match_id=? AND provider='api-football'""",
                [match["match_id"]],
            ).fetchone()
        params = {"id": int(cached[0])} if cached else {"date": kickoff.strftime("%Y-%m-%d")}
        payload = self.adapter._request("/fixtures", params)
        if not payload:
            return ContextSyncResult(True, message="供应商未返回赛程")
        expected_home = map_team_name(competition, str(match["home_team"])).lower()
        expected_away = map_team_name(competition, str(match["away_team"])).lower()
        fixture = next(
            (
                item
                for item in payload.get("response", [])
                if expected_home in item["teams"]["home"]["name"].lower()
                and expected_away in item["teams"]["away"]["name"].lower()
            ),
            None,
        )
        if fixture is None:
            return ContextSyncResult(True, message="未匹配到供应商比赛ID")
        fixture_id = int(fixture["fixture"]["id"])
        home_id = int(fixture["teams"]["home"]["id"])
        away_id = int(fixture["teams"]["away"]["id"])
        status = fixture["fixture"]["status"]["short"]
        with self.database.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO provider_fixtures
                VALUES (?, 'api-football', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [match["match_id"], fixture_id, home_id, away_id, fixture["fixture"]["venue"].get("name"), status],
            )
            goals = fixture.get("goals", {})
            halftime = fixture.get("score", {}).get("halftime", {})
            if status in {"FT", "AET", "PEN"}:
                connection.execute(
                    """INSERT OR REPLACE INTO match_results
                    VALUES (?, ?, ?, ?, ?, ?, 'api-football', CURRENT_TIMESTAMP)""",
                    [
                        match["match_id"],
                        status,
                        goals.get("home"),
                        goals.get("away"),
                        halftime.get("home"),
                        halftime.get("away"),
                    ],
                )
        hours_to_kickoff = (kickoff - pd.Timestamp.now()).total_seconds() / 3600
        should_check_lineup = hours_to_kickoff <= 3 or status in {"FT", "AET", "PEN"}
        current = self.adapter.get_lineups(fixture_id) if should_check_lineup else None
        if current:
            self._store_lineups(str(match["match_id"]), fixture_id, current, True)
        injuries = self.adapter.get_injuries(
            fixture_id,
            home_team_id=home_id,
            away_team_id=away_id,
        )
        if injuries:
            with self.database.connection() as connection:
                for side, players in [("home", injuries.home_injured), ("away", injuries.away_injured)]:
                    connection.execute(
                        "INSERT OR REPLACE INTO injury_snapshots VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                        [
                            match["match_id"],
                            side,
                            json.dumps([player.__dict__ for player in players], ensure_ascii=False),
                        ],
                    )
        previous_found = False
        if self.adapter.account_plan().lower() == "free":
            return ContextSyncResult(
                True,
                True,
                bool(current),
                False,
                bool(injuries),
                status in {"FT", "AET", "PEN"},
                "免费方案不支持2026历史首发；本场首发将在临场同步",
            )
        for side, team_id in [("home", home_id), ("away", away_id)]:
            previous = self.adapter._request(
                "/fixtures",
                {
                    "team": team_id,
                    "season": kickoff.year,
                    "from": (kickoff - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                    "to": (kickoff - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "status": "FT",
                },
            )
            candidates = previous.get("response", []) if previous else []
            prior = next(
                (
                    item
                    for item in reversed(candidates)
                    if pd.Timestamp(item["fixture"]["date"]).tz_localize(None) < kickoff
                ),
                None,
            )
            if prior:
                lineup = self.adapter.get_lineups(int(prior["fixture"]["id"]))
                if lineup:
                    selected = lineup.home if int(prior["teams"]["home"]["id"]) == team_id else lineup.away
                    self._store_team(str(match["match_id"]), int(prior["fixture"]["id"]), side, selected, False)
                    previous_found = True
        return ContextSyncResult(
            True, True, bool(current), previous_found, bool(injuries), status in {"FT", "AET", "PEN"}
        )

    def _store_lineups(self, match_id: str, fixture_id: int, lineups, current: bool) -> None:
        self._store_team(match_id, fixture_id, "home", lineups.home, current)
        self._store_team(match_id, fixture_id, "away", lineups.away, current)

    def _store_team(self, match_id: str, fixture_id: int, side: str, lineup, current: bool) -> None:
        players = [player.__dict__ for player in lineup.starters]
        with self.database.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO lineup_snapshots
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [match_id, fixture_id, side, current, lineup.formation, True, json.dumps(players, ensure_ascii=False)],
            )
