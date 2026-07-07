"""Football-Data.org adapter for match results and standings.

Free tier: 10 requests/min, 100 requests/day
No API key required for basic access.
Docs: https://docs.football-data.org/general/v4/index.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Competition code mapping
COMPETITIONS = {
    "英格兰超级联赛": "PL",
    "西班牙甲级联赛": "PD",
    "意大利甲级联赛": "SA",
    "德国甲级联赛": "BL1",
    "法国甲级联赛": "FL1",
    "世界杯国家队": "WC",
}


@dataclass
class Standing:
    team_name: str
    position: int
    points: int
    played: int
    won: int
    draw: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    form: str  # e.g. "WWDLW"


@dataclass
class MatchResult:
    match_id: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    status: str  # "FINISHED", "SCHEDULED", "IN_PLAY"
    matchday: int
    utc_date: str


class FootballDataOrgAdapter:
    """Fetch match results and standings from Football-Data.org (free)."""

    def __init__(self) -> None:
        self.enabled = True  # No key required
        self._client = httpx.Client(timeout=15, headers={"Accept": "application/json"})

    def get_standings(self, league_name: str) -> list[Standing]:
        """Get current league standings."""
        comp_code = COMPETITIONS.get(league_name)
        if not comp_code:
            logger.warning("Football-Data.org: unsupported league '%s'", league_name)
            return []

        try:
            resp = self._client.get(f"{BASE_URL}/competitions/{comp_code}/standings")
            if resp.status_code == 429:
                logger.warning("Football-Data.org: rate limited")
                return []
            if resp.status_code != 200:
                logger.warning("Football-Data.org error: %d", resp.status_code)
                return []

            data = resp.json()
            standings = []
            for table in data.get("standings", []):
                if table.get("type") != "TOTAL":
                    continue
                for row in table.get("table", []):
                    team = row.get("team", {})
                    standings.append(Standing(
                        team_name=team.get("name", ""),
                        position=row.get("position", 0),
                        points=row.get("points", 0),
                        played=row.get("playedGames", 0),
                        won=row.get("won", 0),
                        draw=row.get("draw", 0),
                        lost=row.get("lost", 0),
                        goals_for=row.get("goalsFor", 0),
                        goals_against=row.get("goalsAgainst", 0),
                        goal_difference=row.get("goalDifference", 0),
                        form=row.get("form", ""),
                    ))
            return standings

        except Exception as e:
            logger.warning("Football-Data.org standings failed: %s", e)
            return []

    def get_recent_results(self, league_name: str, limit: int = 10) -> list[MatchResult]:
        """Get recent match results."""
        comp_code = COMPETITIONS.get(league_name)
        if not comp_code:
            return []

        try:
            resp = self._client.get(
                f"{BASE_URL}/competitions/{comp_code}/matches",
                params={"status": "FINISHED"},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for match in data.get("matches", [])[:limit]:
                score = match.get("score", {})
                ft = score.get("fullTime", {})
                results.append(MatchResult(
                    match_id=f"fdorg:{match.get('id', '')}",
                    home_team=match.get("homeTeam", {}).get("name", ""),
                    away_team=match.get("awayTeam", {}).get("name", ""),
                    home_goals=ft.get("home", 0) or 0,
                    away_goals=ft.get("away", 0) or 0,
                    status=match.get("status", ""),
                    matchday=match.get("matchday", 0),
                    utc_date=match.get("utcDate", ""),
                ))
            return results

        except Exception as e:
            logger.warning("Football-Data.org results failed: %s", e)
            return []

    def get_upcoming_matches(self, league_name: str, limit: int = 10) -> list[MatchResult]:
        """Get upcoming scheduled matches."""
        comp_code = COMPETITIONS.get(league_name)
        if not comp_code:
            return []

        try:
            resp = self._client.get(
                f"{BASE_URL}/competitions/{comp_code}/matches",
                params={"status": "SCHEDULED"},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for match in data.get("matches", [])[:limit]:
                results.append(MatchResult(
                    match_id=f"fdorg:{match.get('id', '')}",
                    home_team=match.get("homeTeam", {}).get("name", ""),
                    away_team=match.get("awayTeam", {}).get("name", ""),
                    home_goals=0,
                    away_goals=0,
                    status=match.get("status", ""),
                    matchday=match.get("matchday", 0),
                    utc_date=match.get("utcDate", ""),
                ))
            return results

        except Exception as e:
            logger.warning("Football-Data.org upcoming failed: %s", e)
            return []
