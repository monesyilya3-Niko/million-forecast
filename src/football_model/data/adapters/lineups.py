"""Injury and lineup adapter using API-Football (free tier).

API-Football provides lineups, injuries, and squad data.
Free tier: 100 requests/day. Requires API key from api-football.com.

If no API key is configured, the adapter gracefully returns None
and the system continues without these features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerInfo:
    """Information about a single player."""

    name: str
    position: str
    number: int
    is_starter: bool
    is_injured: bool = False
    injury_type: str = ""


@dataclass(frozen=True)
class TeamLineup:
    """Lineup for one team."""

    team_name: str
    formation: str
    starters: list[PlayerInfo]
    substitutes: list[PlayerInfo]
    injured: list[PlayerInfo]


@dataclass(frozen=True)
class MatchLineups:
    """Lineups for both teams in a match."""

    home: TeamLineup
    away: TeamLineup
    lineup_confirmed: bool


@dataclass(frozen=True)
class InjuryReport:
    """Injury report for a match."""

    home_injured: list[PlayerInfo]
    away_injured: list[PlayerInfo]
    total_injured: int


class ApiFootballAdapter:
    """Adapter for API-Football data (lineups, injuries)."""

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
        self._available = bool(self.api_key)
        if not self._available:
            logger.info("API-Football key not set, lineup/injury features disabled")

    @property
    def available(self) -> bool:
        return self._available

    def account_plan(self) -> str:
        data = self._request("/status", {})
        if not data:
            return ""
        return str(data.get("response", {}).get("subscription", {}).get("plan", ""))

    def _request(self, endpoint: str, params: dict) -> dict | None:
        """Make authenticated request to API-Football."""
        if not self._available:
            return None

        headers = {"x-apisports-key": self.api_key}
        try:
            response = httpx.get(
                f"{self.BASE_URL}{endpoint}",
                headers=headers,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("errors"):
                logger.warning(f"API-Football errors: {data['errors']}")
                return None
            return data
        except Exception as e:
            logger.warning(f"API-Football request failed: {e}")
            return None

    def get_lineups(
        self,
        fixture_id: int,
    ) -> MatchLineups | None:
        """Get lineups for a fixture.

        Args:
            fixture_id: API-Football fixture ID

        Returns:
            MatchLineups or None
        """
        data = self._request("/fixtures/lineups", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return None

        teams = data["response"]
        if len(teams) < 2:
            return None

        def parse_team(team_data: dict) -> TeamLineup:
            starters = []
            substitutes = []
            for player in team_data.get("startXI", []):
                p = player.get("player", {})
                starters.append(
                    PlayerInfo(
                        name=p.get("name", ""),
                        position=p.get("pos", ""),
                        number=p.get("number", 0),
                        is_starter=True,
                    )
                )
            for player in team_data.get("substitutes", []):
                p = player.get("player", {})
                substitutes.append(
                    PlayerInfo(
                        name=p.get("name", ""),
                        position=p.get("pos", ""),
                        number=p.get("number", 0),
                        is_starter=False,
                    )
                )

            return TeamLineup(
                team_name=team_data.get("team", {}).get("name", ""),
                formation=team_data.get("formation", ""),
                starters=starters,
                substitutes=substitutes,
                injured=[],
            )

        return MatchLineups(
            home=parse_team(teams[0]),
            away=parse_team(teams[1]),
            lineup_confirmed=True,
        )

    def get_injuries(
        self,
        fixture_id: int,
        *,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
    ) -> InjuryReport | None:
        """Get injuries for a fixture.

        Args:
            fixture_id: API-Football fixture ID

        Returns:
            InjuryReport or None
        """
        if home_team_id is None or away_team_id is None:
            logger.warning("Home/away provider team IDs are required for injury attribution")
            return None
        data = self._request("/injuries", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return None

        home_injured = []
        away_injured = []

        for entry in data["response"]:
            player = entry.get("player", {})
            injury = player.get("reason", "") or player.get("type", "")

            info = PlayerInfo(
                name=player.get("name", ""),
                position=player.get("pos", ""),
                number=player.get("number", 0),
                is_starter=False,
                is_injured=True,
                injury_type=injury,
            )

            team_id = entry.get("team", {}).get("id")
            if team_id == home_team_id:
                home_injured.append(info)
            elif team_id == away_team_id:
                away_injured.append(info)

        return InjuryReport(
            home_injured=home_injured,
            away_injured=away_injured,
            total_injured=len(home_injured) + len(away_injured),
        )


def lineup_to_features(
    lineups: MatchLineups | None,
    injuries: InjuryReport | None,
) -> dict[str, float]:
    """Convert lineup/injury data to numeric features."""
    features = {
        "lineup_confirmed": 0.0,
        "home_starters_count": 11.0,
        "away_starters_count": 11.0,
        "home_injured_count": 0.0,
        "away_injured_count": 0.0,
        "total_injured": 0.0,
    }

    if lineups:
        features["lineup_confirmed"] = 1.0
        features["home_starters_count"] = float(len(lineups.home.starters))
        features["away_starters_count"] = float(len(lineups.away.starters))
        features["home_injured_count"] = float(len(lineups.home.injured))
        features["away_injured_count"] = float(len(lineups.away.injured))

    if injuries:
        features["home_injured_count"] = float(len(injuries.home_injured))
        features["away_injured_count"] = float(len(injuries.away_injured))
        features["total_injured"] = float(injuries.total_injured)

    return features
