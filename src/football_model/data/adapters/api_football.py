"""API-Football adapter for injury and squad data.

Free tier: 100 requests/day at api-football.com
Get your key at: https://www.api-football.com/

Set environment variable: API_FOOTBALL_KEY=your_key_here
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"


@dataclass
class InjuryInfo:
    player_name: str
    team_name: str
    injury_type: str
    reason: str
    status: str  # "injured", "doubtful", "suspended"


class APIFootballAdapter:
    """Fetch injury data from API-Football (free tier: 100 req/day)."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("API_FOOTBALL_KEY", "")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.info("API_FOOTBALL_KEY not set, injury data disabled")

    def get_injuries(self, team_name: str, league_name: str = "") -> list[InjuryInfo]:
        """Get current injuries for a team."""
        if not self.enabled:
            return []

        try:
            headers = {
                "x-apisports-key": self.api_key,
            }
            params = {
                "team": team_name,
                "season": "2025",
            }
            resp = httpx.get(
                f"{API_FOOTBALL_BASE}/injuries",
                headers=headers,
                params=params,
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("API-Football error: %d", resp.status_code)
                return []

            data = resp.json()
            injuries = []
            for item in data.get("response", []):
                player = item.get("player", {})
                team = item.get("team", {})
                injury = item.get("player", {}).get("reason", "")

                injuries.append(InjuryInfo(
                    player_name=player.get("name", "Unknown"),
                    team_name=team.get("name", team_name),
                    injury_type=player.get("type", ""),
                    reason=injury,
                    status="injured" if "injury" in injury.lower() else "doubtful",
                ))

            return injuries

        except Exception as e:
            logger.warning("API-Football injury fetch failed: %s", e)
            return []

    def get_team_id(self, team_name: str) -> int | None:
        """Search for team ID by name."""
        if not self.enabled:
            return None

        try:
            headers = {"x-apisports-key": self.api_key}
            resp = httpx.get(
                f"{API_FOOTBALL_BASE}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                teams = data.get("response", [])
                if teams:
                    return teams[0].get("team", {}).get("id")
        except Exception as e:
            logger.debug("Team search failed: %s", e)
        return None
