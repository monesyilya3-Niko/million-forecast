"""Live match state provider using API-Football.

Provides real-time match data including scores, cards, corners,
shots, and xG when available.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from .base import BaseProvider, ProviderHealth, ProviderResult

logger = logging.getLogger(__name__)


class LiveStateProvider(BaseProvider):
    """Fetch live match state from API-Football.

    Requires API_FOOTBALL_KEY environment variable.
    Free tier: 100 requests/day.
    """

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str | None = None) -> None:
        import os
        self._api_key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
        self._available = bool(self._api_key)
        self._error_count = 0
        self._last_error: str | None = None
        self._last_sync: datetime | None = None

    @property
    def name(self) -> str:
        return "api-football"

    @property
    def available(self) -> bool:
        return self._available

    def health_check(self) -> ProviderHealth:
        if not self._available:
            return ProviderHealth(
                provider_name=self.name,
                is_available=False,
                last_check=datetime.now(),
                last_error="API_FOOTBALL_KEY not set",
            )

        try:
            data = self._request("/status", {})
            if data:
                remaining = data.get("response", {}).get("requests", {}).get("remaining", None)
                return ProviderHealth(
                    provider_name=self.name,
                    is_available=True,
                    last_check=datetime.now(),
                    last_sync=self._last_sync,
                    error_count=self._error_count,
                    api_calls_remaining=remaining,
                )
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)

        return ProviderHealth(
            provider_name=self.name,
            is_available=False,
            last_check=datetime.now(),
            last_sync=self._last_sync,
            error_count=self._error_count,
            last_error=self._last_error,
        )

    def _request(self, endpoint: str, params: dict) -> dict | None:
        """Make authenticated request to API-Football."""
        if not self._available:
            return None

        headers = {"x-apisports-key": self._api_key}
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
                logger.warning("API-Football errors: %s", data["errors"])
                return None
            return data
        except Exception as e:
            logger.warning("API-Football request failed: %s", e)
            self._error_count += 1
            self._last_error = str(e)
            return None

    def fetch_live_state(self, match_id: str) -> ProviderResult:
        """Fetch live match state from API-Football.

        Returns dict with:
        - status, minute, stoppage_time
        - home_score, away_score
        - home_red_cards, away_red_cards
        - home_yellow_cards, away_yellow_cards
        - home_corners, away_corners
        - home_shots, away_shots
        - home_shots_on_target, away_shots_on_target
        """
        # match_id format: "sporttery:12345" -> need provider_fixture_id
        # This should be resolved by the caller using provider_fixtures table
        return ProviderResult(
            success=False,
            error_message="Use fetch_live_state_by_fixture_id instead",
            provider=self.name,
        )

    def fetch_live_state_by_fixture_id(self, fixture_id: int) -> ProviderResult:
        """Fetch live match state using API-Football fixture ID."""
        data = self._request("/fixtures", {"id": fixture_id})

        if not data or not data.get("response"):
            return ProviderResult(
                success=False,
                error_message="No data returned from API-Football",
                provider=self.name,
            )

        fixture = data["response"][0]
        status = fixture["fixture"]["status"]
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})

        # Extract statistics if available
        statistics = fixture.get("statistics", [])
        home_stats = {}
        away_stats = {}

        for team_stats in statistics:
            team_id = team_stats.get("team", {}).get("id")
            stats_list = team_stats.get("statistics", [])

            stats_dict = {}
            for stat in stats_list:
                stats_dict[stat.get("type", "")] = stat.get("value")

            # Determine if home or away
            home_id = fixture["teams"]["home"]["id"]
            if team_id == home_id:
                home_stats = stats_dict
            else:
                away_stats = stats_dict

        # Map status
        status_short = status.get("short", "NS")
        if status_short in ("NS", "TBD", "PST", "CANC"):
            match_status = "scheduled"
        elif status_short in ("1H", "HT", "2H", "ET", "BT", "P", "INT"):
            match_status = "live"
        elif status_short == "HT":
            match_status = "halftime"
        elif status_short in ("FT", "AET", "PEN"):
            match_status = "finished"
        else:
            match_status = "unknown"

        result_data = {
            "status": match_status,
            "status_short": status_short,
            "minute": status.get("elapsed"),
            "stoppage_time": status.get("extra"),
            "home_score": goals.get("home"),
            "away_score": goals.get("away"),
            "home_halftime_score": score.get("halftime", {}).get("home"),
            "away_halftime_score": score.get("halftime", {}).get("away"),
            "home_red_cards": self._extract_card_count(home_stats.get("Red Cards")),
            "away_red_cards": self._extract_card_count(away_stats.get("Red Cards")),
            "home_yellow_cards": self._extract_card_count(home_stats.get("Yellow Cards")),
            "away_yellow_cards": self._extract_card_count(away_stats.get("Yellow Cards")),
            "home_corners": self._extract_stat(home_stats.get("Corner Kicks")),
            "away_corners": self._extract_stat(away_stats.get("Corner Kicks")),
            "home_shots": self._extract_stat(home_stats.get("Total Shots")),
            "away_shots": self._extract_stat(away_stats.get("Total Shots")),
            "home_shots_on_target": self._extract_stat(home_stats.get("Shots on Goal")),
            "away_shots_on_target": self._extract_stat(away_stats.get("Shots on Goal")),
            "home_possession": self._extract_possession(home_stats.get("Ball Possession")),
            "away_possession": self._extract_possession(away_stats.get("Ball Possession")),
        }

        self._last_sync = datetime.now()

        return ProviderResult(
            success=True,
            data=result_data,
            records_count=1,
            provider=self.name,
        )

    def fetch_results(self, match_ids: list[str] | None = None) -> ProviderResult:
        """Fetch finished match results."""
        # Get today's finished fixtures
        data = self._request("/fixtures", {"date": datetime.now().strftime("%Y-%m-%d"), "status": "FT"})

        if not data or not data.get("response"):
            return ProviderResult(
                success=False,
                error_message="No finished matches found",
                provider=self.name,
            )

        results = []
        for fixture in data["response"]:
            goals = fixture.get("goals", {})
            results.append({
                "fixture_id": fixture["fixture"]["id"],
                "home_team": fixture["teams"]["home"]["name"],
                "away_team": fixture["teams"]["away"]["name"],
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": fixture["fixture"]["status"]["short"],
            })

        self._last_sync = datetime.now()

        return ProviderResult(
            success=True,
            data=results,
            records_count=len(results),
            provider=self.name,
        )

    @staticmethod
    def _extract_stat(value) -> int | None:
        """Extract integer stat value."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_card_count(value) -> int:
        """Extract card count, handling various formats."""
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_possession(value) -> float | None:
        """Extract possession percentage."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                return float(value.replace("%", ""))
            return float(value)
        except (TypeError, ValueError):
            return None
