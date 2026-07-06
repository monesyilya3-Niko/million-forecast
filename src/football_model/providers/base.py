"""Base provider interface for external data sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderHealth:
    """Health status of a provider."""

    provider_name: str
    is_available: bool
    last_check: datetime
    last_sync: datetime | None = None
    error_count: int = 0
    last_error: str | None = None
    api_calls_remaining: int | None = None


@dataclass(frozen=True)
class ProviderResult:
    """Result from a provider operation."""

    success: bool
    data: dict | list | pd.DataFrame | None = None
    records_count: int = 0
    error_message: str | None = None
    provider: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)


class BaseProvider(ABC):
    """Abstract base class for all data providers.

    Each provider must implement:
    - health_check(): Return provider health status
    - name: Provider identifier string
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Check if the provider is available and healthy."""
        ...

    def fetch_fixtures(self, date: str | None = None) -> ProviderResult:
        """Fetch fixtures/matches. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_fixtures",
            provider=self.name,
        )

    def fetch_live_state(self, match_id: str) -> ProviderResult:
        """Fetch live match state. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_live_state",
            provider=self.name,
        )

    def fetch_odds(self, match_id: str) -> ProviderResult:
        """Fetch odds for a match. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_odds",
            provider=self.name,
        )

    def fetch_lineups(self, match_id: str) -> ProviderResult:
        """Fetch lineups for a match. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_lineups",
            provider=self.name,
        )

    def fetch_injuries(self, match_id: str) -> ProviderResult:
        """Fetch injuries for a match. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_injuries",
            provider=self.name,
        )

    def fetch_previous_matches(self, team_name: str, limit: int = 5) -> ProviderResult:
        """Fetch previous matches for a team. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_previous_matches",
            provider=self.name,
        )

    def fetch_results(self, match_ids: list[str] | None = None) -> ProviderResult:
        """Fetch match results. Override if supported."""
        return ProviderResult(
            success=False,
            error_message=f"{self.name} does not support fetch_results",
            provider=self.name,
        )
