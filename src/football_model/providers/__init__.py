"""Provider adapters for external data sources.

Each provider implements a common interface for fetching match data,
odds, lineups, injuries, and other football intelligence.
"""

from .base import BaseProvider, ProviderHealth, ProviderResult
from .live_state_provider import LiveStateProvider

__all__ = [
    "BaseProvider",
    "LiveStateProvider",
    "ProviderHealth",
    "ProviderResult",
]
