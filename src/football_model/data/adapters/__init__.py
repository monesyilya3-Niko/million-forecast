"""External data source adapters."""

from .football_data import FootballDataCsvAdapter
from .lineups import ApiFootballAdapter
from .odds_aggregator import OddsAggregator
from .sporttery import SportteryAdapter, SportterySnapshot
from .weather import WeatherAdapter

__all__ = [
    "ApiFootballAdapter",
    "FootballDataCsvAdapter",
    "OddsAggregator",
    "SportteryAdapter",
    "SportterySnapshot",
    "WeatherAdapter",
]
