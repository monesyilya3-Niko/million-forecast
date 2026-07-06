"""In-memory normalization layer for odds supplied by configured adapters.

This module does not scrape or fetch a bookmaker. AH/OU become available only
after a licensed provider adapter supplies those records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketOdds:
    """Odds for a single market."""

    match_id: str
    market: str  # "HAD", "HHAD", "AH", "OU", "TTG"
    selection: str
    odds: float
    line: str | None = None  # Handicap line or total line
    source: str = "unknown"
    captured_at: pd.Timestamp | None = None


class OddsAggregator:
    """Aggregate odds from multiple sources.

    Provides a unified interface for accessing odds across
    different markets and data sources.
    """

    SUPPORTED_MARKETS = {
        "HAD": "胜平负",
        "HHAD": "让球胜平负",
        "AH": "亚洲盘让球",
        "OU": "大小球",
        "TTG": "总进球数",
        "CRS": "比分",
        "HAFU": "半全场",
    }

    def __init__(self) -> None:
        self._odds_cache: dict[str, list[MarketOdds]] = {}

    def add_odds(self, odds: MarketOdds) -> None:
        """Add odds to the cache."""
        key = f"{odds.match_id}:{odds.market}"
        if key not in self._odds_cache:
            self._odds_cache[key] = []
        self._odds_cache[key].append(odds)

    def get_latest_odds(
        self,
        match_id: str,
        market: str,
    ) -> MarketOdds | None:
        """Get latest odds for a match and market."""
        key = f"{match_id}:{market}"
        odds_list = self._odds_cache.get(key, [])
        if not odds_list:
            return None
        return max(odds_list, key=lambda x: x.captured_at or pd.Timestamp.min)

    def get_all_odds(self, match_id: str) -> dict[str, MarketOdds]:
        """Get latest odds for all markets of a match."""
        result = {}
        for market in self.SUPPORTED_MARKETS:
            odds = self.get_latest_odds(match_id, market)
            if odds:
                result[market] = odds
        return result

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load odds from a DataFrame.

        Expected columns: match_id, market, selection, odds, [line], [source], [captured_at]
        """
        for _, row in df.iterrows():
            odds = MarketOdds(
                match_id=row["match_id"],
                market=row["market"],
                selection=row["selection"],
                odds=float(row["odds"]),
                line=row.get("line"),
                source=row.get("source", "unknown"),
                captured_at=pd.Timestamp(row["captured_at"]) if "captured_at" in row else None,
            )
            self.add_odds(odds)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all cached odds to a DataFrame."""
        rows = []
        for odds_list in self._odds_cache.values():
            for odds in odds_list:
                rows.append(
                    {
                        "match_id": odds.match_id,
                        "market": odds.market,
                        "selection": odds.selection,
                        "odds": odds.odds,
                        "line": odds.line,
                        "source": odds.source,
                        "captured_at": odds.captured_at,
                    }
                )
        return pd.DataFrame(rows)


def compute_implied_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> dict[str, float]:
    """Compute implied probabilities from odds (removing overround).

    Args:
        odds_home: Home win odds
        odds_draw: Draw odds
        odds_away: Away win odds

    Returns:
        Dictionary with normalized probabilities
    """
    raw_probs = {
        "home_win": 1.0 / odds_home,
        "draw": 1.0 / odds_draw,
        "away_win": 1.0 / odds_away,
    }

    # Remove overround
    total = sum(raw_probs.values())
    return {k: v / total for k, v in raw_probs.items()}


def compute_handicap_probabilities(
    home_xg: float,
    away_xg: float,
    handicap: float,
    max_goals: int = 10,
) -> dict[str, float]:
    """Compute Asian handicap probabilities.

    Args:
        home_xg: Home expected goals
        away_xg: Away expected goals
        handicap: Asian handicap line (positive = home advantage)
        max_goals: Maximum goals to consider

    Returns:
        Dictionary with home_win, push, away_win probabilities
    """
    import numpy as np
    from scipy.special import gammaln, xlogy

    goals = np.arange(max_goals + 1, dtype=np.float64)
    home_pmf = np.exp(-home_xg + xlogy(goals, home_xg) - gammaln(goals + 1))
    away_pmf = np.exp(-away_xg + xlogy(goals, away_xg) - gammaln(goals + 1))

    matrix = np.outer(home_pmf, away_pmf)
    matrix = matrix / matrix.sum()

    home_win = 0.0
    push = 0.0
    away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            adjusted = h + handicap - a
            if adjusted > 0:
                home_win += matrix[h, a]
            elif adjusted == 0:
                push += matrix[h, a]
            else:
                away_win += matrix[h, a]

    return {
        "home_win": float(home_win),
        "push": float(push),
        "away_win": float(away_win),
    }


def compute_over_under_probabilities(
    home_xg: float,
    away_xg: float,
    total_line: float,
    max_goals: int = 10,
) -> dict[str, float]:
    """Compute over/under probabilities.

    Args:
        home_xg: Home expected goals
        away_xg: Away expected goals
        total_line: Total goals line (e.g., 2.5)
        max_goals: Maximum goals to consider

    Returns:
        Dictionary with over, under probabilities
    """
    import numpy as np
    from scipy.special import gammaln, xlogy

    goals = np.arange(max_goals + 1, dtype=np.float64)
    home_pmf = np.exp(-home_xg + xlogy(goals, home_xg) - gammaln(goals + 1))
    away_pmf = np.exp(-away_xg + xlogy(goals, away_xg) - gammaln(goals + 1))

    matrix = np.outer(home_pmf, away_pmf)
    matrix = matrix / matrix.sum()

    over = 0.0
    under = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > total_line:
                over += matrix[h, a]
            else:
                under += matrix[h, a]

    return {
        "over": float(over),
        "under": float(under),
    }
