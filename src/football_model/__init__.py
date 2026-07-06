"""Local football probability analysis package."""

from .engine import (
    MarketSummary,
    estimate_expected_goals,
    infer_expected_goals_from_market,
    market_comparison,
    score_matrix,
    summarize_market,
)

__all__ = [
    "MarketSummary",
    "estimate_expected_goals",
    "infer_expected_goals_from_market",
    "market_comparison",
    "score_matrix",
    "summarize_market",
]
