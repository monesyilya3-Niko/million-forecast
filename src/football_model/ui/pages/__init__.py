"""Application pages."""

from .backtest import render_backtest
from .batch import render_batch
from .data_center import render_data_center
from .live_matches import render_live_matches
from .match_analysis import render_match_analysis
from .match_detail import render_match_detail
from .model_center import render_model_center
from .parlay import render_parlay
from .recommendations import render_recommendations
from .results import render_results
from .single_match import render_single_match
from .system_status import render_system_status

__all__ = [
    "render_backtest",
    "render_batch",
    "render_data_center",
    "render_live_matches",
    "render_match_analysis",
    "render_match_detail",
    "render_model_center",
    "render_parlay",
    "render_recommendations",
    "render_results",
    "render_single_match",
    "render_system_status",
]
