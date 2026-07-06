from __future__ import annotations

import logging
import pandas as pd

from football_model.models import build_match_intelligence

logger = logging.getLogger(__name__)


def _history() -> pd.DataFrame:
    """Create sample match history for testing."""
    logger.info("Creating sample match history")
    rows = []
    teams = ["Strong", "Average", "Weak", "Other"]
    for index in range(160):
        home = teams[index % 4]
        away = teams[(index + 1) % 4]
        rows.append(
            {
                "kickoff": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "home_team": home,
                "away_team": away,
                "home_goals": 3 if home == "Strong" else 1,
                "away_goals": 0 if away == "Weak" else 1,
            }
        )
    return pd.DataFrame(rows)


def test_team_intelligence_returns_form_elo_and_xg() -> None:
    result = build_match_intelligence(_history(), "Strong", "Weak", as_of="2025-01-01")
    assert result.home.matches > 20
    assert result.away.matches > 20
    assert result.home.elo > result.away.elo
    assert result.home_xg > result.away_xg
    assert result.elo_home_probability > result.elo_away_probability
    assert abs(result.elo_home_probability + result.elo_draw_probability + result.elo_away_probability - 1) < 1e-9


def test_team_intelligence_respects_cutoff() -> None:
    result = build_match_intelligence(_history(), "Strong", "Weak", as_of="2024-03-01")
    assert result.data_cutoff < pd.Timestamp("2024-03-01")
