import numpy as np

from football_model.engine import (
    estimate_expected_goals,
    handicap_probabilities,
    market_comparison,
    score_matrix,
    summarize_market,
)


def test_score_matrix_is_normalized() -> None:
    matrix = score_matrix(1.6, 1.1)
    assert matrix.shape == (11, 11)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix >= 0)


def test_market_probabilities_sum_to_one() -> None:
    matrix = score_matrix(1.6, 1.1)
    summary = summarize_market(matrix, 1.6, 1.1)
    assert np.isclose(sum(summary.probabilities.values()), 1.0)
    assert np.isclose(sum(summary.total_goals.values()), 1.0)


def test_equal_teams_still_have_home_advantage_from_league_baseline() -> None:
    home_xg, away_xg = estimate_expected_goals(
        home_scored=1.48,
        home_conceded=1.18,
        away_scored=1.18,
        away_conceded=1.48,
        league_home_avg=1.48,
        league_away_avg=1.18,
    )
    assert home_xg > away_xg


def test_market_comparison_contains_ev() -> None:
    comparison = market_comparison(
        {"主胜": 0.5, "平局": 0.3, "客胜": 0.2},
        {"主胜": 2.2, "平局": 3.1, "客胜": 4.0},
    )
    assert np.isclose(comparison.loc[0, "理论EV"], 0.1)
    assert np.isclose(comparison["市场概率"].sum(), 1.0)


def test_zero_handicap_matches_one_x_two() -> None:
    matrix = score_matrix(1.6, 1.1)
    summary = summarize_market(matrix, 1.6, 1.1)
    handicap = handicap_probabilities(matrix, 0)
    assert np.isclose(handicap["胜"], summary.home_win)
    assert np.isclose(handicap["平"], summary.draw)
    assert np.isclose(handicap["负"], summary.away_win)
