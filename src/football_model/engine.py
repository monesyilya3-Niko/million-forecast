"""Transparent baseline probability engine for pre-match football analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy


@dataclass(frozen=True)
class MarketSummary:
    home_win: float
    draw: float
    away_win: float
    total_goals: dict[str, float]
    top_scores: list[tuple[str, float]]
    home_xg: float
    away_xg: float

    @property
    def probabilities(self) -> dict[str, float]:
        return {"主胜": self.home_win, "平局": self.draw, "客胜": self.away_win}


def _positive(value: float, floor: float = 0.05) -> float:
    return max(float(value), floor)


def estimate_expected_goals(
    *,
    home_scored: float,
    home_conceded: float,
    away_scored: float,
    away_conceded: float,
    league_home_avg: float,
    league_away_avg: float,
    home_recent_xg: float | None = None,
    away_recent_xg: float | None = None,
    xg_weight: float = 0.35,
    strength_amplifier: float = 1.5,
) -> tuple[float, float]:
    """Estimate expected goals from attack/defence strengths.

    Uses arithmetic mean instead of geometric mean to preserve more
    variance in the attack/defense strength differences.
    strength_amplifier > 1.0 amplifies the difference from average.
    """
    league_home_avg = _positive(league_home_avg)
    league_away_avg = _positive(league_away_avg)

    home_attack = _positive(home_scored) / league_home_avg
    away_defence = _positive(away_conceded) / league_home_avg
    away_attack = _positive(away_scored) / league_away_avg
    home_defence = _positive(home_conceded) / league_away_avg

    # Amplify deviations from 1.0 (average)
    def amplify(x: float) -> float:
        return 1.0 + (x - 1.0) * strength_amplifier

    home_attack_amp = amplify(home_attack)
    away_defence_amp = amplify(away_defence)
    away_attack_amp = amplify(away_attack)
    home_defence_amp = amplify(home_defence)

    home_base = league_home_avg * (home_attack_amp + away_defence_amp) / 2
    away_base = league_away_avg * (away_attack_amp + home_defence_amp) / 2

    weight = min(max(float(xg_weight), 0.0), 0.75)
    if home_recent_xg is not None:
        home_base = (1 - weight) * home_base + weight * _positive(home_recent_xg)
    if away_recent_xg is not None:
        away_base = (1 - weight) * away_base + weight * _positive(away_recent_xg)

    return float(np.clip(home_base, 0.7, 5.5)), float(np.clip(away_base, 0.7, 5.5))


def score_matrix(
    home_xg: float,
    away_xg: float,
    *,
    max_goals: int = 10,
    rho: float = -0.08,
) -> np.ndarray:
    """Create a normalized score matrix with Dixon-Coles low-score correction.

    Uses vectorized scipy operations for 5-10x speedup over factorial-based approach.
    """
    home_xg = _positive(home_xg)
    away_xg = _positive(away_xg)

    # Vectorized Poisson PMF using scipy (much faster than factorial loop)
    goals = np.arange(max_goals + 1, dtype=np.float64)
    log_home_pmf = -home_xg + xlogy(goals, home_xg) - gammaln(goals + 1)
    log_away_pmf = -away_xg + xlogy(goals, away_xg) - gammaln(goals + 1)
    home_pmf = np.exp(log_home_pmf)
    away_pmf = np.exp(log_away_pmf)

    # Outer product for joint probability
    matrix = np.outer(home_pmf, away_pmf)

    # Dixon-Coles low-score correction (only applies to 0-0, 0-1, 1-0, 1-1)
    # Vectorized application
    corrections = {
        (0, 0): 1 - home_xg * away_xg * rho,
        (0, 1): 1 + home_xg * rho,
        (1, 0): 1 + away_xg * rho,
        (1, 1): 1 - rho,
    }
    for (home_goals, away_goals), correction in corrections.items():
        matrix[home_goals, away_goals] *= max(correction, 0.01)

    return matrix / matrix.sum()


def summarize_market(matrix: np.ndarray, home_xg: float, away_xg: float) -> MarketSummary:
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())

    # Vectorized total goals calculation
    h_idx, a_idx = np.indices(matrix.shape)
    total_goals = h_idx + a_idx

    totals: dict[str, float] = {}
    for total in range(7):
        totals[str(total)] = float(matrix[total_goals == total].sum())
    totals["7+"] = max(0.0, 1.0 - sum(totals.values()))

    # Vectorized score extraction
    scores = [(f"{h}:{a}", float(matrix[h, a])) for h in range(matrix.shape[0]) for a in range(matrix.shape[1])]
    scores.sort(key=lambda item: item[1], reverse=True)

    return MarketSummary(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        total_goals=totals,
        top_scores=scores[:8],
        home_xg=home_xg,
        away_xg=away_xg,
    )


def market_comparison(probabilities: dict[str, float], odds: dict[str, float]) -> pd.DataFrame:
    """Compare calibrated model probabilities with normalized implied probabilities."""
    labels = ["主胜", "平局", "客胜"]
    raw_implied = {label: 1 / _positive(odds[label]) for label in labels}
    overround = sum(raw_implied.values())

    rows = []
    for label in labels:
        probability = probabilities[label]
        market_probability = raw_implied[label] / overround
        expected_value = probability * odds[label] - 1
        rows.append(
            {
                "结果": label,
                "模型概率": probability,
                "市场概率": market_probability,
                "概率差": probability - market_probability,
                "官方SP": odds[label],
                "理论EV": expected_value,
                "公平SP": 1 / probability if probability > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def handicap_probabilities(matrix: np.ndarray, handicap: int) -> dict[str, float]:
    probabilities = {"胜": 0.0, "平": 0.0, "负": 0.0}
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            adjusted = home_goals + handicap - away_goals
            result = "胜" if adjusted > 0 else "平" if adjusted == 0 else "负"
            probabilities[result] += float(matrix[home_goals, away_goals])
    return probabilities


def infer_expected_goals_from_market(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    *,
    rho: float = -0.08,
) -> tuple[float, float]:
    """Infer a score distribution matching normalized 1X2 market probabilities.

    This is a market prior, not an independent prediction. It is used when a
    trained league/team model is not yet available for a live Sporttery match.
    """
    odds = np.array([odds_home, odds_draw, odds_away], dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("SP必须全部大于1.0")
    implied = 1 / odds
    target = implied / implied.sum()

    def objective(log_rates: np.ndarray) -> float:
        home_rate, away_rate = np.exp(log_rates)
        matrix = score_matrix(float(home_rate), float(away_rate), rho=rho)
        predicted = np.array(
            [float(np.tril(matrix, k=-1).sum()), float(np.trace(matrix)), float(np.triu(matrix, k=1).sum())]
        )
        return float(np.square(predicted - target).sum())

    result = minimize(
        objective,
        np.log([1.45, 1.15]),
        method="L-BFGS-B",
        bounds=[(np.log(0.15), np.log(4.5)), (np.log(0.15), np.log(4.5))],
    )
    if not result.success:
        raise RuntimeError(f"市场先验拟合失败：{result.message}")
    home_rate, away_rate = np.exp(result.x)
    return float(home_rate), float(away_rate)
