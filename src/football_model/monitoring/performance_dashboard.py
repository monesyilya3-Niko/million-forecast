"""Model performance dashboard for tracking prediction quality.

Provides real-time tracking of model accuracy, calibration,
and profitability metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance metrics for a model."""

    model_name: str
    period: str  # "daily", "weekly", "monthly", "all_time"

    # Accuracy metrics
    total_predictions: int
    correct_predictions: int
    accuracy: float

    # Probability metrics
    log_loss: float
    brier_score: float
    ece: float  # Expected Calibration Error

    # Outcome breakdown
    home_win_accuracy: float
    draw_accuracy: float
    away_win_accuracy: float

    # Profitability (if odds available)
    total_bets: int
    winning_bets: int
    losing_bets: int
    roi: float
    yield_pct: float
    profit: float

    # Trends
    accuracy_trend: float  # Change vs previous period
    roi_trend: float


class PerformanceDashboard:
    """Track and analyze model performance over time."""

    def __init__(
        self,
        prediction_history: pd.DataFrame,
        odds_history: pd.DataFrame | None = None,
    ) -> None:
        """Initialize dashboard.

        Args:
            prediction_history: DataFrame with columns:
                - timestamp, match_id, model_name, prediction (H/D/A),
                  probability_home, probability_draw, probability_away,
                  actual_result (H/D/A)
            odds_history: Optional DataFrame with betting odds
        """
        self.predictions = prediction_history
        self.odds = odds_history

    def compute_metrics(
        self,
        model_name: str,
        period: str = "all_time",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Compute performance metrics for a model.

        Args:
            model_name: Name of the model
            period: Time period
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            PerformanceMetrics
        """
        # Filter by model
        model_preds = self.predictions[self.predictions["model_name"] == model_name].copy()

        # Filter by date
        if start_date:
            model_preds = model_preds[model_preds["timestamp"] >= start_date]
        if end_date:
            model_preds = model_preds[model_preds["timestamp"] <= end_date]

        if model_preds.empty:
            return self._empty_metrics(model_name, period)

        # Map outcomes
        outcome_map = {"H": 0, "D": 1, "A": 2}

        # Extract predictions and actuals
        predictions = model_preds["prediction"].map(outcome_map).values
        actuals = model_preds["actual_result"].map(outcome_map).values

        # Probabilities
        proba = model_preds[["probability_home", "probability_draw", "probability_away"]].values

        # Accuracy
        correct = (predictions == actuals).sum()
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0

        # Outcome-specific accuracy
        home_mask = actuals == 0
        draw_mask = actuals == 1
        away_mask = actuals == 2

        home_win_accuracy = float((predictions[home_mask] == actuals[home_mask]).mean()) if home_mask.sum() > 0 else 0
        draw_accuracy = float((predictions[draw_mask] == actuals[draw_mask]).mean()) if draw_mask.sum() > 0 else 0
        away_win_accuracy = float((predictions[away_mask] == actuals[away_mask]).mean()) if away_mask.sum() > 0 else 0

        # Log loss
        from sklearn.metrics import log_loss

        try:
            ll = log_loss(actuals, proba, labels=[0, 1, 2])
        except ValueError:
            ll = float("nan")

        # Brier score
        one_hot = np.eye(3)[actuals]
        brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

        # ECE
        ece = self._compute_ece(proba, actuals)

        # Profitability
        total_bets, winning_bets, losing_bets, roi, yield_pct, profit = self._compute_profitability(model_preds)

        # Trends
        accuracy_trend, roi_trend = self._compute_trends(model_name, period)

        return PerformanceMetrics(
            model_name=model_name,
            period=period,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=float(accuracy),
            log_loss=float(ll),
            brier_score=brier,
            ece=ece,
            home_win_accuracy=home_win_accuracy,
            draw_accuracy=draw_accuracy,
            away_win_accuracy=away_win_accuracy,
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            roi=roi,
            yield_pct=yield_pct,
            profit=profit,
            accuracy_trend=accuracy_trend,
            roi_trend=roi_trend,
        )

    def _compute_ece(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        correct = (predictions == outcomes).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()
            bin_size = mask.sum()

            ece += bin_size * abs(bin_accuracy - bin_confidence)

        ece /= len(probabilities)
        return float(ece)

    def _compute_profitability(
        self,
        predictions: pd.DataFrame,
    ) -> tuple[int, int, int, float, float, float]:
        """Compute profitability metrics."""
        if self.odds is None or self.odds.empty:
            return 0, 0, 0, 0.0, 0.0, 0.0

        # Merge predictions with odds
        merged = predictions.merge(
            self.odds,
            on=["match_id", "timestamp"],
            how="inner",
        )

        if merged.empty:
            return 0, 0, 0, 0.0, 0.0, 0.0

        # Simulate flat betting on highest confidence prediction
        total_bets = len(merged)
        winning_bets = 0
        total_profit = 0.0

        for _, row in merged.iterrows():
            # Get odds for predicted outcome
            if row["prediction"] == "H":
                odds = row.get("odds_home", 0)
            elif row["prediction"] == "D":
                odds = row.get("odds_draw", 0)
            else:
                odds = row.get("odds_away", 0)

            if odds <= 1:
                continue

            # Check if bet won
            if row["prediction"] == row["actual_result"]:
                winning_bets += 1
                total_profit += (odds - 1) * 100  # Assuming 100 unit stake
            else:
                total_profit -= 100

        losing_bets = total_bets - winning_bets
        roi = total_profit / (total_bets * 100) if total_bets > 0 else 0
        yield_pct = total_profit / total_bets if total_bets > 0 else 0

        return total_bets, winning_bets, losing_bets, float(roi), float(yield_pct), float(total_profit)

    def _compute_trends(
        self,
        model_name: str,
        period: str,
    ) -> tuple[float, float]:
        """Compute trends vs previous period."""
        # Simplified: return 0 for now
        return 0.0, 0.0

    def _empty_metrics(self, model_name: str, period: str) -> PerformanceMetrics:
        """Return empty metrics."""
        return PerformanceMetrics(
            model_name=model_name,
            period=period,
            total_predictions=0,
            correct_predictions=0,
            accuracy=0.0,
            log_loss=float("nan"),
            brier_score=0.0,
            ece=0.0,
            home_win_accuracy=0.0,
            draw_accuracy=0.0,
            away_win_accuracy=0.0,
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            roi=0.0,
            yield_pct=0.0,
            profit=0.0,
            accuracy_trend=0.0,
            roi_trend=0.0,
        )


def generate_performance_report(metrics: PerformanceMetrics) -> str:
    """Generate a human-readable performance report."""
    report = f"""
=== 模型性能报告: {metrics.model_name} ===
统计周期: {metrics.period}

--- 准确率 ---
总预测数: {metrics.total_predictions}
正确预测: {metrics.correct_predictions}
准确率: {metrics.accuracy:.1%}

--- 概率质量 ---
Log Loss: {metrics.log_loss:.4f}
Brier Score: {metrics.brier_score:.4f}
ECE: {metrics.ece:.4f}

--- 分类准确率 ---
主胜准确率: {metrics.home_win_accuracy:.1%}
平局准确率: {metrics.draw_accuracy:.1%}
客胜准确率: {metrics.away_win_accuracy:.1%}
"""

    if metrics.total_bets > 0:
        report += f"""
--- 盈利能力 ---
总投注: {metrics.total_bets}
赢: {metrics.winning_bets} | 输: {metrics.losing_bets}
胜率: {metrics.winning_bets / metrics.total_bets:.1%}
ROI: {metrics.roi:.2%}
收益率: {metrics.yield_pct:.2f}%
利润: {metrics.profit:+.2f}
"""

    return report
