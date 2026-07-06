"""Walk-forward backtesting system for football prediction models.

Implements chronological train/test splits with expanding or rolling windows,
computing out-of-sample performance metrics and detecting model drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Results from a walk-forward backtest."""

    model_name: str
    competition: str
    total_matches: int
    windows: int

    # Aggregate metrics
    avg_log_loss: float
    avg_brier: float
    avg_accuracy: float

    # Per-window metrics (for drift detection)
    window_log_loss: list[float]
    window_brier: list[float]
    window_accuracy: list[float]

    # Calibration
    calibration_bins: pd.DataFrame

    # Profitability (if odds available)
    roi: float | None = None
    yield_pct: float | None = None

    # Drift detection
    drift_detected: bool = False
    drift_p_value: float | None = None


class WalkForwardBacktester:
    """Walk-forward backtesting for football prediction models.

    Implements expanding window or rolling window backtesting with
    strict chronological ordering to prevent data leakage.
    """

    def __init__(
        self,
        initial_train_size: int = 200,
        test_size: int = 50,
        step_size: int = 50,
        window_type: str = "expanding",  # "expanding" or "rolling"
        drift_threshold: float = 0.05,  # p-value for drift detection
    ) -> None:
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.window_type = window_type
        self.drift_threshold = drift_threshold

    def run(
        self,
        model_name: str,
        competition: str,
        matches: pd.DataFrame,
        predictions: pd.DataFrame,
        odds: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            model_name: Name of the model
            competition: Competition name
            matches: Historical matches (sorted by kickoff)
            predictions: Model predictions (must match matches index)
            odds: Optional odds data for profitability calculation

        Returns:
            BacktestResult with metrics
        """
        # Validate inputs
        required_matches = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        required_predictions = {"home_win", "draw", "away_win"}

        missing_matches = required_matches - set(matches.columns)
        if missing_matches:
            raise ValueError(f"比赛数据缺少字段：{', '.join(sorted(missing_matches))}")

        missing_predictions = required_predictions - set(predictions.columns)
        if missing_predictions:
            raise ValueError(f"预测数据缺少字段：{', '.join(sorted(missing_predictions))}")

        if len(matches) != len(predictions):
            raise ValueError("比赛与预测行数必须一致")
        # Preserve row pairing before chronological sorting.
        paired = matches.reset_index(drop=True).copy()
        prediction_columns = ["home_win", "draw", "away_win"]
        for column in prediction_columns:
            paired[f"_prediction_{column}"] = predictions.reset_index(drop=True)[column].to_numpy()
        paired = paired.sort_values("kickoff").reset_index(drop=True)
        matches = paired.drop(columns=[f"_prediction_{column}" for column in prediction_columns])
        predictions = paired[[f"_prediction_{column}" for column in prediction_columns]].rename(
            columns={f"_prediction_{column}": column for column in prediction_columns}
        )

        # Create outcome labels
        outcomes = []
        for _, row in matches.iterrows():
            if row["home_goals"] > row["away_goals"]:
                outcomes.append(0)  # Home win
            elif row["home_goals"] == row["away_goals"]:
                outcomes.append(1)  # Draw
            else:
                outcomes.append(2)  # Away win
        outcomes = np.array(outcomes)

        # Extract probabilities
        probabilities = predictions[["home_win", "draw", "away_win"]].to_numpy()
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        # Run walk-forward validation
        n = len(matches)
        window_metrics = []

        for start in range(self.initial_train_size, n - self.test_size + 1, self.step_size):
            end = min(start + self.test_size, n)

            # Get test indices
            test_indices = list(range(start, end))

            # Compute metrics for this window
            y_true = outcomes[test_indices]
            y_proba = probabilities[test_indices]
            y_pred = y_proba.argmax(axis=1)

            # Log loss
            try:
                window_log_loss = log_loss(y_true, y_proba, labels=[0, 1, 2])
            except ValueError:
                window_log_loss = float("nan")

            # Brier score (one-vs-rest)
            window_brier = float(np.mean(np.sum((y_proba - np.eye(3)[y_true]) ** 2, axis=1)))

            # Accuracy
            window_accuracy = float((y_pred == y_true).mean())

            window_metrics.append(
                {
                    "start": start,
                    "end": end,
                    "log_loss": window_log_loss,
                    "brier": window_brier,
                    "accuracy": window_accuracy,
                    "matches": len(test_indices),
                }
            )

        # Aggregate metrics
        metrics_df = pd.DataFrame(window_metrics)

        avg_log_loss = float(metrics_df["log_loss"].mean())
        avg_brier = float(metrics_df["brier"].mean())
        avg_accuracy = float(metrics_df["accuracy"].mean())

        # Calibration analysis
        calibration = self._compute_calibration(outcomes, probabilities)

        # Drift detection
        drift_detected, drift_p_value = self._detect_drift(metrics_df)

        # Profitability (if odds available)
        roi, yield_pct = None, None
        if odds is not None:
            roi, yield_pct = self._compute_profitability(outcomes, probabilities, odds, matches.index)

        logger.info(
            f"Backtest complete: {len(window_metrics)} windows, "
            f"avg_log_loss={avg_log_loss:.4f}, avg_accuracy={avg_accuracy:.3f}"
        )

        return BacktestResult(
            model_name=model_name,
            competition=competition,
            total_matches=n,
            windows=len(window_metrics),
            avg_log_loss=avg_log_loss,
            avg_brier=avg_brier,
            avg_accuracy=avg_accuracy,
            window_log_loss=metrics_df["log_loss"].tolist(),
            window_brier=metrics_df["brier"].tolist(),
            window_accuracy=metrics_df["accuracy"].tolist(),
            calibration_bins=calibration,
            roi=roi,
            yield_pct=yield_pct,
            drift_detected=drift_detected,
            drift_p_value=drift_p_value,
        )

    def run_retraining(
        self,
        model_name: str,
        competition: str,
        matches: pd.DataFrame,
        trainer: Callable[[pd.DataFrame], object],
        predictor: Callable[[object, pd.DataFrame], pd.DataFrame],
    ) -> BacktestResult:
        """Run a genuine walk-forward test, retraining in every window."""
        ordered = matches.sort_values("kickoff").reset_index(drop=True)
        window_metrics: list[dict[str, float]] = []
        all_outcomes: list[int] = []
        all_probabilities: list[np.ndarray] = []
        for start in range(
            self.initial_train_size,
            len(ordered) - self.test_size + 1,
            self.step_size,
        ):
            train_start = 0 if self.window_type == "expanding" else max(0, start - self.initial_train_size)
            train = ordered.iloc[train_start:start].copy()
            test = ordered.iloc[start : start + self.test_size].copy()
            model = trainer(train)
            predicted = predictor(model, test)
            required = ["home_win", "draw", "away_win"]
            if len(predicted) != len(test) or not set(required).issubset(predicted.columns):
                raise ValueError("predictor必须为测试窗口逐场返回home_win/draw/away_win")
            probabilities = predicted[required].to_numpy(dtype=float, copy=True)
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            outcomes = np.where(
                test["home_goals"].to_numpy() > test["away_goals"].to_numpy(),
                0,
                np.where(test["home_goals"].to_numpy() == test["away_goals"].to_numpy(), 1, 2),
            )
            window_metrics.append(
                {
                    "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
                    "brier": float(np.mean(np.sum((probabilities - np.eye(3)[outcomes]) ** 2, axis=1))),
                    "accuracy": float((probabilities.argmax(axis=1) == outcomes).mean()),
                }
            )
            all_outcomes.extend(outcomes.tolist())
            all_probabilities.extend(probabilities)
        if not window_metrics:
            raise ValueError("数据不足以形成步进回测窗口")
        metrics = pd.DataFrame(window_metrics)
        outcomes_array = np.asarray(all_outcomes)
        probability_array = np.asarray(all_probabilities)
        drift_detected, drift_p_value = self._detect_drift(metrics)
        return BacktestResult(
            model_name=model_name,
            competition=competition,
            total_matches=len(all_outcomes),
            windows=len(window_metrics),
            avg_log_loss=float(metrics["log_loss"].mean()),
            avg_brier=float(metrics["brier"].mean()),
            avg_accuracy=float(metrics["accuracy"].mean()),
            window_log_loss=metrics["log_loss"].tolist(),
            window_brier=metrics["brier"].tolist(),
            window_accuracy=metrics["accuracy"].tolist(),
            calibration_bins=self._compute_calibration(outcomes_array, probability_array),
            drift_detected=drift_detected,
            drift_p_value=drift_p_value,
        )

    def _compute_calibration(
        self,
        outcomes: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """Compute calibration bins."""
        # Use max probability as confidence
        confidence = probabilities.max(axis=1)
        correct = (probabilities.argmax(axis=1) == outcomes).astype(int)

        # Create bins
        bins = pd.cut(confidence, bins=np.linspace(0, 1, 11), include_lowest=True)

        calibration = (
            pd.DataFrame({"confidence": confidence, "correct": correct, "bin": bins})
            .groupby("bin", observed=True)
            .agg(
                predicted_prob=("confidence", "mean"),
                actual_rate=("correct", "mean"),
                count=("correct", "size"),
            )
            .reset_index()
        )

        return calibration

    def _detect_drift(self, metrics_df: pd.DataFrame) -> tuple[bool, float | None]:
        """Detect model drift using statistical test."""
        if len(metrics_df) < 4:
            return False, None

        # Split into first half and second half
        mid = len(metrics_df) // 2
        first_half = metrics_df["log_loss"].iloc[:mid]
        second_half = metrics_df["log_loss"].iloc[mid:]
        if first_half.std(ddof=0) < 1e-12 and second_half.std(ddof=0) < 1e-12:
            return False, None

        # Welch's t-test (unequal variances)
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(
            first_half.dropna(),
            second_half.dropna(),
            equal_var=False,
        )

        # Drift if second half is significantly worse (higher log loss)
        drift_detected = (p_value < self.drift_threshold) and (second_half.mean() > first_half.mean())

        return drift_detected, float(p_value)

    def _compute_profitability(
        self,
        outcomes: np.ndarray,
        probabilities: np.ndarray,
        odds: pd.DataFrame,
        indices: pd.Index,
    ) -> tuple[float, float]:
        """Compute ROI and yield if odds are available."""
        required = {"odds_home", "odds_draw", "odds_away"}
        if not required.issubset(odds.columns) or len(odds) != len(probabilities):
            return 0.0, 0.0
        selected = probabilities.argmax(axis=1)
        prices = odds[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        profits = np.where(selected == outcomes, prices[np.arange(len(selected)), selected] - 1, -1.0)
        return float(profits.mean()), float(profits.mean() * 100)


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a human-readable backtest report."""
    report = f"""
=== 回测报告: {result.model_name} ===
赛事: {result.competition}
总比赛数: {result.total_matches}
回测窗口数: {result.windows}

--- 核心指标 ---
平均 Log Loss: {result.avg_log_loss:.4f}
平均 Brier Score: {result.avg_brier:.4f}
平均准确率: {result.avg_accuracy:.1%}

--- 稳定性 ---
Log Loss 标准差: {np.std(result.window_log_loss):.4f}
准确率标准差: {np.std(result.window_accuracy):.4f}
"""

    if result.drift_detected:
        report += f"\n⚠️ 检测到模型漂移 (p={result.drift_p_value:.4f})\n"
    else:
        report += f"\n✅ 未检测到模型漂移 (p={result.drift_p_value:.4f})\n"

    if result.roi is not None:
        report += f"\n--- 盈利能力 ---\nROI: {result.roi:.2%}\n收益率: {result.yield_pct:.2%}\n"

    report += "\n--- 校准表 ---\n"
    report += result.calibration_bins.to_string(index=False)

    return report
