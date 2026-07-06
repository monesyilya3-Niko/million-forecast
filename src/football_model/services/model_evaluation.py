"""Model evaluation service.

Provides comprehensive model evaluation including Log Loss, Brier Score,
Calibration, ROI, CLV, and segmented performance analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration bin data."""

    bin_label: str
    predicted_prob: float
    actual_rate: float
    count: int
    gap: float  # |predicted - actual|


@dataclass(frozen=True)
class SegmentPerformance:
    """Performance in a segment."""

    segment_name: str
    segment_value: str
    matches: int
    log_loss: float
    brier_score: float
    accuracy: float
    roi: float | None = None


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Complete model evaluation result."""

    model_version: str
    competition: str
    total_matches: int

    # Core metrics
    log_loss: float
    brier_score: float
    accuracy: float
    top1_hit_rate: float

    # Calibration
    calibration_ece: float  # Expected Calibration Error
    calibration_bins: list[CalibrationBin]

    # Profitability
    roi: float
    yield_pct: float
    closing_line_value: float

    # Segmented performance
    by_league: list[SegmentPerformance]
    by_confidence: list[SegmentPerformance]
    by_odds_range: list[SegmentPerformance]

    # Overfitting check
    train_log_loss: float | None = None
    val_log_loss: float | None = None
    test_log_loss: float | None = None
    is_overfitting: bool = False

    # Drift
    drift_detected: bool = False


class ModelEvaluationService:
    """Service for evaluating model performance."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def evaluate_predictions(
        self,
        predictions: pd.DataFrame,
        outcomes: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        model_version: str = "unknown",
        competition: str = "unknown",
    ) -> ModelEvaluationResult:
        """Evaluate model predictions against actual outcomes.

        Args:
            predictions: DataFrame with columns [home_win, draw, away_win]
            outcomes: DataFrame with columns [home_goals, away_goals]
            odds: Optional DataFrame with columns [odds_home, odds_draw, odds_away]
            model_version: Model version identifier
            competition: Competition name

        Returns:
            ModelEvaluationResult with all metrics
        """
        n = len(predictions)
        if n == 0 or n != len(outcomes):
            return self._empty_result(model_version, competition)

        # Convert outcomes to labels (0=H, 1=D, 2=A)
        y_true = np.where(
            outcomes["home_goals"] > outcomes["away_goals"], 0,
            np.where(outcomes["home_goals"] == outcomes["away_goals"], 1, 2)
        )

        # Get probabilities
        proba = predictions[["home_win", "draw", "away_win"]].values
        proba = proba / proba.sum(axis=1, keepdims=True)

        # Predictions
        y_pred = proba.argmax(axis=1)

        # Core metrics
        ll = float(log_loss(y_true, proba, labels=[0, 1, 2]))
        one_hot = np.eye(3)[y_true]
        brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
        accuracy = float((y_pred == y_true).mean())
        top1_hit = accuracy  # Same as accuracy for 1X2

        # Calibration
        calibration_bins = self._compute_calibration_bins(y_true, proba)
        ece = self._compute_ece(y_true, proba)

        # Profitability
        roi, yield_pct, clv = 0.0, 0.0, 0.0
        if odds is not None and len(odds) == n:
            roi, yield_pct = self._compute_roi(y_true, y_pred, proba, odds)
            clv = self._compute_clv(proba, odds)

        return ModelEvaluationResult(
            model_version=model_version,
            competition=competition,
            total_matches=n,
            log_loss=ll,
            brier_score=brier,
            accuracy=accuracy,
            top1_hit_rate=top1_hit,
            calibration_ece=ece,
            calibration_bins=calibration_bins,
            roi=roi,
            yield_pct=yield_pct,
            closing_line_value=clv,
            by_league=[],
            by_confidence=[],
            by_odds_range=[],
        )

    def evaluate_with_segments(
        self,
        predictions: pd.DataFrame,
        outcomes: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        leagues: pd.Series | None = None,
        model_version: str = "unknown",
        competition: str = "unknown",
    ) -> ModelEvaluationResult:
        """Evaluate with segmented analysis."""
        # Base evaluation
        result = self.evaluate_predictions(predictions, outcomes, odds, model_version, competition)

        proba = predictions[["home_win", "draw", "away_win"]].values
        proba = proba / proba.sum(axis=1, keepdims=True)
        y_true = np.where(
            outcomes["home_goals"] > outcomes["away_goals"], 0,
            np.where(outcomes["home_goals"] == outcomes["away_goals"], 1, 2)
        )
        y_pred = proba.argmax(axis=1)

        # By confidence segments
        confidence_segments = self._segment_by_confidence(y_true, proba, y_pred, odds)

        # By odds range segments
        odds_segments = self._segment_by_odds_range(y_true, proba, y_pred, odds)

        # By league segments
        league_segments = []
        if leagues is not None:
            league_segments = self._segment_by_league(y_true, proba, y_pred, odds, leagues)

        return ModelEvaluationResult(
            model_version=result.model_version,
            competition=result.competition,
            total_matches=result.total_matches,
            log_loss=result.log_loss,
            brier_score=result.brier_score,
            accuracy=result.accuracy,
            top1_hit_rate=result.top1_hit_rate,
            calibration_ece=result.calibration_ece,
            calibration_bins=result.calibration_bins,
            roi=result.roi,
            yield_pct=result.yield_pct,
            closing_line_value=result.closing_line_value,
            by_league=league_segments,
            by_confidence=confidence_segments,
            by_odds_range=odds_segments,
        )

    def _compute_calibration_bins(self, y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[CalibrationBin]:
        """Compute calibration bins."""
        confidence = proba.max(axis=1)
        predictions = proba.argmax(axis=1)
        correct = (predictions == y_true).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        result = []

        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() == 0:
                continue

            bin_conf = float(confidence[mask].mean())
            bin_acc = float(correct[mask].mean())
            count = int(mask.sum())
            gap = abs(bin_conf - bin_acc)

            result.append(CalibrationBin(
                bin_label=f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                predicted_prob=bin_conf,
                actual_rate=bin_acc,
                count=count,
                gap=gap,
            ))

        return result

    def _compute_ece(self, y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        confidence = proba.max(axis=1)
        predictions = proba.argmax(axis=1)
        correct = (predictions == y_true).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() == 0:
                continue

            bin_acc = float(correct[mask].mean())
            bin_conf = float(confidence[mask].mean())
            bin_size = mask.sum()

            ece += bin_size * abs(bin_acc - bin_conf)

        return float(ece / len(y_true))

    def _compute_roi(self, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, odds: pd.DataFrame) -> tuple[float, float]:
        """Compute ROI and yield."""
        if "odds_home" not in odds.columns:
            return 0.0, 0.0

        odds_arr = odds[["odds_home", "odds_draw", "odds_away"]].values
        profits = np.where(y_pred == y_true, odds_arr[np.arange(len(y_pred)), y_pred] - 1, -1.0)
        roi = float(profits.mean())
        yield_pct = float(profits.mean() * 100)
        return roi, yield_pct

    def _compute_clv(self, proba: np.ndarray, odds: pd.DataFrame) -> float:
        """Compute Closing Line Value."""
        if "odds_home" not in odds.columns:
            return 0.0

        # Compare model probability to market implied probability
        market_implied = 1 / odds[["odds_home", "odds_draw", "odds_away"]].values
        market_implied = market_implied / market_implied.sum(axis=1, keepdims=True)

        # CLV = average edge
        edge = proba - market_implied
        return float(np.abs(edge).mean())

    def _segment_by_confidence(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        y_pred: np.ndarray,
        odds: pd.DataFrame | None,
    ) -> list[SegmentPerformance]:
        """Segment performance by confidence level."""
        confidence = proba.max(axis=1)
        segments = []

        for label, low, high in [("低置信度", 0, 0.40), ("中置信度", 0.40, 0.55), ("高置信度", 0.55, 1.0)]:
            mask = (confidence >= low) & (confidence < high)
            if mask.sum() == 0:
                continue

            seg_true = y_true[mask]
            seg_proba = proba[mask]
            seg_pred = y_pred[mask]

            ll = float(log_loss(seg_true, seg_proba, labels=[0, 1, 2])) if len(np.unique(seg_true)) > 1 else 0.0
            one_hot = np.eye(3)[seg_true]
            brier = float(np.mean(np.sum((seg_proba - one_hot) ** 2, axis=1)))
            acc = float((seg_pred == seg_true).mean())

            roi = None
            if odds is not None and "odds_home" in odds.columns:
                odds_arr = odds[["odds_home", "odds_draw", "odds_away"]].values[mask]
                profits = np.where(seg_pred == seg_true, odds_arr[np.arange(len(seg_pred)), seg_pred] - 1, -1.0)
                roi = float(profits.mean())

            segments.append(SegmentPerformance(
                segment_name="置信度",
                segment_value=label,
                matches=int(mask.sum()),
                log_loss=ll,
                brier_score=brier,
                accuracy=acc,
                roi=roi,
            ))

        return segments

    def _segment_by_odds_range(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        y_pred: np.ndarray,
        odds: pd.DataFrame | None,
    ) -> list[SegmentPerformance]:
        """Segment performance by odds range."""
        if odds is None or "odds_home" not in odds.columns:
            return []

        # Use max odds as the range indicator
        odds_arr = odds[["odds_home", "odds_draw", "odds_away"]].values
        max_odds = odds_arr.max(axis=1)

        segments = []
        for label, low, high in [("低赔(1.0-2.0)", 1.0, 2.0), ("中赔(2.0-4.0)", 2.0, 4.0), ("高赔(4.0+)", 4.0, 100.0)]:
            mask = (max_odds >= low) & (max_odds < high)
            if mask.sum() == 0:
                continue

            seg_true = y_true[mask]
            seg_proba = proba[mask]
            seg_pred = y_pred[mask]

            ll = float(log_loss(seg_true, seg_proba, labels=[0, 1, 2])) if len(np.unique(seg_true)) > 1 else 0.0
            one_hot = np.eye(3)[seg_true]
            brier = float(np.mean(np.sum((seg_proba - one_hot) ** 2, axis=1)))
            acc = float((seg_pred == seg_true).mean())

            seg_odds = odds_arr[mask]
            profits = np.where(seg_pred == seg_true, seg_odds[np.arange(len(seg_pred)), seg_pred] - 1, -1.0)
            roi = float(profits.mean())

            segments.append(SegmentPerformance(
                segment_name="赔率区间",
                segment_value=label,
                matches=int(mask.sum()),
                log_loss=ll,
                brier_score=brier,
                accuracy=acc,
                roi=roi,
            ))

        return segments

    def _segment_by_league(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        y_pred: np.ndarray,
        odds: pd.DataFrame | None,
        leagues: pd.Series,
    ) -> list[SegmentPerformance]:
        """Segment performance by league."""
        segments = []
        unique_leagues = leagues.unique()

        for league in unique_leagues:
            mask = leagues == league
            if mask.sum() < 10:
                continue

            seg_true = y_true[mask]
            seg_proba = proba[mask]
            seg_pred = y_pred[mask]

            ll = float(log_loss(seg_true, seg_proba, labels=[0, 1, 2])) if len(np.unique(seg_true)) > 1 else 0.0
            one_hot = np.eye(3)[seg_true]
            brier = float(np.mean(np.sum((seg_proba - one_hot) ** 2, axis=1)))
            acc = float((seg_pred == seg_true).mean())

            roi = None
            if odds is not None and "odds_home" in odds.columns:
                odds_arr = odds[["odds_home", "odds_draw", "odds_away"]].values[mask]
                profits = np.where(seg_pred == seg_true, odds_arr[np.arange(len(seg_pred)), seg_pred] - 1, -1.0)
                roi = float(profits.mean())

            segments.append(SegmentPerformance(
                segment_name="联赛",
                segment_value=str(league),
                matches=int(mask.sum()),
                log_loss=ll,
                brier_score=brier,
                accuracy=acc,
                roi=roi,
            ))

        return segments

    def _empty_result(self, model_version: str, competition: str) -> ModelEvaluationResult:
        """Return empty result."""
        return ModelEvaluationResult(
            model_version=model_version,
            competition=competition,
            total_matches=0,
            log_loss=float("nan"),
            brier_score=float("nan"),
            accuracy=0.0,
            top1_hit_rate=0.0,
            calibration_ece=float("nan"),
            calibration_bins=[],
            roi=0.0,
            yield_pct=0.0,
            closing_line_value=0.0,
            by_league=[],
            by_confidence=[],
            by_odds_range=[],
        )
