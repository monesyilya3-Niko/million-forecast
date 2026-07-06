"""Probability calibration for football prediction models.

Implements Platt Scaling and Isotonic Regression to improve
the calibration of model probabilities, ensuring that predicted
probabilities match observed frequencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityCalibrator:
    """Calibrate predicted probabilities to match observed frequencies.

    Supports Platt Scaling (logistic regression) and Isotonic Regression.
    """

    method: str  # "platt" or "isotonic"
    calibrators: dict[str, object]  # One calibrator per outcome
    trained: bool = False

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        method: str = "platt",
    ) -> ProbabilityCalibrator:
        """Fit calibrator on validation data.

        Args:
            probabilities: Predicted probabilities (n_samples, 3)
            outcomes: True outcomes (0=H, 1=D, 2=A)
            method: Calibration method ("platt" or "isotonic")

        Returns:
            Fitted ProbabilityCalibrator
        """
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method}")

        calibrators = {}

        # Fit one calibrator per outcome (one-vs-rest)
        for outcome_idx, outcome_name in enumerate(["home_win", "draw", "away_win"]):
            # Binary labels for this outcome
            y_binary = (outcomes == outcome_idx).astype(int)

            # Get predicted probabilities for this outcome
            y_pred = probabilities[:, outcome_idx]

            if method == "platt":
                # Platt Scaling: logistic regression
                calibrator = LogisticRegression(C=1.0, max_iter=1000)
                calibrator.fit(y_pred.reshape(-1, 1), y_binary)
            else:
                # Isotonic Regression
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_pred, y_binary)

            calibrators[outcome_name] = calibrator

        logger.info(f"Probability calibrator fitted ({method}), {len(probabilities)} samples")

        return cls(
            method=method,
            calibrators=calibrators,
            trained=True,
        )

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate predicted probabilities.

        Args:
            probabilities: Raw predicted probabilities (n_samples, 3)

        Returns:
            Calibrated probabilities (n_samples, 3)
        """
        if not self.trained:
            raise RuntimeError("Calibrator not fitted yet")

        calibrated = np.zeros_like(probabilities)

        for outcome_idx, outcome_name in enumerate(["home_win", "draw", "away_win"]):
            calibrator = self.calibrators[outcome_name]
            raw_probs = probabilities[:, outcome_idx]

            if self.method == "platt":
                # Platt scaling returns probabilities
                calibrated[:, outcome_idx] = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                # Isotonic regression returns calibrated values
                calibrated[:, outcome_idx] = calibrator.predict(raw_probs)

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums

        return calibrated

    def save(self, artifact_path: str | Path) -> Path:
        """Save calibrator to file."""
        import pickle

        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

        logger.info(f"Calibrator saved to {path}")
        return path

    @classmethod
    def load(cls, artifact_path: str | Path) -> ProbabilityCalibrator:
        """Load calibrator from file."""
        import pickle

        with Path(artifact_path).open("rb") as f:
            calibrator = pickle.load(f)

        return calibrator


def compute_calibration_metrics(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration metrics.

    Args:
        probabilities: Predicted probabilities (n_samples, 3)
        outcomes: True outcomes (0=H, 1=D, 2=A)
        n_bins: Number of calibration bins

    Returns:
        Dictionary with calibration metrics
    """
    # Expected Calibration Error (ECE)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == outcomes).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        bin_size = mask.sum()

        ece += bin_size * abs(bin_accuracy - bin_confidence)
        mce = max(mce, abs(bin_accuracy - bin_confidence))

    ece /= len(probabilities)

    # Brier Score
    one_hot = np.eye(3)[outcomes]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    # Log Loss
    from sklearn.metrics import log_loss

    ll = log_loss(outcomes, probabilities, labels=[0, 1, 2])

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "log_loss": float(ll),
        "accuracy": float(correct.mean()),
    }
