"""Probability calibration using temperature scaling.

Temperature scaling is a simple post-hoc calibration method that
adjusts model probabilities to better match observed frequencies.

A temperature T > 1 softens the distribution (more uniform),
T < 1 sharpens it (more confident).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    temperature: float
    ece_before: float  # Expected Calibration Error before
    ece_after: float   # Expected Calibration Error after
    n_samples: int


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probability array.

    Args:
        probs: Array of shape (n, 3) with [home, draw, away] probabilities
        temperature: Scaling factor

    Returns:
        Calibrated probabilities
    """
    if temperature <= 0:
        return probs

    # Convert to logits
    logits = np.log(np.clip(probs, 1e-10, 1.0))
    scaled_logits = logits / temperature
    # Softmax
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return calibrated


def compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: Predicted probabilities (n, 3)
        outcomes: True outcomes (0=home, 1=draw, 2=away)
        n_bins: Number of calibration bins

    Returns:
        ECE score (lower is better)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == outcomes).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return float(ece)


def find_optimal_temperature(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> CalibrationResult:
    """Find optimal temperature for calibration.

    Args:
        probs: Model probabilities (n, 3)
        outcomes: True outcomes (0=home, 1=draw, 2=away)

    Returns:
        CalibrationResult with optimal temperature
    """
    ece_before = compute_ece(probs, outcomes)

    def objective(t: float) -> float:
        if t <= 0.01:
            return 1e6
        calibrated = temperature_scale(probs, t)
        return compute_ece(calibrated, outcomes)

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    optimal_t = result.x

    calibrated = temperature_scale(probs, optimal_t)
    ece_after = compute_ece(calibrated, outcomes)

    logger.info(
        "Calibration: T=%.3f, ECE before=%.4f, after=%.4f",
        optimal_t, ece_before, ece_after,
    )

    return CalibrationResult(
        temperature=float(optimal_t),
        ece_before=float(ece_before),
        ece_after=float(ece_after),
        n_samples=len(probs),
    )


def apply_calibration(
    probs: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply pre-computed temperature to new predictions.

    Args:
        probs: Raw model probabilities (n, 3)
        temperature: Pre-computed optimal temperature

    Returns:
        Calibrated probabilities
    """
    return temperature_scale(probs, temperature)


class ProbabilityCalibrator:
    """Calibrate model probabilities using temperature scaling.

    Usage:
        calibrator = ProbabilityCalibrator()
        calibrator.fit(probs, outcomes)
        calibrated = calibrator.calibrate(new_probs)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature
        self.is_fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> CalibrationResult:
        """Find optimal temperature from validation data."""
        result = find_optimal_temperature(probs, outcomes)
        self.temperature = result.temperature
        self.is_fitted = True
        return result

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions."""
        if not self.is_fitted:
            return probs
        return temperature_scale(probs, self.temperature)

    def save(self, path: str) -> None:
        """Save calibration parameters."""
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps({"temperature": self.temperature}))

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        """Load calibration parameters."""
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        cal = cls(temperature=data.get("temperature", 1.0))
        cal.is_fitted = True
        return cal
