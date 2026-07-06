"""Ensemble model framework for football prediction.

Combines multiple models (Dixon-Coles, Poisson, XGBoost) with
dynamic weighting based on historical performance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_model.models.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Weight configuration for a single model."""

    name: str
    weight: float
    log_loss: float  # Historical performance
    last_updated: str


class EnsembleModel:
    """Ensemble of multiple football prediction models.

    Combines predictions from multiple models using dynamic weighting
    based on recent out-of-sample performance.
    """

    def __init__(
        self,
        models: dict[str, object],
        weights: dict[str, float] | None = None,
        calibrator: ProbabilityCalibrator | None = None,
        decay_factor: float = 0.95,
    ) -> None:
        """Initialize ensemble model.

        Args:
            models: Dictionary of model_name -> model
            weights: Initial weights (if None, equal weights)
            calibrator: Optional probability calibrator
            decay_factor: Decay factor for historical performance
        """
        self.models = models
        self.calibrator = calibrator
        self.decay_factor = decay_factor

        # Initialize weights
        if weights is None:
            n = len(models)
            self.weights = {name: 1.0 / n for name in models}
        else:
            self.weights = weights

        # Track performance history
        self.performance_history: dict[str, list[float]] = {name: [] for name in models}

    def predict(
        self,
        features: dict[str, float],
        odds: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Generate ensemble prediction.

        Args:
            features: Feature dictionary for the match
            odds: Optional market odds for market-aware blending

        Returns:
            Dictionary with home_win, draw, away_win probabilities
        """
        # Get predictions from each model
        model_predictions = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_probabilities"):
                    pred = model.predict_probabilities(features)
                elif hasattr(model, "expected_goals"):
                    # For Dixon-Coles style models, need to compute probabilities
                    home_xg, away_xg = model.expected_goals(
                        features.get("home_team", ""),
                        features.get("away_team", ""),
                    )
                    pred = self._xg_to_probabilities(home_xg, away_xg)
                else:
                    logger.warning(f"Model {name} has no predict method")
                    continue

                model_predictions[name] = pred
            except Exception as e:
                logger.error(f"Error getting prediction from {name}: {e}")
                continue

        if not model_predictions:
            logger.warning("No models returned predictions, using uniform")
            return {"home_win": 0.33, "draw": 0.33, "away_win": 0.33}

        # Combine predictions using weights
        ensemble = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

        for name, pred in model_predictions.items():
            weight = self.weights.get(name, 0.0)
            for outcome in ensemble:
                ensemble[outcome] += weight * pred[outcome]

        # Normalize
        total = sum(ensemble.values())
        ensemble = {k: v / total for k, v in ensemble.items()}

        # Apply calibration if available
        if self.calibrator is not None:
            prob_array = np.array([[ensemble["home_win"], ensemble["draw"], ensemble["away_win"]]])
            calibrated = self.calibrator.calibrate(prob_array)[0]
            ensemble = {
                "home_win": float(calibrated[0]),
                "draw": float(calibrated[1]),
                "away_win": float(calibrated[2]),
            }

        return ensemble

    def update_weights(
        self,
        model_name: str,
        log_loss: float,
    ) -> None:
        """Update model weight based on recent performance.

        Args:
            model_name: Name of the model
            log_loss: Log loss on recent predictions
        """
        if model_name not in self.performance_history:
            return

        # Add to history
        self.performance_history[model_name].append(log_loss)

        # Keep only recent history
        max_history = 100
        if len(self.performance_history[model_name]) > max_history:
            self.performance_history[model_name] = self.performance_history[model_name][-max_history:]

        # Recompute weights based on exponential decay
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        """Recompute weights based on historical performance."""
        # Compute weighted average log loss for each model
        avg_losses = {}
        for name, history in self.performance_history.items():
            if not history:
                avg_losses[name] = 1.0  # Default loss
                continue

            # Apply exponential decay
            weights = np.power(self.decay_factor, np.arange(len(history))[::-1])
            avg_loss = float(np.average(history, weights=weights))
            avg_losses[name] = avg_loss

        # Convert to weights (inverse of loss)
        inv_losses = {name: 1.0 / loss for name, loss in avg_losses.items()}
        total = sum(inv_losses.values())

        self.weights = {name: inv / total for name, inv in inv_losses.items()}

        logger.info(f"Ensemble weights updated: {self.weights}")

    def _xg_to_probabilities(
        self,
        home_xg: float,
        away_xg: float,
        max_goals: int = 10,
    ) -> dict[str, float]:
        """Convert expected goals to outcome probabilities."""
        from scipy.special import gammaln, xlogy

        goals = np.arange(max_goals + 1, dtype=np.float64)
        home_pmf = np.exp(-home_xg + xlogy(goals, home_xg) - gammaln(goals + 1))
        away_pmf = np.exp(-away_xg + xlogy(goals, away_xg) - gammaln(goals + 1))

        matrix = np.outer(home_pmf, away_pmf)
        matrix = matrix / matrix.sum()

        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())

        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def save(self, artifact_path: str | Path) -> Path:
        """Save ensemble configuration."""
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "weights": self.weights,
            "performance_history": {
                name: history[-50:]  # Save last 50
                for name, history in self.performance_history.items()
            },
            "decay_factor": self.decay_factor,
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Ensemble config saved to {path}")
        return path

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        models: dict[str, object],
        calibrator: ProbabilityCalibrator | None = None,
    ) -> EnsembleModel:
        """Load ensemble configuration."""
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

        ensemble = cls(
            models=models,
            weights=payload["weights"],
            calibrator=calibrator,
            decay_factor=payload.get("decay_factor", 0.95),
        )

        # Restore performance history
        for name, history in payload.get("performance_history", {}).items():
            if name in ensemble.performance_history:
                ensemble.performance_history[name] = history

        return ensemble


def compute_dynamic_weights(
    model_losses: dict[str, list[float]],
    decay_factor: float = 0.95,
) -> dict[str, float]:
    """Compute dynamic weights based on historical losses.

    Args:
        model_losses: Dictionary of model_name -> list of losses
        decay_factor: Decay factor for exponential weighting

    Returns:
        Dictionary of model_name -> weight
    """
    avg_losses = {}

    for name, losses in model_losses.items():
        if not losses:
            avg_losses[name] = 1.0
            continue

        weights = np.power(decay_factor, np.arange(len(losses))[::-1])
        avg_loss = float(np.average(losses, weights=weights))
        avg_losses[name] = avg_loss

    # Inverse loss weighting
    inv_losses = {name: 1.0 / loss for name, loss in avg_losses.items()}
    total = sum(inv_losses.values())

    return {name: inv / total for name, inv in inv_losses.items()}
