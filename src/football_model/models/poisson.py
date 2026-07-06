"""Poisson regression model for football match prediction.

Uses separate Poisson distributions for home and away goals, with
log-link functions and regularization. More flexible than Dixon-Coles
for capturing team-specific attacking/defending patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy

from football_model.engine import score_matrix
from football_model.features.pipeline import FeaturePipeline, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class PoissonModel:
    """Poisson regression model for football match prediction.

    Models home_goals ~ Poisson(lambda_home) and away_goals ~ Poisson(lambda_away)
    where log(lambda) = X @ beta.
    """

    competition: str
    feature_names: list[str]
    home_coefficients: dict[str, float]
    away_coefficients: dict[str, float]
    home_intercept: float
    away_intercept: float
    trained_at: str
    training_cutoff: str
    metrics: dict[str, float | int | str]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        history: pd.DataFrame,
        *,
        competition: str,
        regularization: float = 0.01,
    ) -> PoissonModel:
        """Train Poisson regression model.

        Args:
            frame: Training matches (must include home_goals, away_goals)
            history: Historical matches for feature generation
            competition: Competition name
            regularization: L2 regularization strength

        Returns:
            Trained PoissonModel
        """
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"训练数据缺少字段：{', '.join(sorted(missing))}")

        data = frame.dropna(subset=list(required)).copy()
        if len(data) < 50:
            raise ValueError("至少需要50场已完成比赛才能训练Poisson模型")

        # Generate features
        pipeline = FeaturePipeline()
        data_with_features = pipeline.transform(data, history)

        # Get feature columns
        feature_names = get_feature_names()
        X = data_with_features[feature_names].fillna(0).to_numpy(dtype=float)

        # Targets
        home_goals = data_with_features["home_goals"].to_numpy(dtype=float)
        away_goals = data_with_features["away_goals"].to_numpy(dtype=float)

        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Chronological holdout evaluation. Feature generation already applies a
        # strict per-match cutoff, so the validation rows contain no future data.
        split_index = max(50, int(len(X) * 0.8))
        holdout_metrics: dict[str, float | int | str] = {}
        if len(X) - split_index >= 30:
            validation_home = _fit_poisson(
                X_with_intercept[:split_index],
                home_goals[:split_index],
                regularization=regularization,
            )
            validation_away = _fit_poisson(
                X_with_intercept[:split_index],
                away_goals[:split_index],
                regularization=regularization,
            )
            home_rates = np.exp(np.clip(X_with_intercept[split_index:] @ validation_home.x, -5, 5))
            away_rates = np.exp(np.clip(X_with_intercept[split_index:] @ validation_away.x, -5, 5))
            losses = []
            briers = []
            correct = 0
            for offset, (home_rate, away_rate) in enumerate(zip(home_rates, away_rates, strict=True)):
                matrix = score_matrix(float(home_rate), float(away_rate), rho=0.0)
                probabilities = np.array(
                    [
                        float(np.tril(matrix, k=-1).sum()),
                        float(np.trace(matrix)),
                        float(np.triu(matrix, k=1).sum()),
                    ]
                )
                index = split_index + offset
                outcome = (
                    0 if home_goals[index] > away_goals[index] else 1 if home_goals[index] == away_goals[index] else 2
                )
                target = np.eye(3)[outcome]
                losses.append(float(-np.log(np.clip(probabilities[outcome], 1e-9, 1))))
                briers.append(float(np.square(probabilities - target).sum()))
                correct += int(int(np.argmax(probabilities)) == outcome)
            holdout_metrics = {
                "holdout_matches": len(losses),
                "holdout_log_loss": float(np.mean(losses)),
                "holdout_brier": float(np.mean(briers)),
                "holdout_accuracy": correct / len(losses),
                "validation_method": "chronological_80_20",
            }

        # Train home goals model
        home_result = _fit_poisson(X_with_intercept, home_goals, regularization=regularization)

        # Train away goals model
        away_result = _fit_poisson(X_with_intercept, away_goals, regularization=regularization)

        # Extract coefficients
        home_intercept = float(home_result.x[0])
        home_coefs = dict(zip(feature_names, home_result.x[1:], strict=True))

        away_intercept = float(away_result.x[0])
        away_coefs = dict(zip(feature_names, away_result.x[1:], strict=True))

        # Compute metrics
        home_pred = np.exp(X_with_intercept @ home_result.x)
        away_pred = np.exp(X_with_intercept @ away_result.x)

        # Log-likelihood
        home_ll = float(np.sum(xlogy(home_goals, home_pred) - home_pred - gammaln(home_goals + 1)))
        away_ll = float(np.sum(xlogy(away_goals, away_pred) - away_pred - gammaln(away_goals + 1)))

        # MAE
        home_mae = float(np.mean(np.abs(home_goals - home_pred)))
        away_mae = float(np.mean(np.abs(away_goals - away_pred)))

        trained_at = datetime.now(UTC).isoformat()
        latest = pd.to_datetime(data["kickoff"]).max()

        metrics = {
            "matches": len(data),
            "features": len(feature_names),
            "home_log_likelihood": home_ll,
            "away_log_likelihood": away_ll,
            "total_ll": home_ll + away_ll,
            "home_mae": home_mae,
            "away_mae": away_mae,
            "avg_mae": (home_mae + away_mae) / 2,
            "regularization": regularization,
            "competition": competition,
            **holdout_metrics,
        }

        logger.info(
            f"Poisson model trained: {len(data)} matches, {len(feature_names)} features, MAE={metrics['avg_mae']:.3f}"
        )

        return cls(
            competition=competition,
            feature_names=feature_names,
            home_coefficients=home_coefs,
            away_coefficients=away_coefs,
            home_intercept=home_intercept,
            away_intercept=away_intercept,
            trained_at=trained_at,
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    def expected_goals(
        self,
        features: dict[str, float],
    ) -> tuple[float, float]:
        """Predict expected goals for a match.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (home_xg, away_xg)
        """
        # Build feature vector
        x = np.array([features.get(name, 0.0) for name in self.feature_names])

        # Add intercept
        x_with_intercept = np.concatenate([[1.0], x])

        # Compute lambda
        home_coefs = np.concatenate([[self.home_intercept], list(self.home_coefficients.values())])
        away_coefs = np.concatenate([[self.away_intercept], list(self.away_coefficients.values())])

        home_lambda = np.exp(np.clip(x_with_intercept @ home_coefs, -5, 5))
        away_lambda = np.exp(np.clip(x_with_intercept @ away_coefs, -5, 5))

        return float(np.clip(home_lambda, 0.7, 5.0)), float(np.clip(away_lambda, 0.7, 5.0))

    def predict_probabilities(
        self,
        features: dict[str, float],
        max_goals: int = 10,
    ) -> dict[str, float]:
        """Predict match outcome probabilities.

        Args:
            features: Feature dictionary
            max_goals: Maximum goals to consider

        Returns:
            Dictionary with home_win, draw, away_win probabilities
        """
        home_xg, away_xg = self.expected_goals(features)

        # Compute score matrix
        goals = np.arange(max_goals + 1, dtype=np.float64)
        home_pmf = np.exp(-home_xg + xlogy(goals, home_xg) - gammaln(goals + 1))
        away_pmf = np.exp(-away_xg + xlogy(goals, away_xg) - gammaln(goals + 1))

        matrix = np.outer(home_pmf, away_pmf)
        matrix = matrix / matrix.sum()

        # Compute outcome probabilities
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }

    def save(self, artifact_path: str | Path) -> Path:
        """Save model to JSON file."""
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "competition": self.competition,
            "feature_names": self.feature_names,
            "home_coefficients": self.home_coefficients,
            "away_coefficients": self.away_coefficients,
            "home_intercept": self.home_intercept,
            "away_intercept": self.away_intercept,
            "trained_at": self.trained_at,
            "training_cutoff": self.training_cutoff,
            "metrics": self.metrics,
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Poisson model saved to {path}")
        return path

    @classmethod
    def load(cls, artifact_path: str | Path) -> PoissonModel:
        """Load model from JSON file."""
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        return cls(**payload)


def _fit_poisson(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.01,
) -> object:
    """Fit Poisson regression using L-BFGS-B.

    Args:
        X: Feature matrix (with intercept column)
        y: Target values (goals)
        regularization: L2 regularization strength

    Returns:
        Optimization result
    """
    n_features = X.shape[1]

    def objective(beta: np.ndarray) -> float:
        """Negative log-likelihood with L2 regularization."""
        eta = X @ beta
        eta = np.clip(eta, -20, 20)  # Prevent overflow
        mu = np.exp(eta)

        # Negative log-likelihood
        nll = -np.sum(xlogy(y, mu) - mu - gammaln(y + 1))

        # L2 regularization (don't regularize intercept)
        ridge = regularization * np.sum(beta[1:] ** 2)

        return float(nll + ridge)

    def gradient(beta: np.ndarray) -> np.ndarray:
        """Gradient of negative log-likelihood."""
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)

        # Gradient of NLL
        grad_nll = -X.T @ (y - mu)

        # Gradient of L2 regularization
        grad_ridge = 2 * regularization * beta
        grad_ridge[0] = 0  # Don't regularize intercept

        return grad_nll + grad_ridge

    # Initialize with zeros
    beta0 = np.zeros(n_features)

    # Optimize
    result = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-8},
    )

    if not result.success:
        logger.warning(f"Poisson optimization did not converge: {result.message}")

    return result
