"""Neural network model for football match prediction.

Uses sklearn MLPClassifier as a baseline. Can be upgraded to
PyTorch/TensorFlow when available for better architecture control.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from football_model.features.pipeline import FeaturePipeline, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetModel:
    """Neural network model for football match outcome prediction."""

    competition: str
    feature_names: list[str]
    model_path: Path
    scaler_path: Path
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
        hidden_layers: tuple[int, ...] = (64, 32, 16),
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        validation_fraction: float = 0.15,
    ) -> NeuralNetModel:
        """Train neural network model."""
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"训练数据缺少字段：{', '.join(sorted(missing))}")

        data = frame.dropna(subset=list(required)).copy()
        if len(data) < 100:
            raise ValueError("至少需要100场已完成比赛")

        # Generate features
        pipeline = FeaturePipeline()
        data_with_features = pipeline.transform(data, history)

        feature_names = get_feature_names()
        X = data_with_features[feature_names].fillna(0).to_numpy(dtype=float)

        # Target: H/D/A
        outcomes = []
        for _, row in data_with_features.iterrows():
            if row["home_goals"] > row["away_goals"]:
                outcomes.append(0)
            elif row["home_goals"] == row["away_goals"]:
                outcomes.append(1)
            else:
                outcomes.append(2)
        y = np.array(outcomes)

        # Chronological split
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train MLP
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        model.fit(X_train_scaled, y_train)

        # Metrics
        train_proba = model.predict_proba(X_train_scaled)
        val_proba = model.predict_proba(X_val_scaled)
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        from sklearn.metrics import log_loss

        train_acc = float((train_pred == y_train).mean())
        val_acc = float((val_pred == y_val).mean())
        train_ll = float(log_loss(y_train, train_proba, labels=[0, 1, 2]))
        val_ll = float(log_loss(y_val, val_proba, labels=[0, 1, 2]))

        trained_at = datetime.now(UTC).isoformat()
        latest = pd.to_datetime(data["kickoff"]).max()

        # Save model and scaler
        model_dir = Path("artifacts") / "neural_net" / competition.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"model_{trained_at.replace(':', '-')}.pkl"
        scaler_path = model_dir / f"scaler_{trained_at.replace(':', '-')}.pkl"

        with model_path.open("wb") as f:
            pickle.dump(model, f)
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)

        # Draw prediction analysis
        draw_pred_train = float((train_pred == 1).mean())
        draw_pred_val = float((val_pred == 1).mean())
        draw_actual_train = float((y_train == 1).mean())
        draw_actual_val = float((y_val == 1).mean())

        metrics = {
            "matches": len(data),
            "features": len(feature_names),
            "hidden_layers": list(hidden_layers),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_log_loss": train_ll,
            "val_log_loss": val_ll,
            "draw_pred_train": draw_pred_train,
            "draw_pred_val": draw_pred_val,
            "draw_actual_train": draw_actual_train,
            "draw_actual_val": draw_actual_val,
            "n_iterations": int(model.n_iter_),
            "competition": competition,
        }

        logger.info(
            f"NeuralNet trained: {len(data)} matches, "
            f"val_acc={val_acc:.3f}, val_ll={val_ll:.3f}, "
            f"draw_pred={draw_pred_val:.1%}"
        )

        return cls(
            competition=competition,
            feature_names=feature_names,
            model_path=model_path,
            scaler_path=scaler_path,
            trained_at=trained_at,
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    def predict_probabilities(self, features: dict[str, float]) -> dict[str, float]:
        """Predict match outcome probabilities."""
        with Path(self.model_path).open("rb") as f:
            model = pickle.load(f)
        with Path(self.scaler_path).open("rb") as f:
            scaler = pickle.load(f)

        x = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        x_scaled = scaler.transform(x)
        proba = model.predict_proba(x_scaled)[0]

        # Map to outcomes (model.classes_ = [0, 1, 2])
        classes = model.classes_
        result = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for i, cls in enumerate(classes):
            if cls == 0:
                result["home_win"] = float(proba[i])
            elif cls == 1:
                result["draw"] = float(proba[i])
            elif cls == 2:
                result["away_win"] = float(proba[i])

        total = sum(result.values())
        return {k: v / total for k, v in result.items()}

    def save_metadata(self, artifact_path: str | Path) -> Path:
        """Save model metadata."""
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "competition": self.competition,
            "feature_names": self.feature_names,
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "trained_at": self.trained_at,
            "training_cutoff": self.training_cutoff,
            "metrics": self.metrics,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, artifact_path: str | Path) -> NeuralNetModel:
        """Load model metadata."""
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        return cls(
            competition=payload["competition"],
            feature_names=payload["feature_names"],
            model_path=Path(payload["model_path"]),
            scaler_path=Path(payload["scaler_path"]),
            trained_at=payload["trained_at"],
            training_cutoff=payload["training_cutoff"],
            metrics=payload["metrics"],
        )
