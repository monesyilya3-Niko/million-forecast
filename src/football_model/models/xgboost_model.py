"""XGBoost model for football match outcome classification.

Uses gradient boosting to capture non-linear feature interactions
and complex patterns that statistical models might miss.
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
from sklearn.preprocessing import LabelEncoder

from football_model.features.pipeline import FeaturePipeline, get_feature_names

logger = logging.getLogger(__name__)

# Try to import XGBoost, fallback to sklearn if not available
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class XGBoostModel:
    """XGBoost model for football match outcome prediction.

    Classifies matches into Home Win (H), Draw (D), or Away Win (A).
    Outputs raw class probabilities. It must remain experimental until an
    external validation set and ProbabilityCalibrator are registered.
    """

    competition: str
    feature_names: list[str]
    model_path: Path
    label_encoder: LabelEncoder
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
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        validation_fraction: float = 0.2,
    ) -> XGBoostModel:
        """Train XGBoost model.

        Args:
            frame: Training matches (must include home_goals, away_goals)
            history: Historical matches for feature generation
            competition: Competition name
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            validation_fraction: Fraction of data for validation

        Returns:
            Trained XGBoostModel
        """
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"训练数据缺少字段：{', '.join(sorted(missing))}")

        data = frame.dropna(subset=list(required)).copy()
        if len(data) < 100:
            raise ValueError("至少需要100场已完成比赛才能训练XGBoost模型")

        # Generate features
        pipeline = FeaturePipeline()
        data_with_features = pipeline.transform(data, history)

        # Get feature columns
        feature_names = get_feature_names()
        X = data_with_features[feature_names].fillna(0).to_numpy(dtype=float)

        # Create target (H/D/A)
        outcomes = []
        for _, row in data_with_features.iterrows():
            if row["home_goals"] > row["away_goals"]:
                outcomes.append("H")
            elif row["home_goals"] == row["away_goals"]:
                outcomes.append("D")
            else:
                outcomes.append("A")

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(outcomes)

        # Split into train/validation (chronological)
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            logger.warning("XGBoost not available, using sklearn GradientBoostingClassifier")
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            model.fit(X_train, y_train)

        # Compute metrics
        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_accuracy = float((train_pred == y_train).mean())
        val_accuracy = float((val_pred == y_val).mean())

        # Log loss
        from sklearn.metrics import log_loss

        train_log_loss = float(log_loss(y_train, train_proba))
        val_log_loss = float(log_loss(y_val, val_proba))

        # Feature importance
        if HAS_XGBOOST:
            importance = dict(zip(feature_names, model.feature_importances_, strict=True))
        else:
            importance = dict(zip(feature_names, model.feature_importances_, strict=True))

        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        trained_at = datetime.now(UTC).isoformat()
        latest = pd.to_datetime(data["kickoff"]).max()

        # Save model
        model_dir = Path("artifacts") / "xgboost" / competition.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"model_{trained_at.replace(':', '-')}.pkl"

        with model_path.open("wb") as f:
            pickle.dump(model, f)

        metrics = {
            "matches": len(data),
            "features": len(feature_names),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_log_loss": train_log_loss,
            "val_log_loss": val_log_loss,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "top_features": dict(list(sorted_importance.items())[:10]),
            "competition": competition,
        }

        logger.info(
            f"XGBoost model trained: {len(data)} matches, "
            f"val_accuracy={val_accuracy:.3f}, val_log_loss={val_log_loss:.3f}"
        )

        return cls(
            competition=competition,
            feature_names=feature_names,
            model_path=model_path,
            label_encoder=label_encoder,
            trained_at=trained_at,
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    def predict_probabilities(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Predict match outcome probabilities.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary with home_win, draw, away_win probabilities
        """
        # Load model
        with Path(self.model_path).open("rb") as f:
            model = pickle.load(f)

        # Build feature vector
        x = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        # Predict probabilities
        proba = model.predict_proba(x)[0]

        # Map to outcome labels
        labels = self.label_encoder.classes_
        result = {}
        for i, label in enumerate(labels):
            if label == "H":
                result["home_win"] = float(proba[i])
            elif label == "D":
                result["draw"] = float(proba[i])
            elif label == "A":
                result["away_win"] = float(proba[i])

        # Ensure all outcomes are present
        result.setdefault("home_win", 0.33)
        result.setdefault("draw", 0.33)
        result.setdefault("away_win", 0.33)

        # Normalize
        total = sum(result.values())
        return {k: v / total for k, v in result.items()}

    def save_metadata(self, artifact_path: str | Path) -> Path:
        """Save model metadata (not the model itself)."""
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "competition": self.competition,
            "feature_names": self.feature_names,
            "model_path": str(self.model_path),
            "label_classes": self.label_encoder.classes_.tolist(),
            "trained_at": self.trained_at,
            "training_cutoff": self.training_cutoff,
            "metrics": self.metrics,
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"XGBoost metadata saved to {path}")
        return path

    @classmethod
    def load(cls, artifact_path: str | Path) -> XGBoostModel:
        """Load model metadata."""
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(payload["label_classes"])

        return cls(
            competition=payload["competition"],
            feature_names=payload["feature_names"],
            model_path=Path(payload["model_path"]),
            label_encoder=label_encoder,
            trained_at=payload["trained_at"],
            training_cutoff=payload["training_cutoff"],
            metrics=payload["metrics"],
        )
