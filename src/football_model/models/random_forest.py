"""Random Forest model for football match prediction.

Uses ensemble of decision trees to capture non-linear patterns.
More robust to overfitting than single trees, provides feature importance.
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
from sklearn.ensemble import RandomForestClassifier

from football_model.features.pipeline import FeaturePipeline, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class RandomForestModel:
    """Random Forest model for football match outcome prediction."""
    competition: str
    feature_names: list[str]
    model_path: Path
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
        n_estimators: int = 300,
        max_depth: int = 8,
    ) -> RandomForestModel:
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"缺少字段：{', '.join(sorted(missing))}")

        data = frame.dropna(subset=list(required)).copy()
        if len(data) < 50:
            raise ValueError("至少需要50场已完成比赛")

        # Generate features
        pipeline = FeaturePipeline()
        data_with_features = pipeline.transform(data, history)

        feature_names = get_feature_names()
        X = data_with_features[feature_names].fillna(0).to_numpy(dtype=float)

        # Labels: H=0, D=1, A=2
        y = np.where(
            data_with_features["home_goals"] > data_with_features["away_goals"], 0,
            np.where(data_with_features["home_goals"] == data_with_features["away_goals"], 1, 2)
        )

        # Holdout evaluation
        split = max(50, int(len(X) * 0.8))
        holdout_metrics: dict[str, float | int | str] = {}

        if len(X) - split >= 30:
            X_train, X_holdout = X[:split], X[split:]
            y_train, y_holdout = y[:split], y[split:]

            eval_model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                n_jobs=-1, random_state=42, class_weight="balanced",
            )
            eval_model.fit(X_train, y_train)
            proba = eval_model.predict_proba(X_holdout)

            losses = []
            briers = []
            correct = 0
            for i in range(len(y_holdout)):
                p = proba[i]
                target = np.eye(3)[y_holdout[i]]
                losses.append(float(-np.log(np.clip(p[y_holdout[i]], 1e-9, 1))))
                briers.append(float(np.square(p - target).sum()))
                correct += int(np.argmax(p) == y_holdout[i])

            holdout_metrics = {
                "holdout_matches": len(y_holdout),
                "holdout_log_loss": float(np.mean(losses)),
                "holdout_brier": float(np.mean(briers)),
                "holdout_accuracy": correct / len(y_holdout),
            }

        # Train final model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )
        model.fit(X, y)

        # Feature importance
        importance = dict(zip(feature_names, model.feature_importances_, strict=True))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Save model
        latest = pd.to_datetime(data["kickoff"]).max()
        model_path = Path("artifacts") / "random_forest" / f"{competition}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names}, f)

        metrics = {
            "matches": len(data),
            "features": len(feature_names),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "competition": competition,
            "top_features": {k: float(v) for k, v in top_features},
            **holdout_metrics,
        }

        logger.info(f"RandomForest trained: {len(data)} matches, {len(feature_names)} features")

        return cls(
            competition=competition,
            feature_names=feature_names,
            model_path=model_path,
            trained_at=datetime.now(UTC).isoformat(),
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    @classmethod
    def load(cls, path: str | Path) -> RandomForestModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p
