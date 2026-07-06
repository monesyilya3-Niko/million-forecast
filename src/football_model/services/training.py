from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from football_model.data import LocalDatabase, MatchRepository, ModelRepository
from football_model.engine import score_matrix
from football_model.models import DixonColesModel

logger = logging.getLogger(__name__)


class ModelTrainingService:
    def __init__(self, database: LocalDatabase, artifacts_dir: str | Path) -> None:
        self.matches = MatchRepository(database)
        self.models = ModelRepository(database)
        self.artifacts_dir = Path(artifacts_dir)
        logger.info(f"ModelTrainingService initialized with artifacts dir: {self.artifacts_dir}")

    def train_dixon_coles(self, competition: str, *, team_scope: set[str] | None = None) -> tuple[DixonColesModel, str]:
        """Train a Dixon-Coles model for a specific competition."""
        logger.info(f"Starting training for competition: {competition}")
        training_data = self.matches.training_frame(competition)
        if team_scope:
            training_data = training_data.loc[
                training_data["home_team"].isin(team_scope) & training_data["away_team"].isin(team_scope)
            ].copy()
        validation_metrics = self._time_holdout_metrics(training_data, competition)
        model = DixonColesModel.fit(training_data, competition=competition)
        model.metrics.update(validation_metrics)
        model.metrics["training_cutoff"] = model.training_cutoff
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        competition_hash = hashlib.sha256(competition.encode("utf-8")).hexdigest()[:8]
        version = f"{timestamp}-{competition_hash}"
        model_id = f"dixon-coles:{competition_hash}:{timestamp}"
        artifact_path = self.artifacts_dir / "dixon_coles" / f"{version}.json"
        model.save(artifact_path)
        self.models.register(
            model_id=model_id,
            model_type="Dixon-Coles League",
            version=version,
            artifact_path=artifact_path,
            metrics=model.metrics,
            status="active",
        )
        return model, model_id

    @staticmethod
    def _time_holdout_metrics(training_data, competition: str) -> dict[str, float | int]:
        split_index = max(100, int(len(training_data) * 0.8))
        if len(training_data) - split_index < 30:
            return {}
        train = training_data.iloc[:split_index]
        holdout = training_data.iloc[split_index:]
        evaluation_model = DixonColesModel.fit(train, competition=competition)
        losses: list[float] = []
        briers: list[float] = []
        correct = 0
        for row in holdout.itertuples(index=False):
            home_xg, away_xg = evaluation_model.expected_goals(row.home_team, row.away_team)
            matrix = score_matrix(home_xg, away_xg, rho=evaluation_model.rho)
            probabilities = np.array(
                [
                    float(np.tril(matrix, k=-1).sum()),
                    float(np.trace(matrix)),
                    float(np.triu(matrix, k=1).sum()),
                ]
            )
            outcome = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
            target = np.eye(3)[outcome]
            losses.append(float(-np.log(np.clip(probabilities[outcome], 1e-9, 1))))
            briers.append(float(np.square(probabilities - target).sum()))
            correct += int(int(np.argmax(probabilities)) == outcome)
        return {
            "holdout_matches": len(holdout),
            "holdout_log_loss": float(np.mean(losses)),
            "holdout_brier": float(np.mean(briers)),
            "holdout_accuracy": correct / len(holdout),
            "validation_method": "chronological_80_20",
        }
