from __future__ import annotations

import numpy as np
import pandas as pd

from football_model.backtesting.walk_forward import WalkForwardBacktester
from football_model.data import LocalDatabase, PredictionRepository
from football_model.services.ensemble_service import EnsembleAnalysisService


def test_dynamic_ensemble_weights_are_normalized_and_reward_lower_loss() -> None:
    weights = EnsembleAnalysisService._weights(
        "瑞典超级联赛",
        {"dixon_coles": 1.10, "poisson": 0.95},
    )
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert weights["poisson"] > weights["dixon_coles"]
    assert weights["market"] == 0.35


def test_prediction_repository_persists_full_audit_record(tmp_path) -> None:
    database = LocalDatabase(tmp_path / "prediction.duckdb")
    database.initialize()
    prediction_id = PredictionRepository(database).save(
        match_id="match-1",
        model_version="ensemble-v2",
        cutoff_at="2026-07-05 10:00",
        home_probability=0.5,
        draw_probability=0.3,
        away_probability=0.2,
        home_xg=1.5,
        away_xg=0.9,
        confidence=72,
        components={"market": {"home_win": 0.48}},
        input_odds={"主胜": 2.0, "平局": 3.2, "客胜": 4.0},
    )
    with database.connection(read_only=True) as connection:
        row = connection.execute("SELECT prediction_id, confidence, components_json FROM predictions").fetchone()
    assert row[0] == prediction_id
    assert row[1] == 72
    assert "market" in row[2]


def test_walk_forward_retrains_every_window() -> None:
    matches = pd.DataFrame(
        {
            "kickoff": pd.date_range("2025-01-01", periods=18, freq="D"),
            "home_team": ["A"] * 18,
            "away_team": ["B"] * 18,
            "home_goals": [1, 2, 0] * 6,
            "away_goals": [0, 2, 1] * 6,
        }
    )
    trained_sizes: list[int] = []

    def trainer(frame: pd.DataFrame) -> int:
        trained_sizes.append(len(frame))
        return len(frame)

    def predictor(model: object, frame: pd.DataFrame) -> pd.DataFrame:
        del model
        probabilities = np.tile([0.45, 0.30, 0.25], (len(frame), 1))
        return pd.DataFrame(probabilities, columns=["home_win", "draw", "away_win"])

    result = WalkForwardBacktester(initial_train_size=6, test_size=3, step_size=3).run_retraining(
        "fake",
        "test",
        matches,
        trainer,
        predictor,
    )
    assert result.windows == 4
    assert trained_sizes == [6, 9, 12, 15]
    assert result.total_matches == 12
