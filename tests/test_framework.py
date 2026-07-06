from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from football_model.data import LocalDatabase, MatchRepository, ModelRepository
from football_model.features import FeaturePipeline
from football_model.services import AnalysisService


def test_database_initialization_and_model_registry(tmp_path: Path) -> None:
    database = LocalDatabase(tmp_path / "test.duckdb")
    database.initialize()
    assert database.health_check()
    assert database.table_counts()["model_registry"] == 1
    assert ModelRepository(database).list_models().iloc[0]["status"] == "active"


def test_match_repository_imports_and_deduplicates(tmp_path: Path) -> None:
    database = LocalDatabase(tmp_path / "test.duckdb")
    database.initialize()
    repository = MatchRepository(database)
    frame = pd.DataFrame(
        [
            {
                "kickoff": "2026-08-01 19:30:00",
                "competition": "测试联赛",
                "home_team": "主队",
                "away_team": "客队",
            }
        ]
    )
    assert repository.import_frame(frame) == 1
    assert repository.import_frame(frame) == 1
    assert database.table_counts()["matches"] == 1


def test_feature_pipeline_rejects_future_cutoff() -> None:
    pipeline = FeaturePipeline()
    frame = pd.DataFrame(
        [
            {
                "match_id": "1",
                "kickoff": "2026-08-01 19:30:00",
                "cutoff_at": "2026-08-01 20:00:00",
                "home_team": "主队",
                "away_team": "客队",
            }
        ]
    )
    empty_history = pd.DataFrame(columns=["kickoff", "home_team", "away_team", "home_goals", "away_goals"])
    with pytest.raises(ValueError, match="cutoff_at"):
        pipeline.transform(frame, empty_history)


def test_analysis_service_exposes_consistent_result() -> None:
    result = AnalysisService().analyze(
        home_xg=1.6,
        away_xg=1.1,
        odds_home=2.0,
        odds_draw=3.3,
        odds_away=3.8,
        handicap=-1,
    )
    assert len(result.comparison) == 3
    assert abs(sum(result.summary.probabilities.values()) - 1) < 1e-9
    assert abs(sum(result.handicap.values()) - 1) < 1e-9
