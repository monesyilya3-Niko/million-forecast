from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_model.data.adapters import FootballDataCsvAdapter
from football_model.models import DixonColesModel


def test_football_data_adapter_normalizes_matches_and_odds(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "Date": "01/08/2025",
                "Time": "20:00",
                "HomeTeam": "Home",
                "AwayTeam": "Away",
                "FTHG": 2,
                "FTAG": 1,
                "AvgCH": 1.8,
                "AvgCD": 3.5,
                "AvgCA": 4.2,
            }
        ]
    )
    csv_path = tmp_path / "E0.csv"
    raw.to_csv(csv_path, index=False)
    matches, odds = FootballDataCsvAdapter(tmp_path).normalize(csv_path, season="2526", division="E0")
    assert len(matches) == 1
    assert matches.iloc[0]["competition"] == "英格兰超级联赛"
    assert len(odds) == 3
    assert set(odds["selection"]) == {"H", "D", "A"}


def test_dixon_coles_model_trains_saves_and_loads(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D"]
    rows = []
    for index in range(120):
        home = teams[index % 4]
        away = teams[(index + 1 + index // 4) % 4]
        if home == away:
            away = teams[(teams.index(away) + 1) % 4]
        rows.append(
            {
                "kickoff": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "home_team": home,
                "away_team": away,
                "home_goals": rng.poisson(1.6),
                "away_goals": rng.poisson(1.1),
            }
        )
    model = DixonColesModel.fit(pd.DataFrame(rows), competition="测试联赛")
    home_xg, away_xg = model.expected_goals("A", "B")
    assert 0.1 <= home_xg <= 5
    assert 0.1 <= away_xg <= 5
    artifact = model.save(tmp_path / "model.json")
    loaded = DixonColesModel.load(artifact)
    assert loaded.teams == model.teams
    assert loaded.expected_goals("A", "B") == model.expected_goals("A", "B")
