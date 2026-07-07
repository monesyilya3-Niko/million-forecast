"""Elo rating model for football match prediction.

Uses dynamic team ratings that update after each match,
incorporating home advantage and margin of victory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from football_model.engine import score_matrix

logger = logging.getLogger(__name__)


@dataclass
class EloModel:
    """Elo-based match prediction model."""
    competition: str
    teams: list[str]
    ratings: dict[str, float]
    home_advantage: float
    k_factor: float
    mean_goals: float
    trained_at: str
    training_cutoff: str
    metrics: dict[str, float | int | str]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        competition: str,
        k_factor: float = 20.0,
        home_advantage_elo: float = 100.0,
        initial_rating: float = 1500.0,
    ) -> EloModel:
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"缺少字段：{', '.join(sorted(missing))}")

        data = frame.dropna(subset=list(required)).copy()
        data["kickoff"] = pd.to_datetime(data["kickoff"])
        data = data.sort_values("kickoff").reset_index(drop=True)

        if len(data) < 50:
            raise ValueError("至少需要50场已完成比赛")

        # Initialize ratings
        all_teams = set(data["home_team"]) | set(data["away_team"])
        ratings = {team: initial_rating for team in all_teams}

        # Track predictions for evaluation
        predictions = []
        outcomes = []

        for _, row in data.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            # Ensure teams exist
            if home not in ratings:
                ratings[home] = initial_rating
            if away not in ratings:
                ratings[away] = initial_rating

            # Expected scores
            home_rating = ratings[home] + home_advantage_elo
            away_rating = ratings[away]
            expected_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
            expected_away = 1.0 - expected_home

            # Store prediction
            predictions.append([expected_home, 1 - expected_home - expected_away, expected_away])

            # Actual outcome
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            if hg > ag:
                actual_home = 1.0
            elif hg == ag:
                actual_home = 0.5
            else:
                actual_home = 0.0
            outcomes.append(0 if hg > ag else 1 if hg == ag else 2)

            # Update ratings with margin multiplier
            margin = abs(hg - ag)
            margin_mult = np.log(margin + 1) if margin > 0 else 1.0
            goal_diff = hg - ag
            mov_multiplier = margin_mult * (2.2 / (goal_diff * 0.001 + 2.2)) if goal_diff != 0 else 1.0

            delta = k_factor * mov_multiplier * (actual_home - expected_home)
            ratings[home] = ratings.get(home, initial_rating) + delta
            ratings[away] = ratings.get(away, initial_rating) - delta

        # Compute holdout metrics
        split = max(50, int(len(data) * 0.8))
        holdout_preds = np.array(predictions[split:])
        holdout_outcomes = np.array(outcomes[split:])

        if len(holdout_preds) >= 30:
            holdout_ll = float(np.mean([
                -np.log(np.clip(holdout_preds[i, holdout_outcomes[i]], 1e-9, 1))
                for i in range(len(holdout_outcomes))
            ]))
            one_hot = np.eye(3)[holdout_outcomes]
            holdout_brier = float(np.mean(np.sum((holdout_preds - one_hot) ** 2, axis=1)))
            holdout_acc = float((holdout_preds.argmax(axis=1) == holdout_outcomes).mean())
        else:
            holdout_ll = holdout_brier = holdout_acc = 0.0

        # Final ratings (normalize to mean 1500)
        mean_rating = np.mean(list(ratings.values()))
        ratings = {t: r - mean_rating + 1500 for t, r in ratings.items()}

        home_mean = float(data["home_goals"].mean())
        away_mean = float(data["away_goals"].mean())
        latest = data["kickoff"].max()

        metrics = {
            "matches": len(data),
            "teams": len(all_teams),
            "k_factor": k_factor,
            "home_advantage_elo": home_advantage_elo,
            "holdout_log_loss": holdout_ll,
            "holdout_brier": holdout_brier,
            "holdout_accuracy": holdout_acc,
            "competition": competition,
        }

        return cls(
            competition=competition,
            teams=sorted(all_teams),
            ratings=ratings,
            home_advantage=home_advantage_elo,
            k_factor=k_factor,
            mean_goals=float((home_mean + away_mean) / 2),
            trained_at=datetime.now(UTC).isoformat(),
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home_r = self.ratings.get(home_team, 1500.0) + self.home_advantage
        away_r = self.ratings.get(away_team, 1500.0)
        expected_home = 1.0 / (1.0 + 10 ** ((away_r - home_r) / 400.0))
        total_goals = self.mean_goals * 2
        home_xg = max(0.5, expected_home * total_goals)
        away_xg = max(0.5, (1 - expected_home) * total_goals)
        return float(home_xg), float(away_xg)

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        home_xg, away_xg = self.expected_goals(home_team, away_team)
        matrix = score_matrix(home_xg, away_xg, rho=-0.05)
        return {
            "home_win": float(np.tril(matrix, k=-1).sum()),
            "draw": float(np.trace(matrix)),
            "away_win": float(np.triu(matrix, k=1).sum()),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> EloModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
