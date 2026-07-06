from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)


@dataclass
class DixonColesModel:
    competition: str
    teams: list[str]
    attacks: dict[str, float]
    defenses: dict[str, float]
    intercept: float
    home_advantage: float
    rho: float
    trained_at: str
    training_cutoff: str
    metrics: dict[str, float | int | str]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        competition: str,
        decay: float = 0.0015,
        regularization: float = 0.02,
    ) -> DixonColesModel:
        required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"训练数据缺少字段：{', '.join(sorted(missing))}")
        data = frame.dropna(subset=list(required)).copy()
        if len(data) < 100:
            raise ValueError("至少需要100场已完成比赛才能训练联赛模型")

        data["kickoff"] = pd.to_datetime(data["kickoff"])

        # Filter out teams with too few matches to get stable parameters
        team_counts = data["home_team"].value_counts().add(
            data["away_team"].value_counts(), fill_value=0
        )
        min_matches = 10
        valid_teams = set(team_counts[team_counts >= min_matches].index)
        data = data[
            data["home_team"].isin(valid_teams) & data["away_team"].isin(valid_teams)
        ].copy()
        if len(data) < 50:
            raise ValueError(f"过滤后剩余{len(data)}场，不足50场")

        teams = sorted(valid_teams)
        team_index = {team: index for index, team in enumerate(teams)}
        home_index = data["home_team"].map(team_index).to_numpy(dtype=int)
        away_index = data["away_team"].map(team_index).to_numpy(dtype=int)
        home_goals = data["home_goals"].to_numpy(dtype=int)
        away_goals = data["away_goals"].to_numpy(dtype=int)
        latest = data["kickoff"].max()
        age_days = (latest - data["kickoff"]).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-decay * age_days)
        team_count = len(teams)

        mean_goals = max(float((home_goals.sum() + away_goals.sum()) / (2 * len(data))), 0.2)
        initial = np.zeros(team_count * 2 + 3)
        initial[-3] = np.log(mean_goals)
        initial[-2] = 0.2
        initial[-1] = -0.05

        # Pre-compute masks for tau calculation (optimization)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        def objective(parameters: np.ndarray) -> float:
            attacks = parameters[:team_count]
            defenses = parameters[team_count : team_count * 2]
            intercept, home_advantage, rho = parameters[-3:]
            home_rate = np.exp(intercept + home_advantage + attacks[home_index] + defenses[away_index])
            away_rate = np.exp(intercept + attacks[away_index] + defenses[home_index])
            log_probability = (
                home_goals * np.log(home_rate)
                - home_rate
                - gammaln(home_goals + 1)
                + away_goals * np.log(away_rate)
                - away_rate
                - gammaln(away_goals + 1)
            )
            tau = np.ones(len(data))
            tau[mask_00] = 1 - home_rate[mask_00] * away_rate[mask_00] * rho
            tau[mask_01] = 1 + home_rate[mask_01] * rho
            tau[mask_10] = 1 + away_rate[mask_10] * rho
            tau[mask_11] = 1 - rho
            if np.any(tau <= 0):
                return 1e12
            ridge = regularization * (np.square(attacks).sum() + np.square(defenses).sum())
            centering = 10.0 * np.square(attacks.mean())
            return float(-(weights * (log_probability + np.log(tau))).sum() + ridge + centering)

        bounds = [(-2.5, 2.5)] * (team_count * 2) + [(-1.5, 1.5), (-0.5, 1.0), (-0.2, 0.2)]
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "maxfun": 200000, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Dixon-Coles训练未收敛：{result.message}")

        parameters = result.x
        attacks_array = parameters[:team_count]
        attacks_array = attacks_array - attacks_array.mean()
        defenses_array = parameters[team_count : team_count * 2]
        trained_at = datetime.now(UTC).isoformat()
        metrics: dict[str, float | int | str] = {
            "matches": len(data),
            "teams": team_count,
            "weighted_nll_per_match": float(result.fun / weights.sum()),
            "decay": decay,
            "optimizer_iterations": int(result.nit),
            "competition": competition,
        }
        return cls(
            competition=competition,
            teams=teams,
            attacks=dict(zip(teams, attacks_array, strict=True)),
            defenses=dict(zip(teams, defenses_array, strict=True)),
            intercept=float(parameters[-3]),
            home_advantage=float(parameters[-2]),
            rho=float(parameters[-1]),
            trained_at=trained_at,
            training_cutoff=latest.isoformat(),
            metrics=metrics,
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home_attack = self.attacks.get(home_team, 0.0)
        home_defense = self.defenses.get(home_team, 0.0)
        away_attack = self.attacks.get(away_team, 0.0)
        away_defense = self.defenses.get(away_team, 0.0)
        home_rate = np.exp(self.intercept + self.home_advantage + home_attack + away_defense)
        away_rate = np.exp(self.intercept + away_attack + home_defense)
        # Minimum xG floor: international football averages ~2.5 goals/game
        # Each team should have at least 0.7 xG in a competitive match
        min_xg = 0.7
        return float(np.clip(home_rate, min_xg, 5.0)), float(np.clip(away_rate, min_xg, 5.0))

    def expected_goals_cached(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Cached version of expected_goals for repeated calls."""
        home_attack = self.attacks.get(home_team, 0.0)
        home_defense = self.defenses.get(home_team, 0.0)
        away_attack = self.attacks.get(away_team, 0.0)
        away_defense = self.defenses.get(away_team, 0.0)
        home_rate = np.exp(self.intercept + self.home_advantage + home_attack + away_defense)
        away_rate = np.exp(self.intercept + away_attack + home_defense)
        return float(np.clip(home_rate, 0.1, 5.0)), float(np.clip(away_rate, 0.1, 5.0))

    def save(self, artifact_path: str | Path) -> Path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, artifact_path: str | Path) -> DixonColesModel:
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        return cls(**payload)
