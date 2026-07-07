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
        decay: float | None = None,
        regularization: float | None = None,
        maxiter: int = 100_000,
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
        team_count = len(teams)

        # Adaptive decay: target 10% weight for oldest data
        if decay is None:
            max_age = float(age_days.max())
            decay = -np.log(0.10) / max_age if max_age > 0 else 0.0015
        weights = np.exp(-decay * age_days)

        # Adaptive regularization: scale with sqrt(teams)
        if regularization is None:
            regularization = 0.02 * np.sqrt(team_count / 20.0)

        mean_goals = max(float((home_goals.sum() + away_goals.sum()) / (2 * len(data))), 0.2)
        home_mean = max(float(home_goals.mean()), 0.2)
        away_mean = max(float(away_goals.mean()), 0.2)
        home_advantage_init = np.log(home_mean / away_mean) if away_mean > 0 else 0.2

        # Empirical rho estimation from low-score matches
        low_score_mask = (home_goals <= 1) & (away_goals <= 1)
        if low_score_mask.sum() > 50:
            obs_00 = float(((home_goals == 0) & (away_goals == 0)).sum()) / len(data)
            obs_11 = float(((home_goals == 1) & (away_goals == 1)).sum()) / len(data)
            exp_00 = np.exp(-home_mean) * np.exp(-away_mean)
            exp_11 = home_mean * np.exp(-home_mean) * away_mean * np.exp(-away_mean)
            rho_candidates = []
            if exp_00 > 0.01:
                rho_candidates.append((obs_00 / exp_00 - 1) / (home_mean * away_mean) if home_mean * away_mean > 0 else 0)
            if exp_11 > 0.01:
                rho_candidates.append((1 - obs_11 / exp_11))
            rho_init = float(np.clip(np.median(rho_candidates) if rho_candidates else -0.05, -0.3, 0.1))
        else:
            rho_init = -0.05

        initial = np.zeros(team_count * 2 + 3)
        initial[-3] = np.log(mean_goals)
        initial[-2] = np.clip(home_advantage_init, 0.0, 0.5)
        initial[-1] = rho_init

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
            options={"maxiter": maxiter, "maxfun": maxiter * 40, "ftol": 1e-12},
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
            "decay": float(decay),
            "regularization": float(regularization),
            "maxiter": maxiter,
            "optimizer_iterations": int(result.nit),
            "optimizer_evaluations": int(result.nfev),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "rho_init": float(rho_init),
            "home_advantage_init": float(home_advantage_init),
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
