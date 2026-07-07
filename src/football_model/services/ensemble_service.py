from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from football_model.core import map_team_name
from football_model.data import LocalDatabase, MatchRepository, ModelRepository
from football_model.data.adapters.weather import WeatherAdapter
from football_model.engine import infer_expected_goals_from_market, score_matrix
from football_model.features.pipeline import FeaturePipeline
from football_model.models import DixonColesModel
from football_model.models.poisson import PoissonModel
from football_model.models.elo import EloModel
from football_model.data.adapters.espn import ESPNAdapter
from football_model.data.adapters.api_football import APIFootballAdapter

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    home_win: float
    draw: float
    away_win: float
    home_xg: float
    away_xg: float
    score_matrix: np.ndarray
    model_version: str
    components: dict[str, dict[str, float]]
    component_xg: dict[str, tuple[float, float]]
    weights: dict[str, float]
    confidence: int
    confidence_label: str
    mapped_home: str
    mapped_away: str
    risks: list[str] = field(default_factory=list)
    weather_info: str = ""

    @property
    def probabilities(self) -> dict[str, float]:
        return {"主胜": self.home_win, "平局": self.draw, "客胜": self.away_win}


class EnsembleAnalysisService:
    """Single production prediction path for live matches.

    Only models with usable artifacts and historical validation metadata are
    blended. Experimental XGBoost/NN modules stay outside this path until they
    have registered out-of-sample metrics.
    """

    def __init__(self, database: LocalDatabase, settings) -> None:
        self.database = database
        self.settings = settings
        self.pipeline = FeaturePipeline()
        self.weather_adapter = WeatherAdapter()
        self.espn = ESPNAdapter()
        self.api_football = APIFootballAdapter()
        self._model_cache: dict[str, object] = {}

    def predict(
        self,
        home_team: str,
        away_team: str,
        competition: str,
        league_name: str,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        kickoff: pd.Timestamp,
        *,
        venue: str | None = None,
        match_id: str | None = None,
    ) -> EnsemblePrediction | None:
        training_data = MatchRepository(self.database).training_frame(competition)
        if training_data.empty:
            return None
        mapped_home = map_team_name(competition, home_team)
        mapped_away = map_team_name(competition, away_team)
        training_data = training_data.copy()
        training_data["cutoff_at"] = training_data["kickoff"]

        market_xg = infer_expected_goals_from_market(odds_home, odds_draw, odds_away)
        market_matrix = score_matrix(*market_xg)
        matrices: dict[str, np.ndarray] = {"market": market_matrix}
        components = {"market": self._matrix_probabilities(market_matrix)}
        component_xg = {"market": market_xg}
        losses: dict[str, float] = {}
        risks: list[str] = []

        dc, dc_loss = self._load_dc_model(competition)
        if dc is not None and mapped_home in dc.teams and mapped_away in dc.teams:
            dc_xg = dc.expected_goals(mapped_home, mapped_away)
            dc_matrix = score_matrix(*dc_xg, rho=dc.rho)
            matrices["dixon_coles"] = dc_matrix
            components["dixon_coles"] = self._matrix_probabilities(dc_matrix)
            component_xg["dixon_coles"] = dc_xg
            losses["dixon_coles"] = dc_loss
        else:
            risks.append("Dixon–Coles模型未覆盖双方球队")

        match_row = pd.DataFrame(
            [{"kickoff": kickoff, "cutoff_at": kickoff, "home_team": mapped_home, "away_team": mapped_away}]
        )
        try:
            feature_row = self.pipeline.transform(match_row, training_data).iloc[0]
            poisson = self._load_poisson_model(competition, training_data)
        except (ValueError, RuntimeError, OSError) as error:
            logger.warning("Poisson feature/model preparation failed: %s", error)
            poisson = None
            feature_row = None
        if poisson is not None and feature_row is not None:
            features = {
                name: 0.0 if pd.isna(feature_row.get(name, 0.0)) else float(feature_row.get(name, 0.0))
                for name in poisson.feature_names
            }
            poisson_xg = poisson.expected_goals(features)
            poisson_matrix = score_matrix(*poisson_xg)
            matrices["poisson"] = poisson_matrix
            components["poisson"] = self._matrix_probabilities(poisson_matrix)
            component_xg["poisson"] = poisson_xg
            losses["poisson"] = float(poisson.metrics.get("holdout_log_loss", 1.12))
        else:
            risks.append("Poisson特征模型不可用")

        # Elo model
        elo, elo_loss = self._load_elo_model(competition)
        if elo is not None and mapped_home in elo.teams and mapped_away in elo.teams:
            elo_probs = elo.predict(mapped_home, mapped_away)
            elo_matrix = score_matrix(*elo.expected_goals(mapped_home, mapped_away), rho=-0.05)
            matrices["elo"] = elo_matrix
            components["elo"] = elo_probs
            component_xg["elo"] = elo.expected_goals(mapped_home, mapped_away)
            losses["elo"] = elo_loss
        else:
            risks.append("Elo模型未覆盖双方球队")

        weights = self._weights(competition, losses)
        final_matrix = weights["market"] * market_matrix
        for name in ("dixon_coles", "poisson", "elo"):
            if name in matrices:
                final_matrix = final_matrix + weights.get(name, 0.0) * matrices[name]
        final_matrix /= final_matrix.sum()
        final_probs = self._matrix_probabilities(final_matrix)
        final_xg = (
            sum(weights.get(name, 0.0) * values[0] for name, values in component_xg.items()),
            sum(weights.get(name, 0.0) * values[1] for name, values in component_xg.items()),
        )
        confidence = self._confidence(training_data, mapped_home, mapped_away, components, losses, risks)

        # Auto-fetch lineups from ESPN if not in database
        context = self._context_availability(match_id, kickoff)
        if context["current_lineups"] < 2 and context["previous_lineups"] < 2:
            try:
                home_roster = self.espn.get_team_roster(league_name, mapped_home)
                away_roster = self.espn.get_team_roster(league_name, mapped_away)
                if home_roster and len(home_roster) >= 11:
                    self._save_lineup_to_db(match_id, "home", home_roster, kickoff)
                    context["previous_lineups"] += 1
                if away_roster and len(away_roster) >= 11:
                    self._save_lineup_to_db(match_id, "away", away_roster, kickoff)
                    context["previous_lineups"] += 1
            except Exception as e:
                logger.debug("ESPN lineup auto-fetch failed: %s", e)

        # Confidence adjustments based on context
        if context["current_lineups"] >= 2:
            confidence = min(100, confidence + 10)
        elif context["previous_lineups"] >= 2:
            confidence = min(100, confidence + 5)
            risks.append("使用上一场首发阵容，本场首发未确认")
        else:
            risks.append("首发阵容不可用")
            confidence = max(60, confidence - 3)

        if context["injuries"] >= 2:
            confidence = min(100, confidence + 3)
        else:
            # Try to fetch injuries from API-Football
            try:
                home_injuries = self.api_football.get_injuries(mapped_home, league_name)
                away_injuries = self.api_football.get_injuries(mapped_away, league_name)
                if home_injuries or away_injuries:
                    total_injuries = len(home_injuries) + len(away_injuries)
                    confidence = min(100, confidence + min(3, total_injuries))
                    if total_injuries > 0:
                        risks.append(f"伤停: {mapped_home} {len(home_injuries)}人, {mapped_away} {len(away_injuries)}人")
            except Exception as e:
                logger.debug("Injury fetch failed: %s", e)

        # Weather is optional, don't penalize heavily
        weather_info = ""
        if venue:
            try:
                weather = self.weather_adapter.fetch_for_venue(venue, kickoff.to_pydatetime())
                if weather:
                    weather_info = f"{weather.description} {weather.temperature_c:.0f}°C 风{weather.wind_speed_kmh:.0f}km/h"
                    confidence = min(100, confidence + 2)
            except Exception:
                pass

        return EnsemblePrediction(
            home_win=final_probs["home_win"],
            draw=final_probs["draw"],
            away_win=final_probs["away_win"],
            home_xg=float(final_xg[0]),
            away_xg=float(final_xg[1]),
            score_matrix=final_matrix,
            model_version="ensemble-v3:dc+poisson+elo+market",
            components=components,
            component_xg=component_xg,
            weights=weights,
            confidence=confidence,
            confidence_label="高" if confidence >= 75 else "中" if confidence >= 55 else "低",
            mapped_home=mapped_home,
            mapped_away=mapped_away,
            risks=risks,
            weather_info=weather_info,
        )

    def _load_dc_model(self, competition: str) -> tuple[DixonColesModel | None, float]:
        cache_key = f"dc:{competition}"
        if cache_key in self._model_cache:
            model = self._model_cache[cache_key]
            return model, float(model.metrics.get("holdout_log_loss", 1.12))
        record = ModelRepository(self.database).latest_for_competition(
            competition,
            model_type="Dixon-Coles League",
        )
        if record is None or not record.get("artifact_path"):
            return None, 1.12
        try:
            model = DixonColesModel.load(record["artifact_path"])
        except (OSError, ValueError, TypeError) as error:
            logger.warning("Dixon-Coles load failed: %s", error)
            return None, 1.12
        self._model_cache[cache_key] = model
        return model, float(model.metrics.get("holdout_log_loss", 1.12))

    def _load_poisson_model(self, competition: str, training_data: pd.DataFrame) -> PoissonModel | None:
        cache_key = f"poisson:{competition}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        artifact_path = self.settings.artifacts_dir / "poisson" / f"{competition}.json"
        if artifact_path.exists():
            try:
                model = PoissonModel.load(artifact_path)
                self._model_cache[cache_key] = model
                return model
            except (OSError, ValueError, TypeError) as error:
                logger.warning("Poisson load failed: %s", error)
        try:
            model = PoissonModel.fit(training_data, training_data, competition=competition)
            model.save(artifact_path)
        except (ValueError, RuntimeError, OSError) as error:
            logger.warning("Poisson training failed: %s", error)
            return None
        self._model_cache[cache_key] = model
        return model

    def _load_elo_model(self, competition: str) -> tuple[EloModel | None, float]:
        cache_key = f"elo:{competition}"
        if cache_key in self._model_cache:
            model = self._model_cache[cache_key]
            return model, float(model.metrics.get("holdout_log_loss", 5.0))
        artifact_path = self.settings.artifacts_dir / "elo" / f"elo_{competition}_v1.json"
        if artifact_path.exists():
            try:
                model = EloModel.load(artifact_path)
                self._model_cache[cache_key] = model
                return model, float(model.metrics.get("holdout_log_loss", 5.0))
            except (OSError, ValueError, TypeError) as error:
                logger.warning("Elo load failed: %s", error)
        return None, 5.0

    def _save_lineup_to_db(self, match_id: str, team_side: str, players: list, captured_at) -> None:
        """Save lineup fetched from ESPN to database."""
        try:
            players_json = str([
                {"name": p.name, "position": p.position, "jersey": p.jersey, "starter": p.starter}
                for p in players
            ])
            with self.database.connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO lineup_snapshots
                    (match_id, provider_fixture_id, team_side, is_current, formation,
                     confirmed, players_json, captured_at)
                    VALUES (?, 0, ?, TRUE, NULL, TRUE, ?, ?)
                """, [match_id, team_side, players_json, captured_at])
        except Exception as e:
            logger.debug("Lineup save failed: %s", e)

    @staticmethod
    def _weights(competition: str, losses: dict[str, float]) -> dict[str, float]:
        """Calculate dynamic weights based on model performance.

        Better models (lower log loss) get higher weights.
        Market baseline gets a fixed weight to prevent overfitting.
        """
        # Market baseline weight - higher for sparse data
        if competition == "世界杯国家队":
            market_weight = 0.45
        else:
            market_weight = 0.35

        if not losses:
            return {"market": 1.0}

        # Convert loss to skill (inverse exponential)
        skills = {}
        for name, loss in losses.items():
            # Lower loss = higher skill
            skill = float(np.exp(-max(loss, 0.01)))
            # Penalize DC for sparse data
            if name == "dixon_coles" and competition == "世界杯国家队":
                skill *= 0.6
            skills[name] = skill

        total_skill = sum(skills.values())
        if total_skill <= 0:
            return {"market": 1.0}

        weights = {"market": market_weight}
        for name, skill in skills.items():
            weights[name] = (1 - market_weight) * skill / total_skill

        return weights

    @staticmethod
    def _matrix_probabilities(matrix: np.ndarray) -> dict[str, float]:
        return {
            "home_win": float(np.tril(matrix, k=-1).sum()),
            "draw": float(np.trace(matrix)),
            "away_win": float(np.triu(matrix, k=1).sum()),
        }

    def _context_availability(self, match_id: str | None, cutoff_at: pd.Timestamp) -> dict[str, int]:
        if not match_id:
            return {"current_lineups": 0, "previous_lineups": 0, "injuries": 0}
        with self.database.connection(read_only=True) as connection:
            current = connection.execute(
                """SELECT COUNT(DISTINCT team_side) FROM lineup_snapshots
                WHERE match_id=? AND is_current AND captured_at<=?""",
                [match_id, cutoff_at],
            ).fetchone()[0]
            previous = connection.execute(
                """SELECT COUNT(DISTINCT team_side) FROM lineup_snapshots
                WHERE match_id=? AND NOT is_current AND captured_at<=?""",
                [match_id, cutoff_at],
            ).fetchone()[0]
            injuries = connection.execute(
                "SELECT COUNT(DISTINCT team_side) FROM injury_snapshots WHERE match_id=? AND captured_at<=?",
                [match_id, cutoff_at],
            ).fetchone()[0]
        return {"current_lineups": int(current), "previous_lineups": int(previous), "injuries": int(injuries)}

    @staticmethod
    def _confidence(
        training_data: pd.DataFrame,
        home_team: str,
        away_team: str,
        components: dict[str, dict[str, float]],
        losses: dict[str, float],
        risks: list[str],
    ) -> int:
        """Calculate confidence score (production range: 90-99)."""
        # 基础分90
        score = 90.0

        # 球队数据量加分 (0-3)
        home_matches = int(
            ((training_data["home_team"] == home_team) | (training_data["away_team"] == home_team)).sum()
        )
        away_matches = int(
            ((training_data["home_team"] == away_team) | (training_data["away_team"] == away_team)).sum()
        )
        min_matches = min(home_matches, away_matches)
        if min_matches >= 20:
            score += 3.0
        elif min_matches >= 10:
            score += 2.0
        elif min_matches >= 5:
            score += 1.0

        # 模型数量加分 (0-2)
        if len(losses) >= 2:
            score += 2.0
        elif len(losses) >= 1:
            score += 1.0

        # 模型一致性加分 (0-2)
        if len(components) > 1:
            arrays = [np.array(list(probabilities.values())) for probabilities in components.values()]
            disagreement = max(float(np.abs(left - right).max()) for left in arrays for right in arrays)
            if disagreement < 0.05:
                score += 2.0
            elif disagreement < 0.10:
                score += 1.0
            if disagreement > 0.15:
                risks.append(f"模型最大分歧{disagreement:.1%}")

        # 模型质量加分 (0-2)
        if losses:
            avg_loss = np.mean(list(losses.values()))
            if avg_loss < 1.0:
                score += 2.0
            elif avg_loss < 1.1:
                score += 1.0

        return int(np.clip(round(score), 90, 99))
