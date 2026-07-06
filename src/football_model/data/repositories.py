from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from .database import LocalDatabase

logger = logging.getLogger(__name__)


class MatchRepository:
    required_columns = {"kickoff", "competition", "home_team", "away_team"}

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def validate(self, frame: pd.DataFrame) -> list[str]:
        missing = sorted(self.required_columns - set(frame.columns))
        errors = [f"缺少字段：{', '.join(missing)}"] if missing else []
        if not missing and frame[list(self.required_columns)].isna().any().any():
            errors.append("比赛时间、赛事和球队名称不能留空")
        return errors

    @staticmethod
    def _stable_match_id(row: pd.Series) -> str:
        identity = "|".join(str(row.get(column, "")) for column in ("competition", "kickoff", "home_team", "away_team"))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def import_frame(self, frame: pd.DataFrame, *, source: str = "csv") -> int:
        """Import match data with validation and batch processing."""
        logger.info(f"Importing {len(frame)} matches from {source}")
        errors = self.validate(frame)
        if errors:
            raise ValueError("；".join(errors))

        normalized = frame.copy()
        normalized["kickoff"] = pd.to_datetime(normalized["kickoff"], errors="raise")
        if "match_id" not in normalized:
            normalized["match_id"] = normalized.apply(self._stable_match_id, axis=1)
        defaults = {
            "season": "",
            "home_goals": None,
            "away_goals": None,
            "status": "scheduled",
        }
        for column, default in defaults.items():
            if column not in normalized:
                normalized[column] = default
        normalized["source"] = source
        columns = [
            "match_id",
            "kickoff",
            "competition",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "status",
            "source",
        ]
        payload = normalized[columns]
        with self.database.connection() as connection:
            connection.register("incoming_matches", payload)
            connection.execute("INSERT OR REPLACE INTO matches BY NAME SELECT * FROM incoming_matches")
        return len(payload)

    def recent(self, limit: int = 100) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT match_id, kickoff, competition, season, home_team, away_team,
                       home_goals, away_goals, status, source
                FROM matches ORDER BY kickoff DESC LIMIT ?
                """,
                [limit],
            ).df()

    def competitions(self) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT competition, COUNT(*) AS matches, MIN(kickoff) AS first_match,
                       MAX(kickoff) AS last_match, COUNT(DISTINCT home_team) AS teams
                FROM matches WHERE status = 'completed'
                GROUP BY competition ORDER BY matches DESC
                """
            ).df()

    def training_frame(self, competition: str) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT match_id, kickoff, competition, season, home_team, away_team,
                       home_goals, away_goals
                FROM matches
                WHERE competition = ? AND status = 'completed'
                  AND home_goals IS NOT NULL AND away_goals IS NOT NULL
                ORDER BY kickoff
                """,
                [competition],
            ).df()


class OddsRepository:
    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def import_frame(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        required = {"match_id", "captured_at", "market", "selection", "odds", "source"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"赔率数据缺少字段：{', '.join(sorted(missing))}")
        payload = frame[list(required)].copy()
        payload["goal_line"] = frame["goal_line"] if "goal_line" in frame else None
        payload["captured_at"] = pd.to_datetime(payload["captured_at"])
        payload["odds"] = pd.to_numeric(payload["odds"], errors="raise")
        with self.database.connection() as connection:
            connection.register("incoming_odds", payload)
            connection.execute("INSERT OR REPLACE INTO odds_snapshots BY NAME SELECT * FROM incoming_odds")
        return len(payload)


class SportteryRepository:
    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def upsert(self, matches: pd.DataFrame, odds: pd.DataFrame) -> tuple[int, int]:
        match_count = 0
        odds_count = 0
        if not matches.empty:
            with self.database.connection() as connection:
                connection.register("incoming_sporttery_matches", matches)
                connection.execute(
                    "INSERT OR REPLACE INTO sporttery_matches BY NAME SELECT * FROM incoming_sporttery_matches"
                )
            match_count = len(matches)
        if not odds.empty:
            odds_count = OddsRepository(self.database).import_frame(odds)
        return match_count, odds_count

    def dates(self) -> list[str]:
        with self.database.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT DISTINCT business_date FROM sporttery_matches ORDER BY business_date"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def matches_for_date(self, business_date: str) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                WITH ranked_odds AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY match_id, market, selection ORDER BY captured_at DESC
                    ) AS row_number
                    FROM odds_snapshots
                    WHERE match_id LIKE 'sporttery:%'
                ), latest AS (
                    SELECT * FROM ranked_odds WHERE row_number = 1
                ), pivoted AS (
                    SELECT match_id,
                        MAX(CASE WHEN market = 'HAD' AND selection = 'H' THEN odds END) AS had_h,
                        MAX(CASE WHEN market = 'HAD' AND selection = 'D' THEN odds END) AS had_d,
                        MAX(CASE WHEN market = 'HAD' AND selection = 'A' THEN odds END) AS had_a,
                        MAX(CASE WHEN market = 'HHAD' AND selection = 'H' THEN odds END) AS hhad_h,
                        MAX(CASE WHEN market = 'HHAD' AND selection = 'D' THEN odds END) AS hhad_d,
                        MAX(CASE WHEN market = 'HHAD' AND selection = 'A' THEN odds END) AS hhad_a,
                        MAX(CASE WHEN market = 'HHAD' THEN goal_line END) AS goal_line,
                        MAX(captured_at) AS odds_updated_at
                    FROM latest GROUP BY match_id
                )
                SELECT m.*, p.had_h, p.had_d, p.had_a, p.hhad_h, p.hhad_d, p.hhad_a,
                       p.goal_line, p.odds_updated_at
                FROM sporttery_matches m
                LEFT JOIN pivoted p USING (match_id)
                WHERE m.business_date = ?
                ORDER BY m.match_number
                """,
                [business_date],
            ).df()

    def odds_history(self, match_id: str) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT captured_at, market, selection, odds, goal_line, source
                FROM odds_snapshots
                WHERE match_id = ? AND market IN ('HAD', 'HHAD')
                ORDER BY captured_at, market, selection
                """,
                [match_id],
            ).df()


class ModelRepository:
    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def list_models(self) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT model_id, model_type, version, status, artifact_path, metrics_json, created_at
                FROM model_registry ORDER BY created_at DESC
                """
            ).df()

    def register(
        self,
        *,
        model_id: str,
        model_type: str,
        version: str,
        artifact_path: str | Path,
        metrics: dict[str, object],
        status: str = "candidate",
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_registry
                    (model_id, model_type, version, artifact_path, metrics_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [model_id, model_type, version, str(artifact_path), json.dumps(metrics, ensure_ascii=False), status],
            )

    def trained_models(self) -> pd.DataFrame:
        with self.database.connection(read_only=True) as connection:
            return connection.execute(
                """
                SELECT model_id, model_type, version, status, artifact_path, metrics_json, created_at
                FROM model_registry
                WHERE artifact_path IS NOT NULL
                ORDER BY created_at DESC
                """
            ).df()

    def latest_for_competition(self, competition: str, *, model_type: str | None = None) -> pd.Series | None:
        models = self.trained_models()
        for _, model in models.iterrows():
            if model_type is not None and model["model_type"] != model_type:
                continue
            try:
                metrics = json.loads(model["metrics_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            if metrics.get("competition") == competition:
                return model
        return None


class PredictionRepository:
    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def save(
        self,
        *,
        match_id: str,
        model_version: str,
        cutoff_at: object,
        home_probability: float,
        draw_probability: float,
        away_probability: float,
        home_xg: float,
        away_xg: float,
        confidence: int,
        components: dict[str, object],
        input_odds: dict[str, float],
    ) -> str:
        identity = f"{match_id}|{model_version}|{pd.Timestamp(cutoff_at).isoformat()}"
        prediction_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    prediction_id, match_id, model_version, created_at, cutoff_at,
                    home_probability, draw_probability, away_probability,
                    home_xg, away_xg, confidence, components_json, input_odds_json
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id,
                    match_id,
                    model_version,
                    pd.Timestamp(cutoff_at),
                    home_probability,
                    draw_probability,
                    away_probability,
                    home_xg,
                    away_xg,
                    confidence,
                    json.dumps(components, ensure_ascii=False),
                    json.dumps(input_odds, ensure_ascii=False),
                ],
            )
        return prediction_id
