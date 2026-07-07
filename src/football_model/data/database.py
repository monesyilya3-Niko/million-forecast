from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id VARCHAR PRIMARY KEY,
    kickoff TIMESTAMP,
    competition VARCHAR NOT NULL,
    season VARCHAR,
    home_team VARCHAR NOT NULL,
    away_team VARCHAR NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    status VARCHAR DEFAULT 'scheduled',
    source VARCHAR,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    match_id VARCHAR NOT NULL,
    captured_at TIMESTAMP NOT NULL,
    market VARCHAR NOT NULL,
    selection VARCHAR NOT NULL,
    odds DOUBLE NOT NULL,
    goal_line VARCHAR,
    source VARCHAR,
    PRIMARY KEY (match_id, captured_at, market, selection)
);

CREATE TABLE IF NOT EXISTS sporttery_matches (
    match_id VARCHAR PRIMARY KEY,
    official_match_id BIGINT NOT NULL,
    business_date DATE NOT NULL,
    match_number VARCHAR NOT NULL,
    kickoff TIMESTAMP NOT NULL,
    weekday VARCHAR,
    league_id VARCHAR,
    league_name VARCHAR NOT NULL,
    home_team_id BIGINT,
    home_team VARCHAR NOT NULL,
    away_team_id BIGINT,
    away_team VARCHAR NOT NULL,
    sell_status VARCHAR,
    match_status VARCHAR,
    remark VARCHAR,
    had_single BOOLEAN DEFAULT FALSE,
    hhad_single BOOLEAN DEFAULT FALSE,
    available_pools VARCHAR,
    last_update TIMESTAMP,
    source VARCHAR DEFAULT 'sporttery.cn'
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR PRIMARY KEY,
    match_id VARCHAR,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cutoff_at TIMESTAMP,
    home_probability DOUBLE NOT NULL,
    draw_probability DOUBLE NOT NULL,
    away_probability DOUBLE NOT NULL,
    home_xg DOUBLE NOT NULL,
    away_xg DOUBLE NOT NULL,
    confidence INTEGER,
    components_json VARCHAR,
    input_odds_json VARCHAR
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_type VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    artifact_path VARCHAR,
    metrics_json VARCHAR,
    status VARCHAR DEFAULT 'candidate',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_fixtures (
    match_id VARCHAR,
    provider VARCHAR,
    provider_fixture_id BIGINT,
    home_provider_team_id BIGINT,
    away_provider_team_id BIGINT,
    venue VARCHAR,
    status VARCHAR,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (match_id, provider)
);

CREATE TABLE IF NOT EXISTS lineup_snapshots (
    match_id VARCHAR,
    provider_fixture_id BIGINT,
    team_side VARCHAR,
    is_current BOOLEAN,
    formation VARCHAR,
    confirmed BOOLEAN,
    players_json VARCHAR,
    captured_at TIMESTAMP,
    PRIMARY KEY (match_id, team_side, is_current, captured_at)
);

CREATE TABLE IF NOT EXISTS injury_snapshots (
    match_id VARCHAR,
    team_side VARCHAR,
    players_json VARCHAR,
    captured_at TIMESTAMP,
    PRIMARY KEY (match_id, team_side, captured_at)
);

CREATE TABLE IF NOT EXISTS match_results (
    match_id VARCHAR PRIMARY KEY,
    status VARCHAR,
    home_goals INTEGER,
    away_goals INTEGER,
    halftime_home_goals INTEGER,
    halftime_away_goals INTEGER,
    provider VARCHAR,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS match_live_states (
    match_id VARCHAR PRIMARY KEY,
    status VARCHAR DEFAULT 'scheduled',
    minute INTEGER,
    stoppage_time INTEGER,
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    home_red_cards INTEGER DEFAULT 0,
    away_red_cards INTEGER DEFAULT 0,
    home_yellow_cards INTEGER DEFAULT 0,
    away_yellow_cards INTEGER DEFAULT 0,
    home_corners INTEGER DEFAULT 0,
    away_corners INTEGER DEFAULT 0,
    home_shots INTEGER DEFAULT 0,
    away_shots INTEGER DEFAULT 0,
    home_shots_on_target INTEGER DEFAULT 0,
    away_shots_on_target INTEGER DEFAULT 0,
    home_xg DOUBLE,
    away_xg DOUBLE,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    provider VARCHAR DEFAULT 'unknown',
    data_quality DOUBLE DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS match_contexts (
    match_id VARCHAR PRIMARY KEY,
    league_name VARCHAR,
    home_team VARCHAR,
    away_team VARCHAR,
    kickoff TIMESTAMP,
    venue VARCHAR,
    weather VARCHAR,
    referee VARCHAR,
    importance_level VARCHAR DEFAULT 'normal',
    schedule_pressure VARCHAR DEFAULT 'normal',
    data_quality DOUBLE DEFAULT 0.0,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_profiles (
    team_name VARCHAR NOT NULL,
    league VARCHAR NOT NULL,
    ranking INTEGER,
    points INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    xg_for DOUBLE,
    xg_against DOUBLE,
    home_strength DOUBLE,
    away_strength DOUBLE,
    form_last_5 VARCHAR,
    form_last_10 VARCHAR,
    elo_rating DOUBLE,
    attack_style VARCHAR,
    defense_style VARCHAR,
    set_piece_strength DOUBLE,
    transition_strength DOUBLE,
    pressing_level VARCHAR,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_name, league)
);

CREATE TABLE IF NOT EXISTS previous_match_reports (
    match_id VARCHAR NOT NULL,
    team_name VARCHAR NOT NULL,
    opponent VARCHAR,
    venue VARCHAR,
    match_date DATE,
    score VARCHAR,
    result VARCHAR,
    formation VARCHAR,
    possession DOUBLE,
    shots INTEGER,
    shots_on_target INTEGER,
    xg DOUBLE,
    key_events VARCHAR,
    goals_scored VARCHAR,
    goals_conceded VARCHAR,
    fatigue_level VARCHAR,
    red_cards INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    substitution_summary VARCHAR,
    tactical_summary VARCHAR,
    impact_on_next VARCHAR,
    data_source VARCHAR DEFAULT 'manual',
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (match_id, team_name)
);

CREATE TABLE IF NOT EXISTS tactical_reports (
    match_id VARCHAR PRIMARY KEY,
    home_formation VARCHAR,
    away_formation VARCHAR,
    home_attack_style VARCHAR,
    away_attack_style VARCHAR,
    home_defense_style VARCHAR,
    away_defense_style VARCHAR,
    home_pressing_level VARCHAR,
    away_pressing_level VARCHAR,
    home_counter_attack DOUBLE,
    away_counter_attack DOUBLE,
    home_wing_strength DOUBLE,
    away_wing_strength DOUBLE,
    home_midfield_control DOUBLE,
    away_midfield_control DOUBLE,
    home_set_piece_threat DOUBLE,
    away_set_piece_threat DOUBLE,
    home_defensive_weakness VARCHAR,
    away_defensive_weakness VARCHAR,
    key_matchups VARCHAR,
    tactical_advantage VARCHAR,
    expected_changes VARCHAR,
    probability_impact VARCHAR,
    data_source VARCHAR DEFAULT 'manual',
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id VARCHAR PRIMARY KEY,
    model_version VARCHAR NOT NULL,
    competition VARCHAR,
    train_start DATE,
    train_end DATE,
    validation_start DATE,
    validation_end DATE,
    test_start DATE,
    test_end DATE,
    features_json VARCHAR,
    log_loss DOUBLE,
    brier_score DOUBLE,
    accuracy DOUBLE,
    calibration_ece DOUBLE,
    roi_backtest DOUBLE,
    closing_line_value DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_sync_logs (
    log_id VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    sync_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    records_synced INTEGER DEFAULT 0,
    error_message VARCHAR,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER
);

CREATE TABLE IF NOT EXISTS p3_draws (
    issue_no VARCHAR PRIMARY KEY,
    draw_date DATE NOT NULL,
    digit_1 INTEGER NOT NULL,
    digit_2 INTEGER NOT NULL,
    digit_3 INTEGER NOT NULL,
    number_text VARCHAR NOT NULL,
    sum_value INTEGER,
    span_value INTEGER,
    odd_count INTEGER,
    even_count INTEGER,
    big_count INTEGER,
    small_count INTEGER,
    pattern_type VARCHAR,
    road_012 VARCHAR,
    source VARCHAR DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dlt_draws (
    issue_no VARCHAR PRIMARY KEY,
    draw_date DATE NOT NULL,
    front_1 INTEGER NOT NULL,
    front_2 INTEGER NOT NULL,
    front_3 INTEGER NOT NULL,
    front_4 INTEGER NOT NULL,
    front_5 INTEGER NOT NULL,
    back_1 INTEGER NOT NULL,
    back_2 INTEGER NOT NULL,
    front_sum INTEGER,
    back_sum INTEGER,
    front_span INTEGER,
    back_span INTEGER,
    front_odd_count INTEGER,
    front_even_count INTEGER,
    zone_1_count INTEGER,
    zone_2_count INTEGER,
    zone_3_count INTEGER,
    source VARCHAR DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lottery_predictions (
    id VARCHAR PRIMARY KEY,
    lottery_type VARCHAR NOT NULL,
    issue_no VARCHAR,
    target_issue_no VARCHAR,
    model_version VARCHAR NOT NULL,
    prediction_type VARCHAR,
    numbers_json VARCHAR,
    score DOUBLE,
    confidence DOUBLE,
    risk_level VARCHAR,
    explanation VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lottery_backtests (
    id VARCHAR PRIMARY KEY,
    lottery_type VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    backtest_start VARCHAR,
    backtest_end VARCHAR,
    strategy_name VARCHAR,
    total_issues INTEGER,
    hit_count INTEGER,
    hit_rate DOUBLE,
    average_score DOUBLE,
    max_drawdown DOUBLE,
    cost_amount DOUBLE,
    return_amount DOUBLE,
    roi DOUBLE,
    metrics_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class LocalDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: duckdb.DuckDBPyConnection | None = None
        logger.info(f"Database initialized at {self.path}")

    @contextmanager
    def connection(self, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        """Context manager for database connections with proper cleanup."""
        # Reuse existing connection if available (DuckDB doesn't allow mixed read-only/read-write)
        if self._connection is not None:
            try:
                yield self._connection
                return
            except Exception:
                self._connection = None
                raise

        connection = duckdb.connect(str(self.path), read_only=False)
        self._connection = connection
        try:
            yield connection
        except duckdb.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            # Don't close - keep for reuse
            pass

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(SCHEMA_SQL)
            connection.execute("ALTER TABLE odds_snapshots ADD COLUMN IF NOT EXISTS goal_line VARCHAR")
            connection.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence INTEGER")
            connection.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS components_json VARCHAR")
            connection.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS input_odds_json VARCHAR")
            connection.execute(
                """
                INSERT INTO model_registry (model_id, model_type, version, artifact_path, metrics_json, status)
                SELECT 'dixon-coles-baseline', 'Dixon-Coles', '0.1.0', NULL,
                       '{"scope":"manual-input","calibrated":false}', 'active'
                WHERE NOT EXISTS (
                    SELECT 1 FROM model_registry WHERE model_id = 'dixon-coles-baseline'
                )
                """
            )

    def health_check(self) -> bool:
        try:
            with self.connection(read_only=True) as connection:
                return connection.execute("SELECT 1").fetchone() == (1,)
        except duckdb.Error:
            return False

    def table_counts(self) -> dict[str, int]:
        tables = ["matches", "sporttery_matches", "odds_snapshots", "predictions", "model_registry"]
        with self.connection(read_only=True) as connection:
            return {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
