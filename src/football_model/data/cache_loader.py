"""Cache loading utilities for HuggingFace Spaces deployment.

Loads cached JSON data into DuckDB tables on first run when
the sporttery API is not accessible (e.g., overseas servers).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_cache_if_empty(
    db: LocalDatabase,
    cache_name: str,
    table: str,
    transform=None,
) -> None:
    """Load cached JSON data into database table if empty."""
    try:
        with db.connection(read_only=True) as conn:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if count > 0:
            return
        cache_path = _CACHE_DIR / cache_name
        if not cache_path.exists():
            return
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        if not records:
            return
        with db.connection() as conn:
            for rec in records:
                try:
                    if transform:
                        transform(conn, rec)
                except Exception:
                    pass
        logger.info("Loaded %d records from %s into %s", len(records), cache_name, table)
    except Exception as e:
        logger.warning("Cache load failed for %s: %s", cache_name, e)


def _ts_to_str(ts, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
    """Convert millisecond timestamp to string."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(fmt)
    return ts


def _transform_sporttery(conn, _m: dict) -> None:
    _kickoff = _ts_to_str(_m.get("kickoff"))
    _biz = _ts_to_str(_m.get("business_date"), "%Y-%m-%d")
    _last_upd = _ts_to_str(_m.get("last_update"))
    conn.execute(
        """INSERT OR IGNORE INTO sporttery_matches
        (match_id, official_match_id, business_date, match_number, kickoff,
         weekday, league_id, league_name, home_team_id, home_team,
         away_team_id, away_team, sell_status, match_status, remark,
         had_single, hhad_single, available_pools, last_update, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [_m.get("match_id"), _m.get("official_match_id", 0),
         _biz, _m.get("match_number", ""), _kickoff,
         _m.get("weekday", ""), _m.get("league_id", ""), _m.get("league_name", ""),
         _m.get("home_team_id", 0), _m.get("home_team", ""),
         _m.get("away_team_id", 0), _m.get("away_team", ""),
         _m.get("sell_status", ""), _m.get("match_status", ""), _m.get("remark", ""),
         _m.get("had_single", False), _m.get("hhad_single", False),
         _m.get("available_pools", ""), _last_upd or _kickoff,
         _m.get("source", "sporttery.cn")],
    )


def _transform_matches(conn, _m: dict) -> None:
    kickoff = _m.get("kickoff")
    if isinstance(kickoff, str) and kickoff.isdigit():
        kickoff = datetime.fromtimestamp(int(kickoff) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR IGNORE INTO matches (match_id, competition, home_team, away_team, home_goals, away_goals, kickoff) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [_m.get("match_id"), _m.get("competition", ""), _m.get("home_team", ""),
         _m.get("away_team", ""), _m.get("home_goals"), _m.get("away_goals"), kickoff],
    )


def _transform_odds(conn, _m: dict) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO odds_snapshots (match_id, market, selection, odds, goal_line, captured_at) VALUES (?, ?, ?, ?, ?, ?)",
        [_m.get("match_id"), _m.get("market", ""), _m.get("selection", ""),
         _m.get("odds"), _m.get("goal_line", ""), _m.get("captured_at", "")],
    )


def _transform_models(conn, _m: dict) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO model_registry (model_id, version, model_type, metrics_json, artifact_path, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [_m.get("model_id"), _m.get("version", "1.0"), _m.get("model_type", ""),
         _m.get("metrics_json", "{}"), _m.get("artifact_path", ""),
         _m.get("status", "trained"), _m.get("created_at", "")],
    )


def load_all_caches(db: LocalDatabase) -> None:
    """Load all cached data into database if tables are empty."""
    _load_cache_if_empty(db, "sporttery_matches_cache.json", "sporttery_matches", _transform_sporttery)
    _load_cache_if_empty(db, "matches_cache.json", "matches", _transform_matches)
    _load_cache_if_empty(db, "odds_cache.json", "odds_snapshots", _transform_odds)
    _load_cache_if_empty(db, "models_cache.json", "model_registry", _transform_models)
