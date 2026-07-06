"""Match live state service.

Manages real-time match state data including scores, cards,
corners, shots, and sync logging.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from football_model.data import LocalDatabase
from football_model.providers.base import ProviderHealth
from football_model.providers.live_state_provider import LiveStateProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchLiveState:
    """Current live state of a match."""

    match_id: str
    status: str  # scheduled, live, halftime, finished, postponed, cancelled
    minute: int | None = None
    stoppage_time: int | None = None
    home_score: int = 0
    away_score: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_xg: float | None = None
    away_xg: float | None = None
    last_update: datetime | None = None
    provider: str = "unknown"
    data_quality: float = 0.0


@dataclass(frozen=True)
class SyncResult:
    """Result of a sync operation."""

    provider: str
    sync_type: str
    success: bool
    records_synced: int = 0
    error_message: str | None = None
    duration_ms: int = 0


class MatchLiveStateService:
    """Service for managing live match states.

    Syncs with external providers and stores state in database.
    """

    def __init__(self, database: LocalDatabase, provider: LiveStateProvider | None = None) -> None:
        self.database = database
        self.provider = provider or LiveStateProvider()

    def get_live_state(self, match_id: str) -> MatchLiveState | None:
        """Get current live state for a match."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                """SELECT match_id, status, minute, stoppage_time,
                   home_score, away_score,
                   home_red_cards, away_red_cards,
                   home_yellow_cards, away_yellow_cards,
                   home_corners, away_corners,
                   home_shots, away_shots,
                   home_shots_on_target, away_shots_on_target,
                   home_xg, away_xg,
                   last_update, provider, data_quality
                FROM match_live_states WHERE match_id = ?""",
                [match_id],
            ).fetchone()

        if not row:
            return None

        return MatchLiveState(
            match_id=row[0],
            status=row[1] or "scheduled",
            minute=row[2],
            stoppage_time=row[3],
            home_score=row[4] or 0,
            away_score=row[5] or 0,
            home_red_cards=row[6] or 0,
            away_red_cards=row[7] or 0,
            home_yellow_cards=row[8] or 0,
            away_yellow_cards=row[9] or 0,
            home_corners=row[10] or 0,
            away_corners=row[11] or 0,
            home_shots=row[12] or 0,
            away_shots=row[13] or 0,
            home_shots_on_target=row[14] or 0,
            away_shots_on_target=row[15] or 0,
            home_xg=row[16],
            away_xg=row[17],
            last_update=row[18],
            provider=row[19] or "unknown",
            data_quality=row[20] or 0.0,
        )

    def save_live_state(self, state: MatchLiveState) -> None:
        """Save live state to database."""
        with self.database.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO match_live_states
                (match_id, status, minute, stoppage_time,
                 home_score, away_score,
                 home_red_cards, away_red_cards,
                 home_yellow_cards, away_yellow_cards,
                 home_corners, away_corners,
                 home_shots, away_shots,
                 home_shots_on_target, away_shots_on_target,
                 home_xg, away_xg,
                 last_update, provider, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    state.match_id,
                    state.status,
                    state.minute,
                    state.stoppage_time,
                    state.home_score,
                    state.away_score,
                    state.home_red_cards,
                    state.away_red_cards,
                    state.home_yellow_cards,
                    state.away_yellow_cards,
                    state.home_corners,
                    state.away_corners,
                    state.home_shots,
                    state.away_shots,
                    state.home_shots_on_target,
                    state.away_shots_on_target,
                    state.home_xg,
                    state.away_xg,
                    state.last_update or datetime.now(),
                    state.provider,
                    state.data_quality,
                ],
            )

    def sync_match_live_state(self, match_id: str) -> SyncResult:
        """Sync live state for a single match from provider."""
        import time
        start = time.time()

        # Get provider fixture ID
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT provider_fixture_id FROM provider_fixtures WHERE match_id = ?",
                [match_id],
            ).fetchone()

        if not row:
            return SyncResult(
                provider=self.provider.name,
                sync_type="live_state",
                success=False,
                error_message="No provider fixture ID found",
                duration_ms=int((time.time() - start) * 1000),
            )

        fixture_id = int(row[0])
        result = self.provider.fetch_live_state_by_fixture_id(fixture_id)

        if not result.success:
            self._log_sync("live_state", "failed", 0, result.error_message)
            return SyncResult(
                provider=self.provider.name,
                sync_type="live_state",
                success=False,
                error_message=result.error_message,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Save to database
        data = result.data
        state = MatchLiveState(
            match_id=match_id,
            status=data.get("status", "unknown"),
            minute=data.get("minute"),
            stoppage_time=data.get("stoppage_time"),
            home_score=data.get("home_score", 0) or 0,
            away_score=data.get("away_score", 0) or 0,
            home_red_cards=data.get("home_red_cards", 0),
            away_red_cards=data.get("away_red_cards", 0),
            home_yellow_cards=data.get("home_yellow_cards", 0),
            away_yellow_cards=data.get("away_yellow_cards", 0),
            home_corners=data.get("home_corners", 0) or 0,
            away_corners=data.get("away_corners", 0) or 0,
            home_shots=data.get("home_shots", 0) or 0,
            away_shots=data.get("away_shots", 0) or 0,
            home_shots_on_target=data.get("home_shots_on_target", 0) or 0,
            away_shots_on_target=data.get("away_shots_on_target", 0) or 0,
            last_update=datetime.now(),
            provider=self.provider.name,
            data_quality=self._calculate_data_quality(data),
        )

        self.save_live_state(state)
        self._log_sync("live_state", "success", 1)

        duration = int((time.time() - start) * 1000)
        return SyncResult(
            provider=self.provider.name,
            sync_type="live_state",
            success=True,
            records_synced=1,
            duration_ms=duration,
        )

    def sync_all_pending(self) -> list[SyncResult]:
        """Sync live state for all matches that need updating."""
        results = []

        # Get matches that are live or upcoming
        with self.database.connection(read_only=True) as conn:
            matches = conn.execute("""
                SELECT s.match_id, s.kickoff, s.match_status
                FROM sporttery_matches s
                LEFT JOIN match_live_states l ON s.match_id = l.match_id
                WHERE s.kickoff >= CURRENT_TIMESTAMP - INTERVAL '4 hours'
                  AND (l.match_id IS NULL OR l.status NOT IN ('finished', 'cancelled'))
                ORDER BY s.kickoff
            """).fetchall()

        for match in matches:
            match_id = match[0]
            result = self.sync_match_live_state(match_id)
            results.append(result)

        return results

    def get_provider_health(self) -> ProviderHealth:
        """Check provider health status."""
        return self.provider.health_check()

    def get_sync_logs(self, limit: int = 50) -> pd.DataFrame:
        """Get recent sync logs."""
        with self.database.connection(read_only=True) as conn:
            return conn.execute("""
                SELECT log_id, provider, sync_type, status,
                       records_synced, error_message,
                       started_at, completed_at, duration_ms
                FROM provider_sync_logs
                ORDER BY started_at DESC
                LIMIT ?
            """, [limit]).df()

    def _calculate_data_quality(self, data: dict) -> float:
        """Calculate data quality score (0-1)."""
        score = 0.0
        checks = 0

        # Score available
        if data.get("home_score") is not None:
            score += 1.0
        checks += 1

        # Minute available
        if data.get("minute") is not None:
            score += 1.0
        checks += 1

        # Cards available
        if data.get("home_yellow_cards") is not None:
            score += 0.5
        checks += 1

        # Shots available
        if data.get("home_shots") is not None:
            score += 0.5
        checks += 1

        # Corners available
        if data.get("home_corners") is not None:
            score += 0.5
        checks += 1

        return score / checks if checks > 0 else 0.0

    def _log_sync(
        self,
        sync_type: str,
        status: str,
        records: int,
        error: str | None = None,
    ) -> None:
        """Log sync operation to database."""
        log_id = hashlib.sha256(
            f"{self.provider.name}:{sync_type}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        with self.database.connection() as conn:
            conn.execute(
                """INSERT INTO provider_sync_logs
                (log_id, provider, sync_type, status, records_synced, error_message, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [log_id, self.provider.name, sync_type, status, records, error],
            )
