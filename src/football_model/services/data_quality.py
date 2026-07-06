"""Data quality scoring service.

Evaluates the completeness and reliability of match data
for prediction purposes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityReport:
    """Data quality assessment for a match."""

    match_id: str
    overall_score: float  # 0-1
    overall_level: str  # 高 / 中 / 低

    # Component scores
    odds_score: float
    lineup_score: float
    injury_score: float
    history_score: float
    freshness_score: float

    # Flags
    has_odds: bool
    has_lineup: bool
    has_injury: bool
    has_history: bool
    is_fresh: bool

    # Risks
    risks: list[str]
    can_use_independent_model: bool
    should_use_market_only: bool


class DataQualityService:
    """Service for assessing data quality."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def assess(self, match_id: str, league: str) -> DataQualityReport:
        """Assess data quality for a match.

        Args:
            match_id: Match identifier
            league: Competition name

        Returns:
            DataQualityReport with scores and flags
        """
        risks = []

        # Check odds
        odds_score, has_odds = self._check_odds(match_id)
        if not has_odds:
            risks.append("赔率数据不完整")

        # Check lineup
        lineup_score, has_lineup = self._check_lineup(match_id)
        if not has_lineup:
            risks.append("首发阵容未确认")

        # Check injury
        injury_score, has_injury = self._check_injury(match_id)
        if not has_injury:
            risks.append("伤停数据缺失")

        # Check history
        history_score, has_history = self._check_history(league)
        if not has_history:
            risks.append("历史样本不足")

        # Check freshness
        freshness_score, is_fresh = self._check_freshness(match_id)
        if not is_fresh:
            risks.append("数据更新滞后")

        # Overall score
        scores = [odds_score, lineup_score, injury_score, history_score, freshness_score]
        weights = [0.30, 0.15, 0.10, 0.30, 0.15]
        overall = sum(s * w for s, w in zip(scores, weights, strict=True))

        # Level
        if overall >= 0.75:
            level = "高"
        elif overall >= 0.50:
            level = "中"
        else:
            level = "低"

        # Can use independent model?
        can_use_independent = has_history and has_odds
        should_use_market = not has_history or overall < 0.40

        return DataQualityReport(
            match_id=match_id,
            overall_score=overall,
            overall_level=level,
            odds_score=odds_score,
            lineup_score=lineup_score,
            injury_score=injury_score,
            history_score=history_score,
            freshness_score=freshness_score,
            has_odds=has_odds,
            has_lineup=has_lineup,
            has_injury=has_injury,
            has_history=has_history,
            is_fresh=is_fresh,
            risks=risks,
            can_use_independent_model=can_use_independent,
            should_use_market_only=should_use_market,
        )

    def _check_odds(self, match_id: str) -> tuple[float, bool]:
        """Check odds completeness."""
        with self.database.connection(read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM odds_snapshots WHERE match_id = ?",
                [match_id],
            ).fetchone()[0]

        if count >= 6:  # HAD + HHAD complete
            return 1.0, True
        elif count >= 3:  # At least HAD
            return 0.7, True
        elif count >= 1:
            return 0.4, False
        return 0.0, False

    def _check_lineup(self, match_id: str) -> tuple[float, bool]:
        """Check lineup completeness."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                """SELECT COUNT(DISTINCT team_side)
                FROM lineup_snapshots
                WHERE match_id = ? AND is_current = true""",
                [match_id],
            ).fetchone()

        sides = row[0] if row else 0
        if sides >= 2:
            return 1.0, True
        elif sides >= 1:
            return 0.5, False
        return 0.0, False

    def _check_injury(self, match_id: str) -> tuple[float, bool]:
        """Check injury data completeness."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM injury_snapshots WHERE match_id = ?",
                [match_id],
            ).fetchone()

        count = row[0] if row else 0
        if count >= 2:
            return 1.0, True
        elif count >= 1:
            return 0.5, True
        return 0.0, False

    def _check_history(self, league: str) -> tuple[float, bool]:
        """Check historical data availability."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM matches WHERE competition = ? AND status = 'completed'",
                [league],
            ).fetchone()

        count = row[0] if row else 0
        if count >= 500:
            return 1.0, True
        elif count >= 200:
            return 0.7, True
        elif count >= 50:
            return 0.4, False
        return 0.0, False

    def _check_freshness(self, match_id: str) -> tuple[float, bool]:
        """Check data freshness."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                """SELECT MAX(last_update) FROM (
                    SELECT last_update FROM sporttery_matches WHERE match_id = ?
                    UNION ALL
                    SELECT captured_at FROM odds_snapshots WHERE match_id = ?
                )""",
                [match_id, match_id],
            ).fetchone()

        if not row or not row[0]:
            return 0.0, False

        try:
            last_update = pd.to_datetime(row[0])
            hours_ago = (pd.Timestamp.now() - last_update).total_seconds() / 3600

            if hours_ago < 1:
                return 1.0, True
            elif hours_ago < 6:
                return 0.7, True
            elif hours_ago < 24:
                return 0.4, False
            return 0.1, False
        except Exception:
            return 0.0, False


def format_quality_report(report: DataQualityReport) -> str:
    """Format quality report for display."""
    parts = [
        f"数据质量评分: {report.overall_score:.0%} ({report.overall_level})",
        "",
        f"赔率: {report.odds_score:.0%} {'✅' if report.has_odds else '❌'}",
        f"阵容: {report.lineup_score:.0%} {'✅' if report.has_lineup else '❌'}",
        f"伤停: {report.injury_score:.0%} {'✅' if report.has_injury else '❌'}",
        f"历史: {report.history_score:.0%} {'✅' if report.has_history else '❌'}",
        f"新鲜度: {report.freshness_score:.0%} {'✅' if report.is_fresh else '❌'}",
    ]

    if report.risks:
        parts.append("")
        parts.append("风险因素:")
        for risk in report.risks:
            parts.append(f"  - {risk}")

    return "\n".join(parts)
