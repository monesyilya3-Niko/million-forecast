"""Value analysis service.

Calculates Expected Value (Kelly Criterion, risk-adjusted sizing)
for betting opportunities with proper risk warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueAnalysis:
    """Value analysis for a single outcome."""

    outcome: str  # home_win / draw / away_win
    label: str  # 主胜 / 平局 / 客胜
    odds: float
    market_prob: float
    model_prob: float
    probability_diff: float
    ev: float  # Expected Value
    kelly_fraction: float  # Full Kelly
    recommended_kelly: float  # Fractional Kelly (conservative)
    recommended_stake_pct: float  # Recommended stake as % of bankroll
    is_value: bool
    risk_warning: str


@dataclass(frozen=True)
class MatchValueReport:
    """Value analysis report for a match."""

    match_id: str
    outcomes: list[ValueAnalysis]
    best_value: ValueAnalysis | None
    has_value: bool
    overall_risk: str
    warnings: list[str]
    disclaimer: str


class ValueAnalysisService:
    """Service for calculating value analysis with risk controls."""

    # Risk parameters
    MAX_KELLY_FRACTION = 0.25  # Never bet more than 25% of Kelly
    MAX_STAKE_PCT = 0.05  # Never more than 5% of bankroll
    MIN_EV = 0.03  # Minimum 3% EV to consider
    MIN_EDGE = 0.02  # Minimum 2% edge

    def analyze_match(
        self,
        match_id: str,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        model_home: float,
        model_draw: float,
        model_away: float,
        confidence_score: int = 50,
    ) -> MatchValueReport:
        """Analyze value for all outcomes of a match.

        Args:
            match_id: Match identifier
            odds_home/draw/away: Market odds
            model_home/draw/away: Model probabilities
            confidence_score: Model confidence (0-100)

        Returns:
            MatchValueReport with analysis
        """
        warnings = []

        # Confidence adjustment
        conf_factor = confidence_score / 100.0

        outcomes_data = [
            ("home_win", "主胜", odds_home, model_home),
            ("draw", "平局", odds_draw, model_draw),
            ("away_win", "客胜", odds_away, model_away),
        ]

        analyses = []
        for outcome, label, odds, model_prob in outcomes_data:
            if odds <= 1.0:
                continue

            market_prob = 1.0 / odds
            prob_diff = model_prob - market_prob
            ev = (model_prob * odds) - 1

            # Kelly criterion
            b = odds - 1
            if b > 0:
                full_kelly = max(0, (b * model_prob - (1 - model_prob)) / b)
            else:
                full_kelly = 0

            # Apply confidence adjustment
            adjusted_kelly = full_kelly * conf_factor * self.MAX_KELLY_FRACTION
            recommended_stake = min(adjusted_kelly, self.MAX_STAKE_PCT)

            # Is value?
            is_value = ev >= self.MIN_EV and prob_diff >= self.MIN_EDGE

            # Risk warning
            if ev < 0:
                risk_warning = "负期望值，不建议介入"
            elif ev < self.MIN_EV:
                risk_warning = "期望值过低，边际不足"
            elif confidence_score < 50:
                risk_warning = "置信度不足，风险较高"
            elif full_kelly > 0.5:
                risk_warning = "Kelly仓位过高，需控制"
            else:
                risk_warning = ""

            analyses.append(ValueAnalysis(
                outcome=outcome,
                label=label,
                odds=odds,
                market_prob=market_prob,
                model_prob=model_prob,
                probability_diff=prob_diff,
                ev=ev,
                kelly_fraction=full_kelly,
                recommended_kelly=adjusted_kelly,
                recommended_stake_pct=recommended_stake,
                is_value=is_value,
                risk_warning=risk_warning,
            ))

        # Find best value
        value_outcomes = [a for a in analyses if a.is_value]
        best = max(value_outcomes, key=lambda a: a.ev) if value_outcomes else None

        # Overall risk
        if confidence_score < 40:
            overall_risk = "高"
            warnings.append("置信度过低，不建议介入")
        elif confidence_score < 60:
            overall_risk = "中"
        else:
            overall_risk = "低"

        if not value_outcomes:
            warnings.append("未发现正期望值机会")

        # Disclaimer
        disclaimer = (
            "模型识别为潜在价值，但存在风险。"
            "足球比赛存在极大不确定性，任何模型都无法保证盈利。"
            "请理性决策，严格控制风险。"
        )

        return MatchValueReport(
            match_id=match_id,
            outcomes=analyses,
            best_value=best,
            has_value=bool(value_outcomes),
            overall_risk=overall_risk,
            warnings=warnings,
            disclaimer=disclaimer,
        )

    def analyze_parlay(
        self,
        selections: list[dict],
        max_parlay_size: int = 4,
    ) -> dict:
        """Analyze parlay/combination bet.

        Args:
            selections: List of dicts with odds, model_prob, etc.
            max_parlay_size: Maximum recommended parlay size

        Returns:
            Dict with analysis results
        """
        warnings = []

        if len(selections) > max_parlay_size:
            warnings.append(f"组合超过{max_parlay_size}场，风险过高")

        # Combined odds
        combined_odds = 1.0
        combined_prob = 1.0
        for sel in selections:
            combined_odds *= sel.get("odds", 1.0)
            combined_prob *= sel.get("model_prob", 0.33)

        # EV
        ev = (combined_prob * combined_odds) - 1

        # Kelly
        b = combined_odds - 1
        if b > 0:
            kelly = max(0, (b * combined_prob - (1 - combined_prob)) / b)
        else:
            kelly = 0

        recommended_kelly = kelly * 0.25 * 0.5  # Very conservative for parlays

        # Risk assessment
        if len(selections) >= 4:
            risk = "高"
            warnings.append("4场以上过关风险极高，不建议重仓")
        elif len(selections) >= 3:
            risk = "中"
            warnings.append("3场过关风险较高，控制仓位")
        else:
            risk = "低"

        # Same league risk
        leagues = [s.get("league", "") for s in selections]
        if len(set(leagues)) == 1 and len(leagues) > 1:
            warnings.append("同一联赛比赛相关性高，风险叠加")

        # Bankrupt probability
        lose_prob = 1 - combined_prob
        streak_5 = lose_prob ** 5

        return {
            "combined_odds": combined_odds,
            "combined_prob": combined_prob,
            "ev": ev,
            "kelly": kelly,
            "recommended_kelly": recommended_kelly,
            "risk_level": risk,
            "warnings": warnings,
            "lose_probability": lose_prob,
            "streak_5_probability": streak_5,
            "disclaimer": "过关分析仅供参考。组合赔率越高，命中率越低。请理性投注，严格控制风险。",
        }


def format_value_analysis(report: MatchValueReport) -> str:
    """Format value analysis for display."""
    parts = [
        "价值分析报告",
        f"最佳机会: {report.best_value.label if report.best_value else '无'}",
        f"整体风险: {report.overall_risk}",
        "",
    ]

    for a in report.outcomes:
        parts.append(f"{a.label}: 赔率{a.odds:.2f} | EV{a.ev:+.1%} | Kelly{a.kelly_fraction:.1%}")
        if a.risk_warning:
            parts.append(f"  ⚠️ {a.risk_warning}")

    if report.warnings:
        parts.append("")
        parts.append("风险提示:")
        for w in report.warnings:
            parts.append(f"  - {w}")

    return "\n".join(parts)
