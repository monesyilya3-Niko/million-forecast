"""Confidence scoring service.

Calculates match prediction confidence based on data quality,
model agreement, and risk factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from football_model.services.data_quality import DataQualityReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceReport:
    """Confidence assessment for a match prediction."""

    score: int  # 0-100
    level: str  # 高 / 中 / 低
    reason: str

    # Risk factors
    data_risk: str
    odds_volatility_risk: str
    lineup_uncertainty_risk: str
    upset_risk: str
    draw_risk: str
    schedule_risk: str
    market_heat_risk: str
    model_disagreement_risk: str

    # Overall risk level
    risk_level: str  # 高 / 中 / 低
    risk_factors: list[str]


class ConfidenceService:
    """Service for calculating prediction confidence."""

    def calculate(
        self,
        data_quality: DataQualityReport,
        model_probabilities: dict[str, float] | None = None,
        market_probabilities: dict[str, float] | None = None,
        home_form: str = "",
        away_form: str = "",
        days_rest_home: int = 7,
        days_rest_away: int = 7,
    ) -> ConfidenceReport:
        """Calculate confidence score.

        Args:
            data_quality: Data quality report
            model_probabilities: Model-predicted probabilities
            market_probabilities: Market-implied probabilities
            home_form: Home team recent form (e.g., "W-W-D-L-W")
            away_form: Away team recent form
            days_rest_home: Days since last match for home
            days_rest_away: Days since last match for away

        Returns:
            ConfidenceReport with score and risk assessment
        """
        score = 20.0  # Base score
        risk_factors = []

        # Data quality component (0-30)
        score += data_quality.overall_score * 30
        if data_quality.overall_score < 0.5:
            risk_factors.append("数据质量不足")

        # Model agreement (0-20)
        if model_probabilities and market_probabilities:
            model_arr = np.array([model_probabilities.get("home_win", 0.33),
                                  model_probabilities.get("draw", 0.33),
                                  model_probabilities.get("away_win", 0.33)])
            market_arr = np.array([market_probabilities.get("home_win", 0.33),
                                   market_probabilities.get("draw", 0.33),
                                   market_probabilities.get("away_win", 0.33)])
            disagreement = float(np.abs(model_arr - market_arr).max())
            if disagreement < 0.05:
                score += 20
            elif disagreement < 0.10:
                score += 15
            elif disagreement < 0.15:
                score += 10
            else:
                score += 5
                risk_factors.append(f"模型与市场分歧{disagreement:.1%}")
        else:
            score += 10

        # Form consistency (0-15)
        form_score = self._assess_form(home_form, away_form)
        score += form_score
        if form_score < 8:
            risk_factors.append("近期状态不稳定")

        # Schedule pressure (0-15)
        schedule_score = self._assess_schedule(days_rest_home, days_rest_away)
        score += schedule_score
        if schedule_score < 8:
            risk_factors.append("赛程压力大")

        # Final score
        final_score = int(np.clip(round(score), 0, 100))

        # Level
        if final_score >= 75:
            level = "高"
        elif final_score >= 55:
            level = "中"
        else:
            level = "低"

        # Risk assessments
        data_risk = "低" if data_quality.overall_score >= 0.7 else "中" if data_quality.overall_score >= 0.4 else "高"
        odds_risk = "低" if data_quality.has_odds else "高"
        lineup_risk = "低" if data_quality.has_lineup else "中"
        upset_risk = self._assess_upset_risk(model_probabilities, market_probabilities)
        draw_risk = self._assess_draw_risk(model_probabilities)
        schedule_risk = "低" if schedule_score >= 12 else "中" if schedule_score >= 8 else "高"
        market_heat = self._assess_market_heat(market_probabilities)
        model_disagreement = self._assess_model_disagreement(model_probabilities, market_probabilities)

        # Overall risk level
        risk_scores = [
            1 if data_risk == "高" else 0,
            1 if odds_risk == "高" else 0,
            1 if lineup_risk == "高" else 0,
            1 if upset_risk == "高" else 0,
            1 if schedule_risk == "高" else 0,
        ]
        risk_count = sum(risk_scores)
        if risk_count >= 3:
            risk_level = "高"
        elif risk_count >= 1:
            risk_level = "中"
        else:
            risk_level = "低"

        # Reason
        if final_score >= 75:
            reason = "数据完整，模型与市场一致，适合分析参考"
        elif final_score >= 55:
            reason = "数据基本可用，存在一定不确定性"
        else:
            reason = "数据不足或分歧较大，建议谨慎参考"

        return ConfidenceReport(
            score=final_score,
            level=level,
            reason=reason,
            data_risk=data_risk,
            odds_volatility_risk=odds_risk,
            lineup_uncertainty_risk=lineup_risk,
            upset_risk=upset_risk,
            draw_risk=draw_risk,
            schedule_risk=schedule_risk,
            market_heat_risk=market_heat,
            model_disagreement_risk=model_disagreement,
            risk_level=risk_level,
            risk_factors=risk_factors,
        )

    def _assess_form(self, home_form: str, away_form: str) -> float:
        """Assess form consistency (0-15)."""
        if not home_form or not away_form:
            return 7.5

        def form_score(form: str) -> float:
            results = form.split("-")
            if not results:
                return 0.5
            wins = results.count("W")
            draws = results.count("D")
            return (wins * 3 + draws) / (len(results) * 3)

        home_s = form_score(home_form)
        away_s = form_score(away_form)

        # Consistency bonus
        consistency = 1.0 - abs(home_s - away_s)

        return (home_s + away_s + consistency) * 5

    def _assess_schedule(self, rest_home: int, rest_away: int) -> float:
        """Assess schedule pressure (0-15)."""
        score = 15.0

        if rest_home < 3:
            score -= 5
        elif rest_home < 5:
            score -= 2

        if rest_away < 3:
            score -= 5
        elif rest_away < 5:
            score -= 2

        return max(0, score)

    def _assess_upset_risk(
        self,
        model: dict[str, float] | None,
        market: dict[str, float] | None,
    ) -> str:
        """Assess upset risk."""
        if not model:
            return "中"

        # Check if favorite is clear
        max_prob = max(model.values())
        if max_prob > 0.60:
            return "低"
        elif max_prob > 0.45:
            return "中"
        return "高"

    def _assess_draw_risk(self, model: dict[str, float] | None) -> str:
        """Assess draw risk."""
        if not model:
            return "中"

        draw_prob = model.get("draw", 0.33)
        if draw_prob > 0.30:
            return "高"
        elif draw_prob > 0.25:
            return "中"
        return "低"

    def _assess_market_heat(self, market: dict[str, float] | None) -> str:
        """Assess market heat (overround)."""
        if not market:
            return "中"

        total = sum(market.values())
        overround = total - 1.0
        if overround > 0.10:
            return "高"
        elif overround > 0.05:
            return "中"
        return "低"

    def _assess_model_disagreement(
        self,
        model: dict[str, float] | None,
        market: dict[str, float] | None,
    ) -> str:
        """Assess model-market disagreement."""
        if not model or not market:
            return "中"

        model_arr = np.array(list(model.values()))
        market_arr = np.array(list(market.values()))
        max_diff = float(np.abs(model_arr - market_arr).max())

        if max_diff > 0.15:
            return "高"
        elif max_diff > 0.08:
            return "中"
        return "低"
