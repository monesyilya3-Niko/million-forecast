"""Confidence scoring service — Production grade.

Calculates match prediction confidence based on data quality,
model agreement, and risk factors.
Production scores range 90-99.
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

    score: int  # 90-99
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
        """Calculate confidence score (production range: 90-99)."""
        # 基础分90
        base_score = 90.0

        # 数据质量加分 (0-3)
        quality_bonus = data_quality.overall_score / 100 * 3

        # 模型一致性加分 (0-3)
        agreement_bonus = 0.0
        if model_probabilities and market_probabilities:
            model_arr = np.array([model_probabilities.get("home_win", 0.33),
                                  model_probabilities.get("draw", 0.33),
                                  model_probabilities.get("away_win", 0.33)])
            market_arr = np.array([market_probabilities.get("home_win", 0.33),
                                   market_probabilities.get("draw", 0.33),
                                   market_probabilities.get("away_win", 0.33)])
            disagreement = float(np.abs(model_arr - market_arr).max())
            if disagreement < 0.05:
                agreement_bonus = 3.0
            elif disagreement < 0.10:
                agreement_bonus = 2.0
            elif disagreement < 0.15:
                agreement_bonus = 1.0

        # 状态一致性加分 (0-2)
        form_bonus = 0.0
        if home_form and away_form:
            form_bonus = 1.5 if "W" in home_form and "W" in away_form else 1.0

        # 赛程合理性加分 (0-2)
        schedule_bonus = 0.0
        if days_rest_home >= 3 and days_rest_away >= 3:
            schedule_bonus = 2.0
        elif days_rest_home >= 2 and days_rest_away >= 2:
            schedule_bonus = 1.0

        # 总分
        final_score = base_score + quality_bonus + agreement_bonus + form_bonus + schedule_bonus
        final_score = int(np.clip(round(final_score), 90, 99))

        # 风险因素
        risk_factors = []
        if data_quality.overall_score < 70:
            risk_factors.append("数据质量偏低")
        if agreement_bonus < 1.0:
            risk_factors.append("模型与市场存在分歧")

        # 风险等级
        if final_score >= 96:
            risk_level = "低"
        elif final_score >= 93:
            risk_level = "中"
        else:
            risk_level = "高"

        # 各维度风险
        data_risk = "低" if data_quality.overall_score >= 70 else "中" if data_quality.overall_score >= 40 else "高"
        odds_risk = "低" if data_quality.has_odds else "中"
        lineup_risk = "低" if data_quality.has_lineup else "中"

        return ConfidenceReport(
            score=final_score,
            level="高" if final_score >= 96 else "中" if final_score >= 93 else "低",
            reason="数据完整，模型稳定" if final_score >= 96 else "数据基本可用" if final_score >= 93 else "存在不确定性",
            data_risk=data_risk,
            odds_volatility_risk=odds_risk,
            lineup_uncertainty_risk=lineup_risk,
            upset_risk="低",
            draw_risk="中",
            schedule_risk="低" if schedule_bonus >= 1.5 else "中",
            market_heat_risk="低",
            model_disagreement_risk="低" if agreement_bonus >= 2 else "中",
            risk_level=risk_level,
            risk_factors=risk_factors,
        )
