"""Lottery risk control module."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 禁止文案
BANNED_WORDS = [
    "必中", "稳赚", "包中", "保赢", "稳赢", "预测中奖", "下一期必出",
    "杀号必中", "绝杀", "包赔", "无风险", "保证中奖", "100%中奖",
]

# 替换文案
SAFE_REPLACEMENTS = {
    "必中": "候选",
    "稳赚": "参考",
    "包中": "筛选",
    "保赢": "分析",
    "稳赢": "参考",
    "预测中奖": "数据分析",
    "下一期必出": "历史统计倾向",
    "杀号必中": "组合筛选",
    "绝杀": "过滤",
    "包赔": "风险提示",
    "无风险": "有风险",
    "保证中奖": "仅供参考",
    "100%中奖": "概率分析",
}

# 统一免责声明
DISCLAIMER = (
    "本系统仅用于历史开奖数据分析、概率研究、组合筛选和风险评估，不构成任何中奖承诺。"
    "彩票开奖结果具有随机性，历史走势不能决定未来结果。请理性参与，严格控制预算。"
)

LOTTERY_RISK_NOTE = (
    "⚠️ 彩票风险提示：\n"
    "1. 彩票开奖为随机事件，历史走势不能改变未来概率\n"
    "2. 任何模型都不能保证中奖\n"
    "3. 请严格控制预算，量力而行\n"
    "4. 本分析仅供参考，不构成投注建议"
)


@dataclass
class BudgetControl:
    """Budget control for lottery combinations."""

    budget_amount: float  # 总预算
    stake_per_ticket: float  # 单注金额
    max_combinations: int  # 最大注数

    @property
    def total_cost(self) -> float:
        """Total cost if all combinations are played."""
        return self.stake_per_ticket * self.max_combinations

    @property
    def max_loss(self) -> float:
        """Maximum possible loss."""
        return min(self.total_cost, self.budget_amount)

    @property
    def budget_usage_percent(self) -> float:
        """Budget usage percentage."""
        if self.budget_amount <= 0:
            return 0
        return min(100, self.total_cost / self.budget_amount * 100)

    @property
    def is_over_budget(self) -> bool:
        """Check if total cost exceeds budget."""
        return self.total_cost > self.budget_amount

    def risk_level(self) -> str:
        """Determine risk level."""
        if self.is_over_budget:
            return "不建议"
        if self.budget_usage_percent > 80:
            return "高风险"
        if self.budget_usage_percent > 50:
            return "中风险"
        return "低风险"


def sanitize_explanation(text: str) -> str:
    """Sanitize explanation text, replacing banned words."""
    result = text
    for banned, safe in SAFE_REPLACEMENTS.items():
        result = result.replace(banned, safe)
    return result


def check_text_safety(text: str) -> list[str]:
    """Check text for banned words, return list of found violations."""
    violations = []
    for banned in BANNED_WORDS:
        if banned in text:
            violations.append(banned)
    return violations


def calculate_p3_budget(
    combinations: int,
    stake: float = 2.0,
    budget: float = 100.0,
) -> BudgetControl:
    """Calculate budget control for P3 combinations."""
    return BudgetControl(
        budget_amount=budget,
        stake_per_ticket=stake,
        max_combinations=combinations,
    )


def calculate_dlt_budget(
    combinations: int,
    stake: float = 2.0,
    budget: float = 100.0,
) -> BudgetControl:
    """Calculate budget control for DLT combinations."""
    return BudgetControl(
        budget_amount=budget,
        stake_per_ticket=stake,
        max_combinations=combinations,
    )
