"""Random baseline models for lottery analysis.

All model results must be compared against random baseline
to prevent false confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class P3BaselineResult:
    """P3 random baseline result."""

    combinations: list[list[int]]
    hit_count: int
    hit_rate: float
    coverage: float


@dataclass
class DLTBaselineResult:
    """DLT random baseline result."""

    combinations: list[dict]
    front_hit_distribution: dict[int, int]
    back_hit_distribution: dict[int, int]
    coverage: float


class RandomBaselineModel:
    """Random baseline model for lottery comparison.

    Purpose: All model results must compare against this baseline
    to verify the model provides value beyond random chance.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_p3(self, count: int = 10) -> list[list[int]]:
        """Generate random P3 combinations (000-999)."""
        combos = []
        for _ in range(count):
            d1 = self.rng.integers(0, 10)
            d2 = self.rng.integers(0, 10)
            d3 = self.rng.integers(0, 10)
            combos.append([int(d1), int(d2), int(d3)])
        return combos

    def generate_dlt(self, count: int = 5) -> list[dict]:
        """Generate random DLT combinations."""
        combos = []
        for _ in range(count):
            front = sorted(self.rng.choice(range(1, 36), 5, replace=False).tolist())
            back = sorted(self.rng.choice(range(1, 13), 2, replace=False).tolist())
            combos.append({"front": front, "back": back})
        return combos

    def evaluate_p3(
        self,
        actual_draws: list[list[int]],
        combos_per_draw: int = 10,
    ) -> P3BaselineResult:
        """Evaluate random baseline on P3 historical data."""
        total_hits = 0
        total_combos = 0

        for actual in actual_draws:
            random_combos = self.generate_p3(combos_per_draw)
            for combo in random_combos:
                if combo == actual:
                    total_hits += 1
            total_combos += combos_per_draw

        return P3BaselineResult(
            combinations=self.generate_p3(combos_per_draw),
            hit_count=total_hits,
            hit_rate=total_hits / total_combos if total_combos > 0 else 0,
            coverage=combos_per_draw / 1000,
        )

    def evaluate_dlt(
        self,
        actual_draws: list[dict],
        combos_per_draw: int = 5,
    ) -> DLTBaselineResult:
        """Evaluate random baseline on DLT historical data."""
        front_hits = {i: 0 for i in range(6)}
        back_hits = {i: 0 for i in range(3)}

        for actual in actual_draws:
            random_combos = self.generate_dlt(combos_per_draw)
            for combo in random_combos:
                front_match = len(set(combo["front"]) & set(actual["front"]))
                back_match = len(set(combo["back"]) & set(actual["back"]))
                front_hits[front_match] += 1
                back_hits[back_match] += 1

        return DLTBaselineResult(
            combinations=self.generate_dlt(combos_per_draw),
            front_hit_distribution=front_hits,
            back_hit_distribution=back_hits,
            coverage=combos_per_draw / 1000,
        )


def get_random_baseline_comparison(
    model_hit_rate: float,
    random_hit_rate: float,
) -> dict:
    """Compare model performance against random baseline."""
    improvement = model_hit_rate - random_hit_rate
    relative_improvement = improvement / random_hit_rate if random_hit_rate > 0 else 0

    if improvement > 0.05:
        verdict = "模型明显优于随机基线"
    elif improvement > 0.02:
        verdict = "模型略优于随机基线"
    elif improvement > -0.02:
        verdict = "模型与随机基线无显著差异"
    else:
        verdict = "模型表现不如随机基线"

    return {
        "model_hit_rate": model_hit_rate,
        "random_hit_rate": random_hit_rate,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "verdict": verdict,
        "warning": "该策略在当前回测区间内没有明显优于随机基线，不建议作为强信号使用。" if improvement <= 0.02 else "",
    }
