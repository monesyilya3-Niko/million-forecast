"""Lottery analysis services."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from .repositories import LotteryRepository

logger = logging.getLogger(__name__)


class P3AnalysisService:
    """排列三分析服务."""

    def __init__(self, repository: LotteryRepository) -> None:
        self.repo = repository

    def get_frequency(self, draws: pd.DataFrame, window: int = 0) -> dict:
        """计算数字频率."""
        if draws.empty:
            return {}

        if window > 0:
            draws = draws.head(window)

        freq = Counter()
        for _, row in draws.iterrows():
            for col in ["digit_1", "digit_2", "digit_3"]:
                freq[int(row[col])] += 1

        total = sum(freq.values())
        return {str(k): v / total if total > 0 else 0 for k, v in sorted(freq.items())}

    def get_position_frequency(self, draws: pd.DataFrame, position: int) -> dict:
        """计算某位置数字频率."""
        col = f"digit_{position}"
        if draws.empty or col not in draws.columns:
            return {}

        freq = Counter(draws[col].astype(int))
        total = len(draws)
        return {str(k): v / total if total > 0 else 0 for k, v in sorted(freq.items())}

    def get_sum_distribution(self, draws: pd.DataFrame) -> dict:
        """计算和值分布."""
        if draws.empty:
            return {}

        sums = draws["digit_1"].astype(int) + draws["digit_2"].astype(int) + draws["digit_3"].astype(int)
        freq = Counter(sums)
        total = len(draws)
        return {k: v / total for k, v in sorted(freq.items())}

    def get_span_distribution(self, draws: pd.DataFrame) -> dict:
        """计算跨度分布."""
        if draws.empty:
            return {}

        spans = draws.apply(
            lambda r: max(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])) - min(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])),
            axis=1,
        )
        freq = Counter(spans)
        total = len(draws)
        return {k: v / total for k, v in sorted(freq.items())}

    def get_pattern_distribution(self, draws: pd.DataFrame) -> dict:
        """计算形态分布."""
        if draws.empty:
            return {}

        patterns = []
        for _, row in draws.iterrows():
            d = sorted([int(row["digit_1"]), int(row["digit_2"]), int(row["digit_3"])])
            if d[0] == d[1] == d[2]:
                patterns.append("豹子")
            elif d[0] == d[1] or d[1] == d[2]:
                patterns.append("组三")
            else:
                patterns.append("组六")

        freq = Counter(patterns)
        total = len(draws)
        return {k: v / total for k, v in sorted(freq.items())}

    def get_missing_values(self, draws: pd.DataFrame) -> dict:
        """计算遗漏值."""
        if draws.empty:
            return {}

        missing = {}
        for digit in range(10):
            for pos in range(1, 4):
                col = f"digit_{pos}"
                found = False
                for i, row in draws.iterrows():
                    if int(row[col]) == digit:
                        missing[f"{pos}位_{digit}"] = draws.index.get_loc(i) if draws.index.dtype != "int64" else i
                        found = True
                        break
                if not found:
                    missing[f"{pos}位_{digit}"] = len(draws)

        return missing

    def get_hot_cold_numbers(self, draws: pd.DataFrame, window: int = 30) -> dict:
        """计算热号冷号."""
        freq = self.get_frequency(draws, window)
        if not freq:
            return {"hot": [], "cold": [], "warm": []}

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = [int(k) for k, v in sorted_freq[:3]]
        cold = [int(k) for k, v in sorted_freq[-3:]]
        warm = [int(k) for k, v in sorted_freq[3:-3]]

        return {"hot": hot, "cold": cold, "warm": warm}

    def generate_reference_combinations(
        self,
        draws: pd.DataFrame,
        count: int = 10,
        sum_range: tuple[int, int] | None = None,
        span_range: tuple[int, int] | None = None,
        pattern: str | None = None,
        exclude_recent: int = 0,
    ) -> list[dict]:
        """生成参考组合."""
        rng = np.random.default_rng()

        # 排除近期号码
        exclude_set = set()
        if exclude_recent > 0:
            for _, row in draws.head(exclude_recent).iterrows():
                exclude_set.add(f"{int(row['digit_1'])}{int(row['digit_2'])}{int(row['digit_3'])}")

        combinations = []
        attempts = 0
        max_attempts = count * 100

        while len(combinations) < count and attempts < max_attempts:
            attempts += 1
            d1, d2, d3 = rng.integers(0, 10, 3)

            # 排除近期
            num_text = f"{d1}{d2}{d3}"
            if num_text in exclude_set:
                continue

            # 和值过滤
            s = d1 + d2 + d3
            if sum_range and not (sum_range[0] <= s <= sum_range[1]):
                continue

            # 跨度过滤
            span = max(d1, d2, d3) - min(d1, d2, d3)
            if span_range and not (span_range[0] <= span <= span_range[1]):
                continue

            # 形态过滤
            if pattern:
                digits = sorted([d1, d2, d3])
                if pattern == "豹子" and not (digits[0] == digits[1] == digits[2]):
                    continue
                elif pattern == "组三" and not (digits[0] == digits[1] or digits[1] == digits[2]):
                    continue
                elif pattern == "组六" and not (digits[0] != digits[1] and digits[1] != digits[2]):
                    continue

            combinations.append({
                "号码": num_text,
                "和值": s,
                "跨度": span,
                "奇偶": f"{sum(1 for d in [d1,d2,d3] if d%2==1)}奇{sum(1 for d in [d1,d2,d3] if d%2==0)}偶",
                "大小": f"{sum(1 for d in [d1,d2,d3] if d>=5)}大{sum(1 for d in [d1,d2,d3] if d<5)}小",
            })

        return combinations


class DLTAnalysisService:
    """超级大乐透分析服务."""

    def __init__(self, repository: LotteryRepository) -> None:
        self.repo = repository

    def get_front_frequency(self, draws: pd.DataFrame, window: int = 0) -> dict:
        """计算前区号码频率."""
        if draws.empty:
            return {}

        if window > 0:
            draws = draws.head(window)

        freq = Counter()
        for _, row in draws.iterrows():
            for i in range(1, 6):
                freq[int(row[f"front_{i}"])] += 1

        total = sum(freq.values())
        return {k: v / total if total > 0 else 0 for k, v in sorted(freq.items())}

    def get_back_frequency(self, draws: pd.DataFrame, window: int = 0) -> dict:
        """计算后区号码频率."""
        if draws.empty:
            return {}

        if window > 0:
            draws = draws.head(window)

        freq = Counter()
        for _, row in draws.iterrows():
            for i in range(1, 3):
                freq[int(row[f"back_{i}"])] += 1

        total = sum(freq.values())
        return {k: v / total if total > 0 else 0 for k, v in sorted(freq.items())}

    def get_zone_distribution(self, draws: pd.DataFrame) -> dict:
        """计算前区分区比分布."""
        if draws.empty:
            return {}

        zone_patterns = []
        for _, row in draws.iterrows():
            nums = sorted([int(row[f"front_{i}"]) for i in range(1, 6)])
            z1 = sum(1 for n in nums if 1 <= n <= 12)
            z2 = sum(1 for n in nums if 13 <= n <= 24)
            z3 = sum(1 for n in nums if 25 <= n <= 35)
            zone_patterns.append(f"{z1}:{z2}:{z3}")

        freq = Counter(zone_patterns)
        total = len(draws)
        return {k: v / total for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True)}

    def get_odd_even_distribution(self, draws: pd.DataFrame) -> dict:
        """计算奇偶比分布."""
        if draws.empty:
            return {}

        patterns = []
        for _, row in draws.iterrows():
            nums = [int(row[f"front_{i}"]) for i in range(1, 6)]
            odd = sum(1 for n in nums if n % 2 == 1)
            patterns.append(f"{odd}奇{5 - odd}偶")

        freq = Counter(patterns)
        total = len(draws)
        return {k: v / total for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True)}

    def generate_reference_combinations(
        self,
        draws: pd.DataFrame,
        count: int = 5,
        front_sum_range: tuple[int, int] | None = None,
        odd_even: str | None = None,
    ) -> list[dict]:
        """生成参考组合."""
        rng = np.random.default_rng()
        combinations = []
        attempts = 0
        max_attempts = count * 200

        while len(combinations) < count and attempts < max_attempts:
            attempts += 1
            front = sorted(rng.choice(range(1, 36), 5, replace=False).tolist())
            back = sorted(rng.choice(range(1, 13), 2, replace=False).tolist())

            front_sum = sum(front)
            if front_sum_range and not (front_sum_range[0] <= front_sum <= front_sum_range[1]):
                continue

            odd = sum(1 for n in front if n % 2 == 1)
            if odd_even and f"{odd}奇{5 - odd}偶" != odd_even:
                continue

            combinations.append({
                "前区": "-".join(f"{n:02d}" for n in front),
                "后区": "-".join(f"{n:02d}" for n in back),
                "前区和值": front_sum,
                "后区和值": sum(back),
                "奇偶比": f"{odd}奇{5 - odd}偶",
            })

        return combinations
