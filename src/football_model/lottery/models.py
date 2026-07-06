"""Lottery data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class P3Draw:
    """排列三开奖数据."""

    issue_no: str
    draw_date: str
    digit_1: int  # 百位
    digit_2: int  # 十位
    digit_3: int  # 个位

    @property
    def number_text(self) -> str:
        return f"{self.digit_1}{self.digit_2}{self.digit_3}"

    @property
    def sum_value(self) -> int:
        return self.digit_1 + self.digit_2 + self.digit_3

    @property
    def span_value(self) -> int:
        return max(self.digit_1, self.digit_2, self.digit_3) - min(self.digit_1, self.digit_2, self.digit_3)

    @property
    def odd_count(self) -> int:
        return sum(1 for d in [self.digit_1, self.digit_2, self.digit_3] if d % 2 == 1)

    @property
    def even_count(self) -> int:
        return 3 - self.odd_count

    @property
    def big_count(self) -> int:
        return sum(1 for d in [self.digit_1, self.digit_2, self.digit_3] if d >= 5)

    @property
    def small_count(self) -> int:
        return 3 - self.big_count

    @property
    def pattern_type(self) -> str:
        digits = sorted([self.digit_1, self.digit_2, self.digit_3])
        if digits[0] == digits[1] == digits[2]:
            return "豹子"
        elif digits[0] == digits[1] or digits[1] == digits[2]:
            return "组三"
        else:
            return "组六"

    @property
    def road_012(self) -> str:
        return "".join(str(d % 3) for d in [self.digit_1, self.digit_2, self.digit_3])


@dataclass
class DLTDraw:
    """超级大乐透开奖数据."""

    issue_no: str
    draw_date: str
    front_1: int
    front_2: int
    front_3: int
    front_4: int
    front_5: int
    back_1: int
    back_2: int

    @property
    def front_numbers(self) -> list[int]:
        return sorted([self.front_1, self.front_2, self.front_3, self.front_4, self.front_5])

    @property
    def back_numbers(self) -> list[int]:
        return sorted([self.back_1, self.back_2])

    @property
    def front_sum(self) -> int:
        return sum(self.front_numbers)

    @property
    def back_sum(self) -> int:
        return sum(self.back_numbers)

    @property
    def front_span(self) -> int:
        nums = self.front_numbers
        return nums[-1] - nums[0]

    @property
    def back_span(self) -> int:
        return abs(self.back_1 - self.back_2)

    @property
    def front_odd_count(self) -> int:
        return sum(1 for n in self.front_numbers if n % 2 == 1)

    @property
    def front_even_count(self) -> int:
        return 5 - self.front_odd_count

    @property
    def zone_counts(self) -> tuple[int, int, int]:
        """分区: 01-12, 13-24, 25-35"""
        z1 = sum(1 for n in self.front_numbers if 1 <= n <= 12)
        z2 = sum(1 for n in self.front_numbers if 13 <= n <= 24)
        z3 = sum(1 for n in self.front_numbers if 25 <= n <= 35)
        return z1, z2, z3
