"""Lottery data validators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Validation result."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class DataQualityReport:
    """Data quality report for lottery data."""

    lottery_type: str
    total_issues: int
    missing_issues: list[str]
    duplicate_issues: list[str]
    invalid_draws: list[str]
    quality_score: float  # 0-100
    quality_level: str  # 高/中/低/不可用


def validate_p3_draw(digit_1: int, digit_2: int, digit_3: int, issue_no: str, draw_date: str) -> ValidationResult:
    """Validate a P3 draw."""
    errors = []
    warnings = []

    # Number validation
    for i, d in enumerate([digit_1, digit_2, digit_3], 1):
        if not (0 <= d <= 9):
            errors.append(f"第{i}位数字{d}超出范围0-9")

    # Issue number
    if not issue_no:
        errors.append("期号不能为空")

    # Draw date
    if not draw_date:
        errors.append("开奖日期不能为空")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_dlt_draw(
    front: list[int], back: list[int], issue_no: str, draw_date: str
) -> ValidationResult:
    """Validate a DLT draw."""
    errors = []
    warnings = []

    # Front area validation
    if len(front) != 5:
        errors.append(f"前区应有5个号码，实际{len(front)}个")
    else:
        for i, n in enumerate(front, 1):
            if not (1 <= n <= 35):
                errors.append(f"前区第{i}个号码{n}超出范围1-35")
        if len(set(front)) != 5:
            errors.append("前区号码存在重复")

    # Back area validation
    if len(back) != 2:
        errors.append(f"后区应有2个号码，实际{len(back)}个")
    else:
        for i, n in enumerate(back, 1):
            if not (1 <= n <= 12):
                errors.append(f"后区第{i}个号码{n}超出范围1-12")
        if len(set(back)) != 2:
            errors.append("后区号码存在重复")

    # Issue number
    if not issue_no:
        errors.append("期号不能为空")

    # Draw date
    if not draw_date:
        errors.append("开奖日期不能为空")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def detect_missing_issues(database: LocalDatabase, lottery_type: str) -> list[str]:
    """Detect missing issue numbers."""
    table = "p3_draws" if lottery_type == "p3" else "dlt_draws"
    try:
        with database.connection(read_only=True) as conn:
            issues = conn.execute(
                f"SELECT issue_no FROM {table} ORDER BY issue_no"
            ).fetchall()
    except Exception:
        return []

    if len(issues) < 2:
        return []

    issue_list = [str(i[0]) for i in issues]
    missing = []

    # Check for sequential gaps
    for i in range(len(issue_list) - 1):
        curr = issue_list[i]
        next_issue = issue_list[i + 1]
        try:
            curr_num = int(curr)
            next_num = int(next_issue)
            if next_num - curr_num > 1:
                for gap in range(curr_num + 1, next_num):
                    missing.append(str(gap))
        except ValueError:
            continue

    return missing[:50]  # Limit output


def detect_duplicate_issues(database: LocalDatabase, lottery_type: str) -> list[str]:
    """Detect duplicate issue numbers."""
    table = "p3_draws" if lottery_type == "p3" else "dlt_draws"
    try:
        with database.connection(read_only=True) as conn:
            dupes = conn.execute(
                f"SELECT issue_no, COUNT(*) as cnt FROM {table} GROUP BY issue_no HAVING cnt > 1"
            ).fetchall()
        return [str(d[0]) for d in dupes]
    except Exception:
        return []


def calculate_lottery_data_quality(database: LocalDatabase, lottery_type: str) -> DataQualityReport:
    """Calculate data quality score for lottery data."""
    table = "p3_draws" if lottery_type == "p3" else "dlt_draws"
    label = "排列三" if lottery_type == "p3" else "大乐透"

    try:
        with database.connection(read_only=True) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return DataQualityReport(
            lottery_type=label, total_issues=0, missing_issues=["表不存在"],
            duplicate_issues=[], invalid_draws=[], quality_score=0, quality_level="不可用"
        )

    if total == 0:
        return DataQualityReport(
            lottery_type=label, total_issues=0, missing_issues=["无数据"],
            duplicate_issues=[], invalid_draws=[], quality_score=0, quality_level="不可用"
        )

    # Check missing issues
    missing = detect_missing_issues(database, lottery_type)

    # Check duplicates
    dupes = detect_duplicate_issues(database, lottery_type)

    # Calculate score
    score = 100.0
    if missing:
        score -= min(len(missing) * 2, 30)
    if dupes:
        score -= len(dupes) * 10
    if total < 30:
        score -= 20
    elif total < 100:
        score -= 10

    score = max(0, min(100, score))

    if score >= 90:
        level = "高"
    elif score >= 70:
        level = "中"
    elif score >= 40:
        level = "低"
    else:
        level = "不可用"

    return DataQualityReport(
        lottery_type=label,
        total_issues=total,
        missing_issues=[f"缺失{len(missing)}期"] if missing else [],
        duplicate_issues=[f"重复{len(dupes)}期"] if dupes else [],
        invalid_draws=[],
        quality_score=score,
        quality_level=level,
    )
