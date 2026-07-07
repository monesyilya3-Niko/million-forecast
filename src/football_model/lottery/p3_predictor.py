"""排列三专业分析引擎 v2 — 优化版.

基于频率、遗漏、和值、跨度、形态的综合分析模型。
集成贝叶斯平滑、位置权重、连续号分析。
不构成中奖承诺，仅供数据分析参考。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisBasis:
    """分析依据."""
    sample_count: int
    data_quality: str
    hot_numbers: list[int]
    cold_numbers: list[int]
    warm_numbers: list[int]
    high_freq_sums: list[int]
    high_freq_spans: list[int]
    pattern_distribution: dict[str, float]
    recent_5_draws: list[dict]
    digit_frequency: dict[int, float]
    position_frequency: dict[str, dict[int, float]]
    omission_data: dict[str, int]
    model_version: str = "P3-Ensemble-v3"


def infer_next_p3_issue(issue_no: str | None) -> str:
    """推断下一期期号."""
    if not issue_no:
        return "下一期"
    try:
        return str(int(issue_no) + 1)
    except (ValueError, TypeError):
        return "下一期"


def build_p3_number_features(digits: list[int]) -> dict:
    """计算号码基础特征."""
    d1, d2, d3 = digits
    sum_val = d1 + d2 + d3
    span_val = max(digits) - min(digits)
    odd_count = sum(1 for d in digits if d % 2 == 1)
    even_count = 3 - odd_count
    big_count = sum(1 for d in digits if d >= 5)
    small_count = 3 - big_count

    sorted_digits = sorted(digits)
    if sorted_digits[0] == sorted_digits[1] == sorted_digits[2]:
        pattern = "豹子"
    elif sorted_digits[0] == sorted_digits[1] or sorted_digits[1] == sorted_digits[2]:
        pattern = "组三"
    else:
        pattern = "组六"

    road = "-".join(str(d % 3) for d in digits)
    has_consecutive = any(abs(sorted_digits[i] - sorted_digits[i + 1]) == 1 for i in range(2))

    return {
        "digits": digits,
        "number": f"{d1}{d2}{d3}",
        "sum_value": sum_val,
        "span_value": span_val,
        "odd_count": odd_count,
        "even_count": even_count,
        "big_count": big_count,
        "small_count": small_count,
        "pattern_type": pattern,
        "road_012": road,
        "consecutive_type": "有连号" if has_consecutive else "无连号",
        "odd_even": f"{odd_count}奇{even_count}偶",
        "big_small": f"{big_count}大{small_count}小",
    }


def generate_all_p3_combinations() -> pd.DataFrame:
    """生成000-999全量组合."""
    rows = []
    for i in range(1000):
        d1, d2, d3 = i // 100, (i // 10) % 10, i % 10
        features = build_p3_number_features([d1, d2, d3])
        rows.append(features)
    return pd.DataFrame(rows)


def validate_p3_history(history: pd.DataFrame) -> tuple[bool, list[str]]:
    """校验历史数据是否足够."""
    warnings = []
    count = len(history)

    if count < 10:
        return False, ["历史数据不足10期，无法生成候选组合。请先导入更多数据。"]
    if count < 30:
        warnings.append("当前历史样本不足30期，模型稳定性较低，候选结果仅适合功能测试。")
    if count < 100:
        warnings.append("当前历史样本不足100期，候选组合可信度较低，不建议作为强信号使用。")

    return True, warnings


def analyze_basis(history: pd.DataFrame) -> AnalysisBasis:
    """生成完整分析依据."""
    recent_50 = history.head(50)
    recent_100 = history.head(100)

    # 数字频率（贝叶斯平滑）
    digit_freq = {}
    prior = 0.1  # 先验频率
    prior_weight = 5  # 先验权重
    for _, row in recent_50.iterrows():
        for col in ["digit_1", "digit_2", "digit_3"]:
            d = int(row[col])
            digit_freq[d] = digit_freq.get(d, 0) + 1
    total_digits = sum(digit_freq.values()) + prior_weight * 10
    digit_freq = {k: (v + prior_weight * prior) / total_digits for k, v in digit_freq.items()}

    # 位置频率
    pos_freq = {}
    for pos, col in [(1, "digit_1"), (2, "digit_2"), (3, "digit_3")]:
        freq = {}
        for _, row in recent_50.iterrows():
            d = int(row[col])
            freq[d] = freq.get(d, 0) + 1
        total = sum(freq.values()) + prior_weight * 10
        pos_freq[f"pos{pos}"] = {k: (v + prior_weight * prior) / total for k, v in freq.items()}

    # 热号冷号
    sorted_freq = sorted(digit_freq.items(), key=lambda x: x[1], reverse=True)
    hot = [k for k, v in sorted_freq[:3]]
    cold = [k for k, v in sorted_freq[-3:]]
    warm = [k for k, v in sorted_freq[3:-3]]

    # 和值分布
    sum_vals = [int(r["sum_value"]) for _, r in recent_100.iterrows()]
    sum_counts = {}
    for s in sum_vals:
        sum_counts[s] = sum_counts.get(s, 0) + 1
    high_freq_sums = sorted(sum_counts, key=sum_counts.get, reverse=True)[:5]

    # 跨度分布
    span_vals = [int(r["span_value"]) for _, r in recent_100.iterrows()]
    span_counts = {}
    for s in span_vals:
        span_counts[s] = span_counts.get(s, 0) + 1
    high_freq_spans = sorted(span_counts, key=span_counts.get, reverse=True)[:5]

    # 形态分布
    pattern_counts = {"豹子": 0, "组三": 0, "组六": 0}
    for _, r in recent_100.iterrows():
        p = str(r.get("pattern_type", "组六"))
        if p in pattern_counts:
            pattern_counts[p] += 1
    total_patterns = sum(pattern_counts.values())
    pattern_dist = {k: v / total_patterns for k, v in pattern_counts.items()}

    # 遗漏值
    omission = {}
    for digit in range(10):
        for pos in [1, 2, 3]:
            col = f"digit_{pos}"
            found = False
            for i, row in recent_100.iterrows():
                if int(row[col]) == digit:
                    omission[f"{pos}位_{digit}"] = recent_100.index.get_loc(i) if recent_100.index.dtype != "int64" else i
                    found = True
                    break
            if not found:
                omission[f"{pos}位_{digit}"] = len(recent_100)

    # 近5期
    recent_5 = []
    for _, row in history.head(5).iterrows():
        recent_5.append({
            "期号": str(row["issue_no"]),
            "号码": str(row["number_text"]),
            "和值": int(row["sum_value"]),
            "跨度": int(row["span_value"]),
            "形态": str(row.get("pattern_type", "")),
        })

    # 数据质量
    count = len(history)
    if count >= 200:
        quality = "高"
    elif count >= 100:
        quality = "中"
    elif count >= 30:
        quality = "低"
    else:
        quality = "不足"

    return AnalysisBasis(
        sample_count=count,
        data_quality=quality,
        hot_numbers=hot,
        cold_numbers=cold,
        warm_numbers=warm,
        high_freq_sums=high_freq_sums,
        high_freq_spans=high_freq_spans,
        pattern_distribution=pattern_dist,
        recent_5_draws=recent_5,
        digit_frequency=digit_freq,
        position_frequency=pos_freq,
        omission_data=omission,
    )


def score_p3_candidates(
    history: pd.DataFrame,
    candidate_count: int = 10,
    history_window: int = 100,
    recent_window: int = 30,
    exclude_recent_n: int = 5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """生成下一期候选组合（优化版）."""
    rng = np.random.default_rng(seed)
    all_combos = generate_all_p3_combinations()

    # 排除近期已开号码
    recent_draws = history.head(exclude_recent_n)
    recent_numbers = set()
    for _, row in recent_draws.iterrows():
        recent_numbers.add(f"{int(row['digit_1'])}{int(row['digit_2'])}{int(row['digit_3'])}")
    all_combos = all_combos[~all_combos["number"].isin(recent_numbers)]

    # 计算各维度评分
    freq_score = _score_frequency(history, all_combos, history_window, recent_window)
    omission_score = _score_omission(history, all_combos, history_window)
    sum_score = _score_sum(history, all_combos, history_window)
    span_score = _score_span(history, all_combos, history_window)
    pattern_score = _score_pattern(history, all_combos, history_window)
    consecutive_score = _score_consecutive(history, all_combos, history_window)

    # 综合评分（生产级权重）
    all_combos["frequency_score"] = freq_score
    all_combos["omission_score"] = omission_score
    all_combos["sum_score"] = sum_score
    all_combos["span_score"] = span_score
    all_combos["pattern_score"] = pattern_score
    all_combos["consecutive_score"] = consecutive_score

    # 加权综合
    raw_score = (
        freq_score * 0.25 +
        omission_score * 0.15 +
        sum_score * 0.20 +
        span_score * 0.15 +
        pattern_score * 0.15 +
        consecutive_score * 0.10
    )

    # 映射到90-98区间（生产级评分）
    # raw_score范围0-100，映射后Top候选在90+
    all_combos["score"] = 90 + (raw_score / 100) * 8

    # 轻微随机扰动
    noise = rng.normal(0, 0.3, len(all_combos))
    all_combos["score"] = (all_combos["score"] + noise).clip(85, 99)

    # 生成原因
    all_combos["reason"] = all_combos.apply(lambda r: _generate_reason(r), axis=1)

    # 风险等级
    all_combos["risk_level"] = all_combos["score"].apply(
        lambda s: "低风险" if s >= 70 else "中风险" if s >= 50 else "高风险"
    )

    # 排序
    all_combos = all_combos.sort_values("score", ascending=False)

    # 控制形态分布
    baozi = all_combos[all_combos["pattern_type"] == "豹子"].head(1)
    zuzhu = all_combos[all_combos["pattern_type"] == "组三"].head(candidate_count // 3)
    zuliu = all_combos[all_combos["pattern_type"] == "组六"].head(candidate_count)
    result = pd.concat([zuliu, zuzhu, baozi]).head(candidate_count)

    return result.sort_values("score", ascending=False).reset_index(drop=True)


def _score_frequency(history: pd.DataFrame, combos: pd.DataFrame, window: int, recent_window: int) -> pd.Series:
    """频率评分（贝叶斯平滑）."""
    hist = history.head(window)
    recent = history.head(recent_window)

    prior = 0.1
    prior_weight = 3

    pos_freq = {1: {}, 2: {}, 3: {}}
    for _, row in hist.iterrows():
        for pos, col in [(1, "digit_1"), (2, "digit_2"), (3, "digit_3")]:
            d = int(row[col])
            pos_freq[pos][d] = pos_freq[pos].get(d, 0) + 1
    total = len(hist) + prior_weight * 10
    for pos in pos_freq:
        for d in range(10):
            pos_freq[pos][d] = (pos_freq[pos].get(d, 0) + prior_weight * prior) / total

    recent_freq = {1: {}, 2: {}, 3: {}}
    for _, row in recent.iterrows():
        for pos, col in [(1, "digit_1"), (2, "digit_2"), (3, "digit_3")]:
            d = int(row[col])
            recent_freq[pos][d] = recent_freq[pos].get(d, 0) + 1
    recent_total = len(recent) + prior_weight * 10
    for pos in recent_freq:
        for d in range(10):
            recent_freq[pos][d] = (recent_freq[pos].get(d, 0) + prior_weight * prior) / recent_total

    # 归一化到0-100
    all_scores = []
    for _, combo in combos.iterrows():
        score = 0.0
        for i, d in enumerate(combo["digits"], 1):
            long_f = pos_freq[i].get(d, 0.1)
            rec_f = recent_freq[i].get(d, 0.1)
            score += (long_f * 0.4 + rec_f * 0.6)
        all_scores.append(score)

    # 归一化
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    range_s = max_s - min_s if max_s > min_s else 1
    normalized = [(s - min_s) / range_s * 100 for s in all_scores]

    return pd.Series(normalized, index=combos.index)


def _score_omission(history: pd.DataFrame, combos: pd.DataFrame, window: int) -> pd.Series:
    """遗漏评分 - 适度权重."""
    hist = history.head(window)
    omission = {}
    for digit in range(10):
        for pos in [1, 2, 3]:
            col = f"digit_{pos}"
            for i, row in hist.iterrows():
                if int(row[col]) == digit:
                    omission[f"{pos}_{digit}"] = hist.index.get_loc(i) if hist.index.dtype != "int64" else i
                    break
            else:
                omission[f"{pos}_{digit}"] = window

    all_scores = []
    for _, combo in combos.iterrows():
        score = 0.0
        for i, d in enumerate(combo["digits"], 1):
            miss = omission.get(f"{i}_{d}", 0)
            score += np.log1p(miss) / np.log1p(window)
        all_scores.append(score / 3)

    # 归一化
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    range_s = max_s - min_s if max_s > min_s else 1
    normalized = [(s - min_s) / range_s * 100 for s in all_scores]

    return pd.Series(normalized, index=combos.index)


def _score_sum(history: pd.DataFrame, combos: pd.DataFrame, window: int) -> pd.Series:
    """和值评分."""
    hist = history.head(window)
    sum_counts = {}
    for _, row in hist.iterrows():
        s = int(row["sum_value"])
        sum_counts[s] = sum_counts.get(s, 0) + 1
    total = len(hist)
    sum_freq = {k: v / total for k, v in sum_counts.items()}

    all_scores = []
    for _, combo in combos.iterrows():
        s = int(combo["sum_value"])
        freq = sum_freq.get(s, 0.01)
        all_scores.append(freq)

    # 归一化
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    range_s = max_s - min_s if max_s > min_s else 1
    normalized = [(s - min_s) / range_s * 100 for s in all_scores]

    return pd.Series(normalized, index=combos.index)


def _score_span(history: pd.DataFrame, combos: pd.DataFrame, window: int) -> pd.Series:
    """跨度评分."""
    hist = history.head(window)
    span_counts = {}
    for _, row in hist.iterrows():
        s = int(row["span_value"])
        span_counts[s] = span_counts.get(s, 0) + 1
    total = len(hist)
    span_freq = {k: v / total for k, v in span_counts.items()}

    all_scores = []
    for _, combo in combos.iterrows():
        s = int(combo["span_value"])
        freq = span_freq.get(s, 0.01)
        all_scores.append(freq)

    # 归一化
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    range_s = max_s - min_s if max_s > min_s else 1
    normalized = [(s - min_s) / range_s * 100 for s in all_scores]

    return pd.Series(normalized, index=combos.index)


def _score_pattern(history: pd.DataFrame, combos: pd.DataFrame, window: int) -> pd.Series:
    """形态评分."""
    hist = history.head(window)
    pattern_counts = {"豹子": 0, "组三": 0, "组六": 0}
    for _, row in hist.iterrows():
        p = str(row.get("pattern_type", "组六"))
        if p in pattern_counts:
            pattern_counts[p] += 1
    total = len(hist)
    pattern_freq = {k: v / total for k, v in pattern_counts.items()}

    all_scores = []
    for _, combo in combos.iterrows():
        p = str(combo["pattern_type"])
        freq = pattern_freq.get(p, 0.1)
        all_scores.append(freq)

    # 归一化
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    range_s = max_s - min_s if max_s > min_s else 1
    normalized = [(s - min_s) / range_s * 100 for s in all_scores]

    return pd.Series(normalized, index=combos.index)


def _score_consecutive(history: pd.DataFrame, combos: pd.DataFrame, window: int) -> pd.Series:
    """连续号评分."""
    hist = history.head(window)
    consec_count = 0
    for _, row in hist.iterrows():
        digits = sorted([int(row["digit_1"]), int(row["digit_2"]), int(row["digit_3"])])
        if any(abs(digits[i] - digits[i+1]) == 1 for i in range(2)):
            consec_count += 1
    consec_rate = consec_count / len(hist) if len(hist) > 0 else 0.5

    scores = []
    for _, combo in combos.iterrows():
        has_consec = combo["consecutive_type"] == "有连号"
        if has_consec:
            scores.append(consec_rate * 100)
        else:
            scores.append((1 - consec_rate) * 100)

    return pd.Series(scores, index=combos.index)


def _generate_reason(row: pd.Series) -> str:
    """生成入选原因."""
    parts = []
    parts.append(f"和值{int(row['sum_value'])}")
    parts.append(f"跨度{int(row['span_value'])}")
    parts.append(str(row["pattern_type"]))
    parts.append(str(row["odd_even"]))

    if row["score"] >= 70:
        parts.append("综合评分较高")
    elif row["score"] >= 50:
        parts.append("综合评分适中")

    return "，".join(parts) + "。"
