"""彩票回测页面 — 完整版."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository
from football_model.lottery.baseline import RandomBaselineModel, get_random_baseline_comparison
from football_model.ui.components import hero_pro, section_header, empty_state, render_risk_note, plotly_theme

logger = logging.getLogger(__name__)


def render_lottery_backtest(database: LocalDatabase) -> None:
    hero_pro("彩票回测", "模型回测验证与随机基线对比。", "LOTTERY BACKTEST", ["命中率", "ROI", "随机基线"])

    repo = LotteryRepository(database)
    lottery_type = st.selectbox("选择彩种", ["排列三", "大乐透"], key="bt-lottery-type")

    if lottery_type == "排列三":
        _render_p3_backtest(repo)
    else:
        _render_dlt_backtest(repo)

    render_risk_note(
        "回测结果基于历史数据，不代表未来表现。彩票为随机开奖，"
        "历史走势不能改变未来概率。请理性参与，严格控制预算。"
    )


def _render_p3_backtest(repo: LotteryRepository) -> None:
    """排列三回测."""
    from football_model.lottery.services import P3AnalysisService

    service = P3AnalysisService(repo)
    draws = repo.get_p3_draws(limit=500)

    if draws.empty:
        empty_state("暂无排列三数据", "请先在数据中心导入历史开奖数据。", "🎲")
        return

    section_header("回测设置", "选择回测区间和策略。")
    c1, c2, c3 = st.columns(3)
    with c1:
        window = st.selectbox("回测期数", [30, 50, 100, 200], index=1, key="p3-bt-window")
    with c2:
        strategy = st.selectbox("策略", ["随机基线", "频率策略", "遗漏策略", "形态策略"], key="p3-bt-strategy")
    with c3:
        combos_per_draw = st.number_input("每期注数", 1, 50, 10, key="p3-bt-combos")

    test_draws = draws.head(window)
    if len(test_draws) < 10:
        st.warning("数据不足，请导入更多历史数据。")
        return

    # 执行回测
    results = _run_p3_backtest(test_draws, service, strategy, combos_per_draw)

    # 随机基线对比
    baseline = RandomBaselineModel(seed=42)
    actual_draws = [[int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])] for _, r in test_draws.iterrows()]
    baseline_result = baseline.evaluate_p3(actual_draws, combos_per_draw)

    comparison = get_random_baseline_comparison(results["hit_rate"], baseline_result.hit_rate)

    # 显示结果
    section_header("回测结果", f"最近{window}期 · {strategy}。")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("回测期数", str(results["total"]))
    m2.metric("命中次数", str(results["hits"]))
    m3.metric("命中率", f"{results['hit_rate']:.2%}")
    m4.metric("随机基线", f"{baseline_result.hit_rate:.2%}")

    # 对比分析
    section_header("随机基线对比", "模型 vs 随机策略。")
    if comparison["improvement"] > 0.02:
        st.success(f"✅ {comparison['verdict']} (提升 {comparison['relative_improvement']:.1%})")
    elif comparison["improvement"] > -0.02:
        st.warning(f"⚠️ {comparison['verdict']}")
    else:
        st.error(f"❌ {comparison['verdict']}")

    # 成本分析
    section_header("成本分析", "假设每注2元。")
    total_cost = results["total"] * combos_per_draw * 2
    st.metric("总成本", f"¥{total_cost:,.0f}")
    st.caption("缺少奖级奖金数据，仅统计命中，不计算真实收益。")

    # 命中率走势
    section_header("命中率走势", "累计命中率变化。")
    if results["cumulative"]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results["cumulative"]))),
            y=results["cumulative"],
            mode="lines",
            name="累计命中率",
            line=dict(color="#3b82f6"),
        ))
        fig.add_hline(y=baseline_result.hit_rate, line_dash="dash", line_color="#94a3b8", annotation_text="随机基线")
        fig.update_layout(height=250, yaxis_tickformat=".2%")
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def _run_p3_backtest(draws: pd.DataFrame, service, strategy: str, combos_per_draw: int) -> dict:
    """执行排列三回测."""
    total = len(draws) - 1
    hits = 0
    cumulative = []
    rng = np.random.default_rng(42)

    for i in range(total - 1, -1, -1):
        current = draws.iloc[i]
        actual = [int(current["digit_1"]), int(current["digit_2"]), int(current["digit_3"])]

        # 生成预测组合
        if strategy == "随机基线":
            combos = [rng.integers(0, 10, 3).tolist() for _ in range(combos_per_draw)]
        elif strategy == "频率策略":
            freq = service.get_frequency(draws.head(i + 1), 50)
            weights = [freq.get(str(d), 0.1) for d in range(10)]
            weights = np.array(weights) / sum(weights)
            combos = [rng.choice(10, 3, p=weights).tolist() for _ in range(combos_per_draw)]
        elif strategy == "遗漏策略":
            missing = service.get_missing_values(draws.head(i + 1))
            # Higher missing = higher weight
            weights = [max(1, missing.get(f"1_{d}", 0) + missing.get(f"2_{d}", 0) + missing.get(f"3_{d}", 0)) for d in range(10)]
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            combos = [rng.choice(10, 3, p=weights).tolist() for _ in range(combos_per_draw)]
        else:  # 形态策略
            combos = [rng.integers(0, 10, 3).tolist() for _ in range(combos_per_draw)]

        # 检查命中
        hit = any(list(c) == actual for c in combos)
        if hit:
            hits += 1

        rate = hits / (total - i) if (total - i) > 0 else 0
        cumulative.append(rate)

    return {
        "total": total,
        "hits": hits,
        "hit_rate": hits / total if total > 0 else 0,
        "cumulative": cumulative,
    }


def _render_dlt_backtest(repo: LotteryRepository) -> None:
    """大乐透回测."""
    from football_model.lottery.services import DLTAnalysisService

    DLTAnalysisService(repo)  # Initialize service
    draws = repo.get_dlt_draws(limit=500)

    if draws.empty:
        empty_state("暂无大乐透数据", "请先在数据中心导入历史开奖数据。", "🎱")
        return

    section_header("回测设置", "选择回测区间。")
    c1, c2 = st.columns(2)
    with c1:
        window = st.selectbox("回测期数", [30, 50, 100, 200], index=1, key="dlt-bt-window")
    with c2:
        combos_per_draw = st.number_input("每期注数", 1, 20, 5, key="dlt-bt-combos")

    test_draws = draws.head(window)
    if len(test_draws) < 10:
        st.warning("数据不足，请导入更多历史数据。")
        return

    # 随机基线
    baseline = RandomBaselineModel(seed=42)
    actual_draws = []
    for _, row in test_draws.iterrows():
        actual_draws.append({
            "front": sorted([int(row[f"front_{i}"]) for i in range(1, 6)]),
            "back": sorted([int(row[f"back_{i}"]) for i in range(1, 3)]),
        })

    baseline_result = baseline.evaluate_dlt(actual_draws, combos_per_draw)

    # 显示结果
    section_header("前区命中分布", f"最近{window}期 · 每期{combos_per_draw}注。")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("回测期数", str(len(actual_draws)))
    m2.metric("前区命中≥2", str(baseline_result.front_hit_distribution.get(2, 0) + baseline_result.front_hit_distribution.get(3, 0) + baseline_result.front_hit_distribution.get(4, 0) + baseline_result.front_hit_distribution.get(5, 0)))
    m3.metric("前区命中≥3", str(baseline_result.front_hit_distribution.get(3, 0) + baseline_result.front_hit_distribution.get(4, 0) + baseline_result.front_hit_distribution.get(5, 0)))
    m4.metric("后区命中≥1", str(baseline_result.back_hit_distribution.get(1, 0) + baseline_result.back_hit_distribution.get(2, 0)))

    # 分布图
    section_header("前区命中分布", "每期命中个数。")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(k) for k in sorted(baseline_result.front_hit_distribution.keys())],
        y=[baseline_result.front_hit_distribution[k] for k in sorted(baseline_result.front_hit_distribution.keys())],
        marker_color="#3b82f6",
    ))
    fig.update_layout(height=250, xaxis_title="命中个数", yaxis_title="次数")
    fig = _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # 成本分析
    section_header("成本分析", "假设每注2元。")
    total_cost = len(actual_draws) * combos_per_draw * 2
    st.metric("总成本", f"¥{total_cost:,.0f}")
    st.caption("缺少奖级奖金数据，仅统计命中，不计算真实收益。")



_apply_chart_theme = plotly_theme
