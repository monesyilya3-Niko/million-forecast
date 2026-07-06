"""排列三组合生成页面."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository, P3AnalysisService
from football_model.lottery.risk import (
    calculate_p3_budget, DISCLAIMER, LOTTERY_RISK_NOTE,
)
from football_model.ui.components import hero_pro, section_header, empty_state, render_risk_note, lottery_number_ball

logger = logging.getLogger(__name__)


def render_p3_generator(database: LocalDatabase) -> None:
    hero_pro("排列三组合生成", "基于统计过滤的候选组合。", "P3 GENERATOR", ["过滤", "评分", "预算控制"])

    repo = LotteryRepository(database)
    service = P3AnalysisService(repo)
    draws = repo.get_p3_draws(limit=500)

    if draws.empty:
        empty_state("暂无排列三数据", "请先在数据中心导入历史开奖数据。", "🎲")
        return

    # 过滤条件
    section_header("过滤条件", "设置组合筛选规则。")
    with st.expander("基本过滤", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sum_min = st.number_input("和值最小", 0, 27, 5, key="p3g-sum-min")
            sum_max = st.number_input("和值最大", 0, 27, 22, key="p3g-sum-max")
        with f2:
            span_min = st.number_input("跨度最小", 0, 9, 1, key="p3g-span-min")
            span_max = st.number_input("跨度最大", 0, 9, 9, key="p3g-span-max")
        with f3:
            pattern = st.selectbox("形态", ["不限", "豹子", "组三", "组六"], key="p3g-pattern")

    with st.expander("高级过滤", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            include_digits = st.text_input("包含数字(如1,2,3)", "", key="p3g-include")
            exclude_digits = st.text_input("排除数字(如0,9)", "", key="p3g-exclude")
        with f2:
            exclude_recent = st.number_input("排除近N期已开号", 0, 100, 10, key="p3g-exclude-recent")
            odd_filter = st.selectbox("奇偶比", ["不限", "0奇3偶", "1奇2偶", "2奇1偶", "3奇0偶"], key="p3g-odd")

    # 预算控制
    section_header("预算控制", "设置预算和注数限制。")
    b1, b2, b3 = st.columns(3)
    with b1:
        max_combos = st.number_input("最大注数", 1, 100, 10, key="p3g-max-combos")
    with b2:
        stake = st.number_input("单注金额(元)", 1.0, 100.0, 2.0, key="p3g-stake")
    with b3:
        budget = st.number_input("总预算(元)", 10.0, 10000.0, 100.0, key="p3g-budget")

    budget_ctrl = calculate_p3_budget(max_combos, stake, budget)

    # 预算信息
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("总成本", f"¥{budget_ctrl.total_cost:.0f}")
    bc2.metric("最大亏损", f"¥{budget_ctrl.max_loss:.0f}")
    bc3.metric("预算占用", f"{budget_ctrl.budget_usage_percent:.0f}%")
    risk = budget_ctrl.risk_level()
    bc4.metric("风险等级", risk)

    if budget_ctrl.is_over_budget:
        st.warning("⚠️ 总成本超过预算，请减少注数或降低单注金额。")

    # 生成组合
    section_header("候选组合", "点击生成参考组合。")
    if st.button("🎲 生成候选组合", key="p3g-generate", type="primary"):
        # 解析过滤条件
        inc_digits = []
        if include_digits:
            try:
                inc_digits = [int(d.strip()) for d in include_digits.split(",") if d.strip().isdigit()]
            except ValueError:
                pass

        exc_digits = []
        if exclude_digits:
            try:
                exc_digits = [int(d.strip()) for d in exclude_digits.split(",") if d.strip().isdigit()]
            except ValueError:
                pass

        # 生成组合
        combos = service.generate_reference_combinations(
            draws,
            count=max_combos,
            sum_range=(sum_min, sum_max),
            span_range=(span_min, span_max),
            pattern=pattern if pattern != "不限" else None,
            exclude_recent=exclude_recent,
        )

        # 应用额外过滤
        filtered_combos = []
        for combo in combos:
            digits = [int(c) for c in combo["号码"]]

            # 包含数字过滤
            if inc_digits and not any(d in inc_digits for d in digits):
                continue

            # 排除数字过滤
            if exc_digits and any(d in exc_digits for d in digits):
                continue

            # 奇偶过滤
            if odd_filter != "不限":
                odd_count = sum(1 for d in digits if d % 2 == 1)
                expected = int(odd_filter[0])
                if odd_count != expected:
                    continue

            filtered_combos.append(combo)

        if filtered_combos:
            # 显示组合
            section_header("生成结果", f"共 {len(filtered_combos)} 组候选组合。")

            for i, combo in enumerate(filtered_combos):
                d1, d2, d3 = int(combo["号码"][0]), int(combo["号码"][1]), int(combo["号码"][2])
                balls = (
                    lottery_number_ball(d1, "p3-pos1", "small") +
                    lottery_number_ball(d2, "p3-pos2", "small") +
                    lottery_number_ball(d3, "p3-pos3", "small")
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem;margin:0.3rem 0">'
                    f'<span style="color:var(--text-muted);font-size:0.8rem;width:2rem">#{i+1}</span>'
                    f'{balls}'
                    f'<span style="color:var(--text-muted);font-size:0.75rem;margin-left:0.5rem">'
                    f'和值{combo["和值"]} 跨度{combo["跨度"]} {combo["奇偶"]} {combo["大小"]}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

            # 导出
            st.download_button(
                "📥 导出候选组合",
                pd.DataFrame(filtered_combos).to_csv(index=False).encode("utf-8-sig"),
                "p3_combinations.csv",
                "text/csv",
                key="p3g-export",
            )
        else:
            st.warning("未找到满足条件的组合，请放宽过滤条件。")

    # 风险提示
    render_risk_note(DISCLAIMER)
    st.caption(LOTTERY_RISK_NOTE)
