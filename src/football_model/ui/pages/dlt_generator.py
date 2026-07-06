"""超级大乐透组合生成页面."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository, DLTAnalysisService
from football_model.lottery.risk import (
    calculate_dlt_budget, DISCLAIMER, LOTTERY_RISK_NOTE,
)
from football_model.ui.components import hero_pro, section_header, empty_state, render_risk_note, lottery_number_ball

logger = logging.getLogger(__name__)


def render_dlt_generator(database: LocalDatabase) -> None:
    hero_pro("大乐透组合生成", "前区1-35选5、后区1-12选2。", "DLT GENERATOR", ["胆拖", "过滤", "预算控制"])

    repo = LotteryRepository(database)
    service = DLTAnalysisService(repo)
    draws = repo.get_dlt_draws(limit=500)

    if draws.empty:
        empty_state("暂无大乐透数据", "请先在数据中心导入历史开奖数据。", "🎱")
        return

    # 前区设置
    section_header("前区设置", "设置前区号码池和过滤条件。")
    with st.expander("前区号码池", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            front_pool = st.text_input("前区号码池(如1,5,12,18,25,33)", "", key="dltg-front-pool")
            front_dan = st.text_input("前区胆码(必选)", "", key="dltg-front-dan")
        with f2:
            front_exclude = st.text_input("前区排除号码", "", key="dltg-front-exclude")
            front_sum_min, front_sum_max = st.slider(
                "前区和值范围", 15, 150, (50, 120), key="dltg-front-sum"
            )

    # 后区设置
    section_header("后区设置", "设置后区号码池。")
    with st.expander("后区号码池", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            back_pool = st.text_input("后区号码池(如1,3,7,9,12)", "", key="dltg-back-pool")
            back_dan = st.text_input("后区胆码(必选)", "", key="dltg-back-dan")
        with f2:
            back_exclude = st.text_input("后区排除号码", "", key="dltg-back-exclude")

    # 预算控制
    section_header("预算控制", "设置预算和注数限制。")
    b1, b2, b3 = st.columns(3)
    with b1:
        max_combos = st.number_input("最大注数", 1, 50, 5, key="dltg-max-combos")
    with b2:
        stake = st.number_input("单注金额(元)", 1.0, 100.0, 2.0, key="dltg-stake")
    with b3:
        budget = st.number_input("总预算(元)", 10.0, 10000.0, 100.0, key="dltg-budget")

    budget_ctrl = calculate_dlt_budget(max_combos, stake, budget)

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("总成本", f"¥{budget_ctrl.total_cost:.0f}")
    bc2.metric("最大亏损", f"¥{budget_ctrl.max_loss:.0f}")
    bc3.metric("预算占用", f"{budget_ctrl.budget_usage_percent:.0f}%")
    bc4.metric("风险等级", budget_ctrl.risk_level())

    if budget_ctrl.is_over_budget:
        st.warning("⚠️ 总成本超过预算，请减少注数或降低单注金额。")

    # 生成组合
    section_header("候选组合", "点击生成参考组合。")
    if st.button("🎱 生成候选组合", key="dltg-generate", type="primary"):
        # 解析号码池
        fp = _parse_number_list(front_pool, 1, 35)
        fd = _parse_number_list(front_dan, 1, 35)
        fe = _parse_number_list(front_exclude, 1, 35)
        bp = _parse_number_list(back_pool, 1, 12)
        bd = _parse_number_list(back_dan, 1, 12)
        be = _parse_number_list(back_exclude, 1, 12)

        # 生成组合
        combos = _generate_dlt_combinations(
            draws, service, max_combos,
            front_pool=fp, front_dan=fd, front_exclude=fe,
            back_pool=bp, back_dan=bd, back_exclude=be,
            front_sum_range=(front_sum_min, front_sum_max),
        )

        if combos:
            section_header("生成结果", f"共 {len(combos)} 组候选组合。")

            for i, combo in enumerate(combos):
                front_balls = "".join(lottery_number_ball(n, "dlt-front", "small") for n in combo["front"])
                back_balls = "".join(lottery_number_ball(n, "dlt-back", "small") for n in combo["back"])
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem;margin:0.3rem 0">'
                    f'<span style="color:var(--text-muted);font-size:0.8rem;width:2rem">#{i+1}</span>'
                    f'{front_balls}'
                    f'<span style="color:var(--text-muted);margin:0 0.2rem">|</span>'
                    f'{back_balls}'
                    f'<span style="color:var(--text-muted);font-size:0.75rem;margin-left:0.5rem">'
                    f'和值{combo["front_sum"]} 奇偶{combo["odd_even"]}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

            # 导出
            export_rows = []
            for combo in combos:
                export_rows.append({
                    "前区": "-".join(f"{n:02d}" for n in combo["front"]),
                    "后区": "-".join(f"{n:02d}" for n in combo["back"]),
                    "前区和值": combo["front_sum"],
                    "奇偶比": combo["odd_even"],
                })
            st.download_button(
                "📥 导出候选组合",
                pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig"),
                "dlt_combinations.csv",
                "text/csv",
                key="dltg-export",
            )
        else:
            st.warning("未找到满足条件的组合，请放宽过滤条件。")

    render_risk_note(DISCLAIMER)
    st.caption(LOTTERY_RISK_NOTE)


def _parse_number_list(text: str, min_val: int, max_val: int) -> list[int]:
    """Parse comma-separated number list."""
    if not text or not text.strip():
        return []
    try:
        nums = [int(n.strip()) for n in text.split(",") if n.strip().isdigit()]
        return [n for n in nums if min_val <= n <= max_val]
    except ValueError:
        return []


def _generate_dlt_combinations(
    draws: pd.DataFrame,
    service: DLTAnalysisService,
    count: int,
    front_pool: list[int],
    front_dan: list[int],
    front_exclude: list[int],
    back_pool: list[int],
    back_dan: list[int],
    back_exclude: list[int],
    front_sum_range: tuple[int, int],
) -> list[dict]:
    """Generate DLT combinations with filters."""
    rng = np.random.default_rng()
    combinations = []
    attempts = 0
    max_attempts = count * 500

    # Default pools
    all_front = list(range(1, 36))
    all_back = list(range(1, 13))

    front_available = [n for n in (front_pool if front_pool else all_front) if n not in front_exclude]
    back_available = [n for n in (back_pool if back_pool else all_back) if n not in back_exclude]

    while len(combinations) < count and attempts < max_attempts:
        attempts += 1

        # Generate front
        if front_dan:
            remaining = [n for n in front_available if n not in front_dan]
            if len(remaining) < 5 - len(front_dan):
                continue
            extra = rng.choice(remaining, 5 - len(front_dan), replace=False).tolist()
            front = sorted(front_dan + extra)
        else:
            if len(front_available) < 5:
                continue
            front = sorted(rng.choice(front_available, 5, replace=False).tolist())

        # Generate back
        if back_dan:
            remaining = [n for n in back_available if n not in back_dan]
            if len(remaining) < 2 - len(back_dan):
                continue
            extra = rng.choice(remaining, 2 - len(back_dan), replace=False).tolist()
            back = sorted(back_dan + extra)
        else:
            if len(back_available) < 2:
                continue
            back = sorted(rng.choice(back_available, 2, replace=False).tolist())

        # Apply filters
        front_sum = sum(front)
        if not (front_sum_range[0] <= front_sum <= front_sum_range[1]):
            continue

        odd = sum(1 for n in front if n % 2 == 1)
        combinations.append({
            "front": front,
            "back": back,
            "front_sum": front_sum,
            "odd_even": f"{odd}奇{5 - odd}偶",
        })

    return combinations
