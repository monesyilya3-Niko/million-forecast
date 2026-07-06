"""超级大乐透历史数据页面."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository
from football_model.ui.components import hero_pro, section_header, empty_state

logger = logging.getLogger(__name__)


def render_dlt_history(database: LocalDatabase) -> None:
    hero_pro("大乐透历史数据", "查询、筛选和导出历史开奖数据。", "DLT HISTORY", ["查询", "筛选", "导出"])

    repo = LotteryRepository(database)
    draws = repo.get_dlt_draws(limit=999)

    if draws.empty:
        empty_state("暂无大乐透数据", "请先在数据中心导入历史开奖数据。", "🎱")
        return

    # 数据统计
    section_header("数据概览", f"共 {len(draws)} 期。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总期数", str(len(draws)))
    c2.metric("最早期号", str(draws.iloc[-1]["issue_no"]))
    c3.metric("最近期号", str(draws.iloc[0]["issue_no"]))
    c4.metric("数据质量", _get_quality_label(len(draws)))

    # 筛选条件
    section_header("筛选条件", "按条件过滤历史数据。")
    with st.expander("筛选选项", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sum_min = st.number_input("前区和值最小", 15, 150, 15, key="dlth-sum-min")
            sum_max = st.number_input("前区和值最大", 15, 150, 150, key="dlth-sum-max")
        with f2:
            include_front = st.text_input("包含前区号码(如5,12)", "", key="dlth-include-front")
            include_back = st.text_input("包含后区号码(如3,9)", "", key="dlth-include-back")
        with f3:
            st.selectbox("前区奇偶比", ["不限", "0:5", "1:4", "2:3", "3:2", "4:1", "5:0"], key="dlth-odd")

    # 应用筛选
    filtered = draws.copy()
    filtered["前区和值"] = filtered.apply(lambda r: sum(int(r[f"front_{i}"]) for i in range(1, 6)), axis=1)
    filtered = filtered[(filtered["前区和值"] >= sum_min) & (filtered["前区和值"] <= sum_max)]

    if include_front:
        try:
            nums = [int(n.strip()) for n in include_front.split(",") if n.strip().isdigit()]
            if nums:
                mask = filtered.apply(lambda r: any(int(r[f"front_{i}"]) in nums for i in range(1, 6)), axis=1)
                filtered = filtered[mask]
        except ValueError:
            pass

    if include_back:
        try:
            nums = [int(n.strip()) for n in include_back.split(",") if n.strip().isdigit()]
            if nums:
                mask = filtered.apply(lambda r: any(int(r[f"back_{i}"]) in nums for i in range(1, 3)), axis=1)
                filtered = filtered[mask]
        except ValueError:
            pass

    # 构建显示
    section_header("筛选结果", f"共 {len(filtered)} 期。")
    rows = []
    for _, row in filtered.iterrows():
        front = "-".join(f"{int(row[f'front_{i}']):02d}" for i in range(1, 6))
        back = "-".join(f"{int(row[f'back_{i}']):02d}" for i in range(1, 3))
        rows.append({
            "期号": row["issue_no"],
            "日期": row["draw_date"],
            "前区": front,
            "后区": back,
            "前区和值": int(row["前区和值"]),
        })

    display = pd.DataFrame(rows)
    st.dataframe(display, hide_index=True, use_container_width=True)

    # 导出
    section_header("数据导出", "导出筛选后的数据。")
    st.download_button(
        "📥 导出CSV",
        display.to_csv(index=False).encode("utf-8-sig"),
        "dlt_history.csv",
        "text/csv",
        key="dlth-export",
    )


def _get_quality_label(count: int) -> str:
    if count >= 200:
        return "高"
    elif count >= 50:
        return "中"
    elif count >= 20:
        return "低"
    return "不足"
