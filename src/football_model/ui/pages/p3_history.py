"""排列三历史数据页面."""

from __future__ import annotations

import logging

import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository
from football_model.ui.components import hero_pro, section_header, empty_state

logger = logging.getLogger(__name__)


def render_p3_history(database: LocalDatabase) -> None:
    hero_pro("排列三历史数据", "查询、筛选和导出历史开奖数据。", "P3 HISTORY", ["查询", "筛选", "导出"])

    repo = LotteryRepository(database)
    draws = repo.get_p3_draws(limit=9999)

    if draws.empty:
        empty_state("暂无排列三数据", "请先在数据中心导入历史开奖数据。", "🎲")
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
            sum_min = st.number_input("和值最小", 0, 27, 0, key="p3h-sum-min")
            sum_max = st.number_input("和值最大", 0, 27, 27, key="p3h-sum-max")
        with f2:
            span_min = st.number_input("跨度最小", 0, 9, 0, key="p3h-span-min")
            span_max = st.number_input("跨度最大", 0, 9, 9, key="p3h-span-max")
        with f3:
            pattern = st.selectbox("形态", ["全部", "豹子", "组三", "组六"], key="p3h-pattern")

    # 应用筛选
    filtered = draws.copy()
    filtered["和值"] = filtered["digit_1"].astype(int) + filtered["digit_2"].astype(int) + filtered["digit_3"].astype(int)
    filtered["跨度"] = filtered.apply(
        lambda r: max(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])) - min(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])),
        axis=1,
    )

    filtered = filtered[(filtered["和值"] >= sum_min) & (filtered["和值"] <= sum_max)]
    filtered = filtered[(filtered["跨度"] >= span_min) & (filtered["跨度"] <= span_max)]

    if pattern != "全部":
        filtered = filtered[filtered.get("pattern_type", "") == pattern]

    # 显示数据
    section_header("筛选结果", f"共 {len(filtered)} 期。")
    display = filtered[["issue_no", "draw_date", "number_text", "和值", "跨度"]].copy()
    display.columns = ["期号", "日期", "开奖号", "和值", "跨度"]

    if "pattern_type" in filtered.columns:
        display["形态"] = filtered["pattern_type"].values

    st.dataframe(display, hide_index=True, use_container_width=True)

    # 导出
    section_header("数据导出", "导出筛选后的数据。")
    st.download_button(
        "📥 导出CSV",
        display.to_csv(index=False).encode("utf-8-sig"),
        "p3_history.csv",
        "text/csv",
        key="p3h-export",
    )


def _get_quality_label(count: int) -> str:
    if count >= 500:
        return "高"
    elif count >= 100:
        return "中"
    elif count >= 30:
        return "低"
    return "不足"
