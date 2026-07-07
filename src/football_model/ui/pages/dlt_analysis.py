"""超级大乐透分析页面 — 完整版."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository, DLTAnalysisService
from football_model.ui.components import (
    hero_pro, section_header, empty_state, render_risk_note,
    render_dlt_numbers, lottery_number_ball, plotly_theme,
)

logger = logging.getLogger(__name__)


def render_dlt_analysis(database: LocalDatabase) -> None:
    hero_pro("超级大乐透分析", "前区1-35选5 · 后区1-12选2 · 概率分析", "DLT ANALYSIS", ["前区", "后区", "分区"])

    repo = LotteryRepository(database)
    service = DLTAnalysisService(repo)
    draws = repo.get_dlt_draws(limit=500)

    if draws.empty:
        empty_state("暂无大乐透数据", "请先在数据中心导入历史开奖数据。", "🎱")
        _render_import_section(repo)
        return

    # ── 顶部统计 ──
    latest = draws.iloc[0]
    front = sorted([int(latest[f"front_{i}"]) for i in range(1, 6)])
    back = sorted([int(latest[f"back_{i}"]) for i in range(1, 3)])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总期数", str(len(draws)))
    c2.metric("最近期号", str(latest["issue_no"]))
    c3.metric("开奖日期", str(latest["draw_date"]))
    c4.metric("前区和值", str(sum(front)))
    c5.metric("后区和值", str(sum(back)))

    section_header("最近开奖", "前区蓝球 + 后区红球。")
    render_dlt_numbers(front, back)

    # ── Tab结构 ──
    front_tab, back_tab, zone_tab, combo_tab, import_tab = st.tabs([
        "前区分析", "后区分析", "分区奇偶", "组合生成", "数据导入"
    ])

    # ═══ 前区分析 ═══
    with front_tab:
        _render_front_section(draws, service)

    # ═══ 后区分析 ═══
    with back_tab:
        _render_back_section(draws, service)

    # ═══ 分区奇偶 ═══
    with zone_tab:
        _render_zone_section(draws, service)

    # ═══ 组合生成 ═══
    with combo_tab:
        _render_combo_section(draws, service)

    # ═══ 数据导入 ═══
    with import_tab:
        _render_import_section(repo)

    render_risk_note(
        "超级大乐透为随机开奖，前区35选5、后区12选2，中奖概率极低。"
        "历史走势不能改变未来开奖概率。以上分析仅供参考，不构成投注建议。"
        "请理性参与，严格控制预算。"
    )


def _render_front_section(draws: pd.DataFrame, service: DLTAnalysisService) -> None:
    """前区分析."""
    section_header("前区号码频率", "01-35出现频率（最近100期）。")
    freq = service.get_front_frequency(draws, window=100)
    if freq:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{k:02d}" for k in freq.keys()],
            y=list(freq.values()),
            marker_color="#3b82f6",
            text=[f"{v:.1%}" for v in freq.values()],
            textposition="outside",
            textfont=dict(size=8),
        ))
        fig.add_hline(y=5/35, line_dash="dash", line_color="#94a3b8", annotation_text=f"理论均值{5/35:.1%}")
        fig.update_layout(height=320, yaxis_tickformat=".1%")
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # 热冷号
    section_header("前区热冷号", "最近50期。")
    freq_50 = service.get_front_frequency(draws, window=50)
    if freq_50:
        sorted_freq = sorted(freq_50.items(), key=lambda x: x[1], reverse=True)
        hot = [k for k, v in sorted_freq[:8]]
        cold = [k for k, v in sorted_freq[-8:]]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 热号**")
            html = " ".join(lottery_number_ball(n, "dlt-front", "small") for n in hot)
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.markdown("**❄️ 冷号**")
            html = " ".join(lottery_number_ball(n, "dlt-front", "small") for n in cold)
            st.markdown(html, unsafe_allow_html=True)

    # 和值走势
    section_header("前区和值走势", "最近50期。")
    sum_data = draws.head(50).copy()
    sum_data["和值"] = sum_data.apply(lambda r: sum(int(r[f"front_{i}"]) for i in range(1, 6)), axis=1)
    sum_data = sum_data.iloc[::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(sum_data))),
        y=sum_data["和值"].tolist(),
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5),
    ))
    mean_val = sum_data["和值"].mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#94a3b8", annotation_text=f"均值{mean_val:.0f}")
    fig.update_layout(height=280)
    fig = _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_back_section(draws: pd.DataFrame, service: DLTAnalysisService) -> None:
    """后区分析."""
    section_header("后区号码频率", "01-12出现频率（最近100期）。")
    freq = service.get_back_frequency(draws, window=100)
    if freq:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{k:02d}" for k in freq.keys()],
            y=list(freq.values()),
            marker_color="#ef4444",
            text=[f"{v:.1%}" for v in freq.values()],
            textposition="outside",
        ))
        fig.add_hline(y=2/12, line_dash="dash", line_color="#94a3b8", annotation_text=f"理论均值{2/12:.1%}")
        fig.update_layout(height=280, yaxis_tickformat=".1%")
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # 后区热冷号
    section_header("后区热冷号", "最近50期。")
    freq_50 = service.get_back_frequency(draws, window=50)
    if freq_50:
        sorted_freq = sorted(freq_50.items(), key=lambda x: x[1], reverse=True)
        hot = [k for k, v in sorted_freq[:4]]
        cold = [k for k, v in sorted_freq[-4:]]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 热号**")
            html = " ".join(lottery_number_ball(n, "dlt-back", "small") for n in hot)
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.markdown("**❄️ 冷号**")
            html = " ".join(lottery_number_ball(n, "dlt-back", "small") for n in cold)
            st.markdown(html, unsafe_allow_html=True)


def _render_zone_section(draws: pd.DataFrame, service: DLTAnalysisService) -> None:
    """分区奇偶分析."""
    section_header("前区分区比", "01-12/13-24/25-35分布。")
    zone_dist = service.get_zone_distribution(draws)
    if zone_dist:
        top_zones = dict(list(zone_dist.items())[:12])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(top_zones.keys()),
            y=list(top_zones.values()),
            marker_color="#a855f7",
            text=[f"{v:.1%}" for v in top_zones.values()],
            textposition="outside",
        ))
        fig.update_layout(height=280, yaxis_tickformat=".1%")
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    section_header("前区奇偶比", "奇偶比例分布。")
    odd_even = service.get_odd_even_distribution(draws)
    if odd_even:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=list(odd_even.keys()),
                values=list(odd_even.values()),
                hole=0.45,
                textinfo="label+percent",
            ))
            fig.update_layout(height=280)
            fig = _apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            for k, v in sorted(odd_even.items(), key=lambda x: x[1], reverse=True)[:6]:
                st.caption(f"{k}: {v:.1%}")

    section_header("前区大小比", "大(19-35)/小(01-18)比例。")
    draws_copy = draws.copy()
    draws_copy["大小"] = draws_copy.apply(
        lambda r: f"{sum(1 for i in range(1,6) if int(r[f'front_{i}'])>=19)}大{sum(1 for i in range(1,6) if int(r[f'front_{i}'])<19)}小",
        axis=1,
    )
    big_small = draws_copy["大小"].value_counts(normalize=True).to_dict()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=list(big_small.keys()),
            values=list(big_small.values()),
            hole=0.45,
            textinfo="label+percent",
        ))
        fig.update_layout(height=250)
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        for k, v in sorted(big_small.items(), key=lambda x: x[1], reverse=True)[:6]:
            st.caption(f"{k}: {v:.1%}")


def _render_combo_section(draws: pd.DataFrame, service: DLTAnalysisService) -> None:
    """组合生成."""
    section_header("参考组合生成", "基于统计过滤的候选组合。")

    with st.expander("过滤条件", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            sum_min = st.number_input("前区和值最小", 15, 150, 50, key="dlt-sum-min")
            sum_max = st.number_input("前区和值最大", 15, 150, 120, key="dlt-sum-max")
        with f2:
            combo_count = st.number_input("生成数量", 1, 20, 5, key="dlt-combo-count")

    if st.button("🎱 生成参考组合", key="dlt-generate", type="primary"):
        combos = service.generate_reference_combinations(
            draws,
            count=combo_count,
            front_sum_range=(sum_min, sum_max),
        )
        if combos:
            # 渲染号码球
            for i, combo in enumerate(combos):
                front_nums = [int(n) for n in combo["前区"].split("-")]
                back_nums = [int(n) for n in combo["后区"].split("-")]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem;margin:0.3rem 0">'
                    f'<span style="color:var(--text-muted);font-size:0.8rem;width:2rem">#{i+1}</span>'
                    f'{"".join(lottery_number_ball(n, "dlt-front", "small") for n in front_nums)}'
                    f'<span style="color:var(--text-muted);margin:0 0.2rem">|</span>'
                    f'{"".join(lottery_number_ball(n, "dlt-back", "small") for n in back_nums)}'
                    f'<span style="color:var(--text-muted);font-size:0.75rem;margin-left:0.5rem">和值{combo["前区和值"]} 奇偶{combo["奇偶比"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"共生成 {len(combos)} 组参考组合")
        else:
            st.warning("未找到满足条件的组合，请放宽过滤条件。")


def _render_import_section(repo: LotteryRepository) -> None:
    """数据导入."""
    section_header("导入大乐透数据", "支持CSV格式导入。")

    template = pd.DataFrame([{
        "issue_no": "24001", "draw_date": "2024-01-01",
        "front_1": 5, "front_2": 12, "front_3": 18, "front_4": 25, "front_5": 33,
        "back_1": 3, "back_2": 9,
    }])
    st.download_button(
        "📥 下载CSV模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        "dlt_template.csv",
        "text/csv",
        key="dlt-template-dl",
    )

    upload = st.file_uploader("上传CSV文件", type=["csv"], key="dlt-upload")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)
            if st.button("确认导入", key="dlt-import-confirm", type="primary"):
                result = repo.import_dlt_from_csv(upload)
                c1, c2, c3 = st.columns(3)
                c1.metric("成功", result.success)
                c2.metric("更新", result.updated)
                c3.metric("错误", result.error_count)
                if result.error_count > 0:
                    for err in result.errors[:5]:
                        st.warning(err)
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")



_apply_chart_theme = plotly_theme
