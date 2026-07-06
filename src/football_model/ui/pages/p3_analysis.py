"""排列三分析页面 — 完整版."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_model.data import LocalDatabase
from football_model.lottery import LotteryRepository, P3AnalysisService
from football_model.ui.components import (
    hero_pro, section_header, empty_state, render_risk_note,
    render_p3_numbers, lottery_number_ball,
)

logger = logging.getLogger(__name__)


def render_p3_analysis(database: LocalDatabase) -> None:
    hero_pro("排列三分析", "历史走势 · 频率统计 · 遗漏分析 · 组合生成", "P3 ANALYSIS", ["直选", "组选", "和值跨度"])

    repo = LotteryRepository(database)
    service = P3AnalysisService(repo)
    draws = repo.get_p3_draws(limit=500)

    if draws.empty:
        empty_state("暂无排列三数据", "请先在数据中心导入历史开奖数据。", "🎲")
        _render_import_section(repo)
        return

    # ── 顶部统计 ──
    latest = draws.iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("总期数", str(len(draws)))
    c2.metric("最近期号", str(latest["issue_no"]))
    c3.metric("开奖日期", str(latest["draw_date"]))
    d1, d2, d3 = int(latest["digit_1"]), int(latest["digit_2"]), int(latest["digit_3"])
    c4.metric("和值", str(d1 + d2 + d3))
    c5.metric("跨度", str(max(d1, d2, d3) - min(d1, d2, d3)))
    c6.metric("形态", str(latest.get("pattern_type", "-")))

    # 最近开奖号码球
    section_header("最近开奖", "最近1期开奖号码。")
    render_p3_numbers(d1, d2, d3)

    # ── Tab结构 ──
    trend_tab, freq_tab, missing_tab, pattern_tab, combo_tab, import_tab = st.tabs([
        "走势分析", "频率统计", "遗漏分析", "形态分析", "组合生成", "数据导入"
    ])

    # ═══ 走势分析 ═══
    with trend_tab:
        _render_trend_section(draws, service)

    # ═══ 频率统计 ═══
    with freq_tab:
        _render_frequency_section(draws, service)

    # ═══ 遗漏分析 ═══
    with missing_tab:
        _render_missing_section(draws, service)

    # ═══ 形态分析 ═══
    with pattern_tab:
        _render_pattern_section(draws, service)

    # ═══ 组合生成 ═══
    with combo_tab:
        _render_combo_section(draws, service)

    # ═══ 数据导入 ═══
    with import_tab:
        _render_import_section(repo)

    render_risk_note(
        "排列三为随机开奖，历史走势不能改变未来开奖概率。"
        "以上分析仅供参考，不构成投注建议。请理性参与，严格控制预算。"
    )


def _render_trend_section(draws: pd.DataFrame, service: P3AnalysisService) -> None:
    """走势分析."""
    section_header("和值走势", "最近50期和值变化。")
    sum_data = draws.head(50).copy()
    sum_data["和值"] = sum_data["digit_1"].astype(int) + sum_data["digit_2"].astype(int) + sum_data["digit_3"].astype(int)
    sum_data = sum_data.iloc[::-1]  # 时间正序

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(sum_data))),
        y=sum_data["和值"].tolist(),
        mode="lines+markers",
        name="和值",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5),
    ))
    # 均值线
    mean_val = sum_data["和值"].mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#94a3b8", annotation_text=f"均值{mean_val:.1f}")
    fig.update_layout(height=280)
    fig = _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    section_header("跨度走势", "最近50期跨度变化。")
    span_data = draws.head(50).copy()
    span_data["跨度"] = span_data.apply(
        lambda r: max(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])) - min(int(r["digit_1"]), int(r["digit_2"]), int(r["digit_3"])),
        axis=1,
    )
    span_data = span_data.iloc[::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(span_data))),
        y=span_data["跨度"].tolist(),
        mode="lines+markers",
        name="跨度",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(height=250)
    fig = _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    section_header("百十个位走势", "最近30期各位数字走势。")
    pos_data = draws.head(30).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(pos_data))), y=pos_data["digit_1"].astype(int).tolist(), mode="lines+markers", name="百位", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=list(range(len(pos_data))), y=pos_data["digit_2"].astype(int).tolist(), mode="lines+markers", name="十位", line=dict(color="#22c55e")))
    fig.add_trace(go.Scatter(x=list(range(len(pos_data))), y=pos_data["digit_3"].astype(int).tolist(), mode="lines+markers", name="个位", line=dict(color="#a855f7")))
    fig.update_layout(height=280)
    fig = _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_frequency_section(draws: pd.DataFrame, service: P3AnalysisService) -> None:
    """频率统计."""
    window_tab1, window_tab2, window_tab3 = st.tabs(["最近30期", "最近50期", "最近100期"])

    for window, tab in [(30, window_tab1), (50, window_tab2), (100, window_tab3)]:
        with tab:
            freq = service.get_frequency(draws, window=window)
            if freq:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(k) for k in freq.keys()],
                    y=list(freq.values()),
                    marker_color="#3b82f6",
                    text=[f"{v:.1%}" for v in freq.values()],
                    textposition="outside",
                ))
                fig.add_hline(y=0.1, line_dash="dash", line_color="#94a3b8", annotation_text="理论均值10%")
                fig.update_layout(height=300, yaxis_tickformat=".1%")
                fig = _apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    # 分位频率
    section_header("分位频率", "百位、十位、个位独立频率。")
    pos1, pos2, pos3 = st.columns(3)
    for pos, col, color in enumerate([(pos1, "#3b82f6"), (pos2, "#22c55e"), (pos3, "#a855f7")], 1):
        col_obj, col_color = color
        with col_obj:
            st.caption(f"{'百' if pos==1 else '十' if pos==2 else '个'}位")
            pos_freq = service.get_position_frequency(draws, pos)
            if pos_freq:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(k) for k in pos_freq.keys()],
                    y=list(pos_freq.values()),
                    marker_color=col_color,
                ))
                fig.add_hline(y=0.1, line_dash="dash", line_color="#94a3b8")
                fig.update_layout(height=200, yaxis_tickformat=".1%", margin=dict(l=30, r=10, t=10, b=30))
                fig = _apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    # 热冷号
    section_header("热冷号排行", "最近30期。")
    hot_cold = service.get_hot_cold_numbers(draws, window=30)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("**🔥 热号**")
        for n in hot_cold["hot"]:
            st.markdown(lottery_number_ball(n, "p3", "small") + ' <span style="color:var(--text-muted)">出现较多</span>', unsafe_allow_html=True)
    with h2:
        st.markdown("**❄️ 冷号**")
        for n in hot_cold["cold"]:
            st.markdown(lottery_number_ball(n, "p3", "small") + ' <span style="color:var(--text-muted)">出现较少</span>', unsafe_allow_html=True)
    with h3:
        st.markdown("**🌤️ 温号**")
        for n in hot_cold["warm"][:5]:
            st.markdown(lottery_number_ball(n, "p3", "small") + ' <span style="color:var(--text-muted)">适中</span>', unsafe_allow_html=True)


def _render_missing_section(draws: pd.DataFrame, service: P3AnalysisService) -> None:
    """遗漏分析."""
    section_header("数字遗漏值", "各数字在各位的当前遗漏期数。")
    missing = service.get_missing_values(draws)

    if missing:
        # 构建表格
        rows = []
        for digit in range(10):
            row = {"数字": digit}
            for pos in [1, 2, 3]:
                key = f"{pos}位_{digit}"
                row[f"{'百' if pos==1 else '十' if pos==2 else '个'}位"] = missing.get(key, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # 遗漏热力图
        section_header("遗漏热力图", "颜色越深遗漏越久。")
        matrix = []
        for digit in range(10):
            row = []
            for pos in [1, 2, 3]:
                row.append(missing.get(f"{pos}位_{digit}", 0))
            matrix.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=["百位", "十位", "个位"],
            y=[str(i) for i in range(10)],
            colorscale=[[0, "#0b1020"], [0.5, "#3b82f6"], [1, "#ef4444"]],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(color="white"),
        ))
        fig.update_layout(height=350)
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def _render_pattern_section(draws: pd.DataFrame, service: P3AnalysisService) -> None:
    """形态分析."""
    section_header("豹子/组三/组六分布", "整体占比。")
    patterns = service.get_pattern_distribution(draws)
    if patterns:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=list(patterns.keys()),
                values=list(patterns.values()),
                hole=0.45,
                marker=dict(colors=["#ef4444", "#eab308", "#22c55e"]),
                textinfo="label+percent",
            ))
            fig.update_layout(height=280)
            fig = _apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            for k, v in patterns.items():
                color = "#ef4444" if k == "豹子" else "#eab308" if k == "组三" else "#22c55e"
                st.markdown(f'<span style="color:{color};font-weight:700">{k}</span>: {v:.1%}', unsafe_allow_html=True)

    section_header("奇偶分布", "奇偶比例。")
    draws_copy = draws.copy()
    draws_copy["奇偶"] = draws_copy.apply(
        lambda r: f"{sum(1 for d in [int(r['digit_1']),int(r['digit_2']),int(r['digit_3'])] if d%2==1)}奇{sum(1 for d in [int(r['digit_1']),int(r['digit_2']),int(r['digit_3'])] if d%2==0)}偶",
        axis=1,
    )
    odd_even = draws_copy["奇偶"].value_counts(normalize=True).to_dict()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=list(odd_even.keys()),
            values=list(odd_even.values()),
            hole=0.45,
            textinfo="label+percent",
        ))
        fig.update_layout(height=250)
        fig = _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        for k, v in sorted(odd_even.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"{k}: {v:.1%}")

    section_header("大小分布", "大(5-9)/小(0-4)比例。")
    draws_copy["大小"] = draws_copy.apply(
        lambda r: f"{sum(1 for d in [int(r['digit_1']),int(r['digit_2']),int(r['digit_3'])] if d>=5)}大{sum(1 for d in [int(r['digit_1']),int(r['digit_2']),int(r['digit_3'])] if d<5)}小",
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
        for k, v in sorted(big_small.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"{k}: {v:.1%}")


def _render_combo_section(draws: pd.DataFrame, service: P3AnalysisService) -> None:
    """组合生成."""
    section_header("参考组合生成", "基于统计过滤的候选组合。")

    with st.expander("过滤条件", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sum_min = st.number_input("和值最小", 0, 27, 5, key="p3-sum-min")
            sum_max = st.number_input("和值最大", 0, 27, 22, key="p3-sum-max")
        with f2:
            span_min = st.number_input("跨度最小", 0, 9, 1, key="p3-span-min")
            span_max = st.number_input("跨度最大", 0, 9, 9, key="p3-span-max")
        with f3:
            pattern = st.selectbox("形态", ["不限", "组三", "组六"], key="p3-pattern")
            exclude_recent = st.number_input("排除近N期已开号", 0, 100, 10, key="p3-exclude")
            combo_count = st.number_input("生成数量", 1, 50, 10, key="p3-combo-count")

    if st.button("🎲 生成参考组合", key="p3-generate", type="primary"):
        combos = service.generate_reference_combinations(
            draws,
            count=combo_count,
            sum_range=(sum_min, sum_max),
            span_range=(span_min, span_max),
            pattern=pattern if pattern != "不限" else None,
            exclude_recent=exclude_recent,
        )
        if combos:
            df = pd.DataFrame(combos)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(f"共生成 {len(combos)} 组参考组合")
        else:
            st.warning("未找到满足条件的组合，请放宽过滤条件。")


def _render_import_section(repo: LotteryRepository) -> None:
    """数据导入."""
    section_header("导入排列三数据", "支持CSV格式导入。")

    template = pd.DataFrame([
        {"issue_no": "2024001", "draw_date": "2024-01-01", "digit_1": 3, "digit_2": 7, "digit_3": 2},
        {"issue_no": "2024002", "draw_date": "2024-01-02", "digit_1": 5, "digit_2": 1, "digit_3": 8},
    ])
    st.download_button(
        "📥 下载CSV模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        "p3_template.csv",
        "text/csv",
        key="p3-template-dl",
    )

    upload = st.file_uploader("上传CSV文件", type=["csv"], key="p3-upload")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)
            if st.button("确认导入", key="p3-import-confirm", type="primary"):
                count = repo.import_p3_from_csv(upload)
                st.success(f"成功导入 {count} 期数据")
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")


def _apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
    )
    return fig
