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
    render_p3_numbers, lottery_number_ball, plotly_theme,
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

    # ── 生成下一期候选组合 ──
    _render_next_issue_section(database, repo, service, draws)

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
                result = repo.import_p3_from_csv(upload)
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


def _render_next_issue_section(database, repo, service, draws):
    """渲染生成下一期候选组合区域."""
    from football_model.lottery.p3_predictor import (
        infer_next_p3_issue, score_p3_candidates,
        analyze_basis, validate_p3_history,
    )

    section_header("下一期候选组合 / Next Issue Analysis", "基于历史数据统计分析生成候选组合。")

    # 风险提示
    st.caption(
        "⚠️ 本功能仅基于历史开奖数据进行统计分析和候选组合筛选，不构成中奖承诺。"
        "排列三开奖结果具有随机性，历史走势不能决定未来结果，请理性参考。"
    )

    # 高级参数
    with st.expander("高级生成参数 / Advanced Parameters", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            candidate_count = st.slider("候选数量 / Count", 5, 50, 10, key="p3-candidate-count")
            history_window = st.slider("历史窗口 / History Window", 30, 300, 100, key="p3-history-window")
        with f2:
            recent_window = st.slider("近期窗口 / Recent Window", 10, 100, 30, key="p3-recent-window")
            exclude_recent_n = st.slider("排除近N期已开号 / Exclude Recent", 0, 30, 5, key="p3-exclude-recent")
        seed = st.number_input("随机种子 / Random Seed", value=42, key="p3-seed")

    # 生成按钮
    if st.button("🎲 生成下一期候选组合 / Generate Candidates", key="p3-generate-next-issue", type="primary"):
        # 校验数据
        valid, warnings = validate_p3_history(draws)
        if not valid:
            empty_state("历史数据不足 / Insufficient Data", "请先导入足够的排列三历史开奖数据后再生成候选组合。", "⚠️")
            for w in warnings:
                st.warning(w)
            return

        # 生成候选组合
        with st.spinner("正在分析历史数据并生成候选组合... / Analyzing..."):
            candidates = score_p3_candidates(
                history=draws,
                candidate_count=candidate_count,
                history_window=history_window,
                recent_window=recent_window,
                exclude_recent_n=exclude_recent_n,
                seed=seed,
            )
            basis = analyze_basis(draws)

        # 保存到session_state
        st.session_state["p3_next_candidates"] = candidates
        st.session_state["p3_next_target_issue"] = infer_next_p3_issue(str(draws.iloc[0]["issue_no"]))
        st.session_state["p3_next_basis"] = basis

        # 显示警告
        for w in warnings:
            st.warning(w)

    # 显示结果（如果存在）
    if "p3_next_candidates" in st.session_state:
        _display_p3_candidates(
            st.session_state["p3_next_candidates"],
            st.session_state.get("p3_next_target_issue", "下一期"),
            st.session_state.get("p3_next_basis", None),
            draws,
        )


def _display_p3_candidates(candidates: pd.DataFrame, target_issue: str, basis, draws: pd.DataFrame):
    """显示候选组合结果."""
    from football_model.ui.components import lottery_number_ball
    from football_model.lottery.p3_predictor import AnalysisBasis

    # 基础信息
    section_header(f"第{target_issue}期候选组合 / Candidates", f"共 {len(candidates)} 组候选。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目标期号 / Issue", str(target_issue))
    c2.metric("候选数量 / Count", str(len(candidates)))
    c3.metric("数据样本 / Samples", f"{basis.sample_count if isinstance(basis, AnalysisBasis) else len(draws)}期")
    c4.metric("数据质量 / Quality", basis.data_quality if isinstance(basis, AnalysisBasis) else "中")

    # 分析依据
    if isinstance(basis, AnalysisBasis):
        section_header("分析依据 / Analysis Basis", "候选组合生成的统计基础。")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**热号 / Hot Numbers**")
            html = " ".join(lottery_number_ball(n, "p3-pos1", "small") for n in basis.hot_numbers)
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("**冷号 / Cold Numbers**")
            html = " ".join(lottery_number_ball(n, "p3-pos3", "small") for n in basis.cold_numbers)
            st.markdown(html, unsafe_allow_html=True)

            st.markdown(f"**高频和值 / Top Sums**: {', '.join(str(s) for s in basis.high_freq_sums)}")
            st.markdown(f"**高频跨度 / Top Spans**: {', '.join(str(s) for s in basis.high_freq_spans)}")

        with col2:
            st.markdown("**形态分布 / Pattern Distribution**")
            for k, v in basis.pattern_distribution.items():
                st.caption(f"{k}: {v:.1%}")

            st.markdown("**近5期 / Recent 5 Draws**")
            for d in basis.recent_5_draws:
                st.caption(f"第{d['期号']}期: {d['号码']} (和值{d['和值']}，跨度{d['跨度']}，{d['形态']})")

    # 候选组合卡片
    section_header("候选组合 / Candidate Combinations", "按综合评分排序。")
    for i, row in candidates.iterrows():
        d1, d2, d3 = int(row["digits"][0]), int(row["digits"][1]), int(row["digits"][2])
        balls = (
            lottery_number_ball(d1, "p3-pos1") +
            lottery_number_ball(d2, "p3-pos2") +
            lottery_number_ball(d3, "p3-pos3")
        )

        risk_color = "#16a34a" if row["risk_level"] == "低风险" else "#ca8a04" if row["risk_level"] == "中风险" else "#ef4444"

        st.markdown(
            f"""<div class="match-card anim-fade" style="padding:0.8rem 1rem">
<div style="display:flex;justify-content:space-between;align-items:center">
  <div style="display:flex;align-items:center;gap:1rem">
    {balls}
    <div>
      <div style="font-size:0.75rem;color:var(--text-muted)">#{i+1} · 直选 {row['number']}</div>
      <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.2rem">
        和值{row['sum_value']} · 跨度{row['span_value']} · {row['pattern_type']} · {row['odd_even']} · {row['big_small']} · 012路{row['road_012']}
      </div>
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:1.2rem;font-weight:700;color:var(--text-primary)">{row['score']:.1f}</div>
    <div style="font-size:0.75rem;color:{risk_color}">{row['risk_level']}</div>
  </div>
</div>
<div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.5rem">{row['reason']}</div>
</div>""",
            unsafe_allow_html=True,
        )

    # 评分细节
    with st.expander("评分细节 / Score Details", expanded=False):
        detail = candidates[["number", "frequency_score", "omission_score", "sum_score", "span_score", "pattern_score", "score"]].copy()
        detail.columns = ["号码", "频率分", "遗漏分", "和值分", "跨度分", "形态分", "综合分"]
        st.dataframe(detail, hide_index=True, use_container_width=True)

    # 表格展示
    with st.expander("完整数据表格 / Full Data Table", expanded=False):
        display = candidates[["number", "sum_value", "span_value", "pattern_type", "odd_even", "big_small", "road_012", "score", "risk_level"]].copy()
        display.columns = ["号码", "和值", "跨度", "形态", "奇偶", "大小", "012路", "评分", "风险"]
        st.dataframe(display, hide_index=True, use_container_width=True)

    # CSV下载
    csv_data = candidates.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载候选组合 CSV / Download CSV",
        data=csv_data,
        file_name=f"p3_candidates_{target_issue}.csv",
        mime="text/csv",
        key="p3-download-next-candidates",
    )

    # 风险提示
    render_risk_note(
        "排列三开奖结果具有随机性。以上候选组合仅基于历史开奖数据的频率、遗漏、和值、跨度和形态分布进行筛选，"
        "不构成中奖承诺。历史走势不能决定未来开奖结果，请理性参考并控制预算。"
    )


