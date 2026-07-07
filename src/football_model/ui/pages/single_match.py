from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from football_model.data import LocalDatabase, ModelRepository
from football_model.engine import estimate_expected_goals
from football_model.models import DixonColesModel
from football_model.services import AnalysisService
from football_model.ui.components import format_percent_columns, hero_pro, plotly_theme, probability_chart, render_risk_note, score_heatmap

logger = logging.getLogger(__name__)


@st.cache_resource
def _load_model(artifact_path: str, modified_at: float) -> DixonColesModel:
    del modified_at
    return DixonColesModel.load(artifact_path)


def _trained_inputs(database: LocalDatabase) -> tuple[str, str, float, float, float, str] | None:
    models = ModelRepository(database).trained_models()
    models = models.loc[models["model_type"] == "Dixon-Coles League"]
    if models.empty:
        return None
    model_ids = models["model_id"].tolist()
    selected_id = st.selectbox("联赛模型版本", model_ids)
    record = models.loc[models["model_id"] == selected_id].iloc[0]
    artifact_path = Path(record["artifact_path"])
    model = _load_model(str(artifact_path), artifact_path.stat().st_mtime)
    home_col, versus, away_col = st.columns([5, 1, 5])
    with home_col:
        home_team = st.selectbox("主队", model.teams, index=0)
    with versus:
        st.markdown("<div style='text-align:center;padding-top:2rem;color:#3b82f6;font-weight:700'>VS</div>", unsafe_allow_html=True)
    with away_col:
        away_team = st.selectbox("客队", model.teams, index=min(1, len(model.teams) - 1))
    home_xg, away_xg = model.expected_goals(home_team, away_team)
    st.caption(
        f"训练赛事：{model.competition} ｜ 样本：{model.metrics['matches']}场 ｜ "
        f"截止：{model.training_cutoff[:10]} ｜ 低比分修正 ρ={model.rho:.3f}"
    )
    return home_team, away_team, home_xg, away_xg, model.rho, str(record["version"])


def _manual_inputs() -> tuple[str, str, float, float, float, str]:
    name_col1, versus, name_col2 = st.columns([5, 1, 5])
    with name_col1:
        home_team = st.text_input("主队", "主队 A")
    with versus:
        st.markdown("<div style='text-align:center;padding-top:2rem;color:#3b82f6;font-weight:700'>VS</div>", unsafe_allow_html=True)
    with name_col2:
        away_team = st.text_input("客队", "客队 B")

    mode = st.radio("预期进球生成方式", ["球队数据估算", "直接输入 xG"], horizontal=True)
    if mode == "球队数据估算":
        left, middle, right = st.columns(3)
        with left:
            st.subheader(f"{home_team} · 近况")
            home_scored = st.number_input("主队场均进球", 0.0, 5.0, 1.75, 0.05)
            home_conceded = st.number_input("主队场均失球", 0.0, 5.0, 1.05, 0.05)
            home_recent_xg = st.number_input("主队近期 xG", 0.0, 5.0, 1.62, 0.05)
        with middle:
            st.subheader("联赛基准")
            league_home_avg = st.number_input("联赛主队场均进球", 0.5, 3.0, 1.48, 0.01)
            league_away_avg = st.number_input("联赛客队场均进球", 0.5, 3.0, 1.18, 0.01)
            xg_weight = st.slider("近期 xG 权重", 0.0, 0.75, 0.35, 0.05)
        with right:
            st.subheader(f"{away_team} · 近况")
            away_scored = st.number_input("客队场均进球", 0.0, 5.0, 1.20, 0.05)
            away_conceded = st.number_input("客队场均失球", 0.0, 5.0, 1.45, 0.05)
            away_recent_xg = st.number_input("客队近期 xG", 0.0, 5.0, 1.15, 0.05)
        home_xg, away_xg = estimate_expected_goals(
            home_scored=home_scored, home_conceded=home_conceded,
            away_scored=away_scored, away_conceded=away_conceded,
            league_home_avg=league_home_avg, league_away_avg=league_away_avg,
            home_recent_xg=home_recent_xg, away_recent_xg=away_recent_xg,
            xg_weight=xg_weight,
        )
    else:
        xg_col1, xg_col2 = st.columns(2)
        home_xg = xg_col1.number_input("主队预期进球", 0.05, 5.0, 1.65, 0.05)
        away_xg = xg_col2.number_input("客队预期进球", 0.05, 5.0, 1.10, 0.05)
    return home_team, away_team, home_xg, away_xg, -0.08, "manual-baseline"


def render_single_match(service: AnalysisService, database: LocalDatabase) -> None:
    hero_pro("单场概率分析", "使用已训练联赛模型或手工参数生成比分分布，并与SP隐含概率对照。", "SINGLE MATCH", ["比分矩阵", "让球", "总进球"])
    trained_models = ModelRepository(database).trained_models()
    has_trained_model = not trained_models.loc[trained_models["model_type"] == "Dixon-Coles League"].empty
    options = ["已训练联赛模型", "手工参数模型"] if has_trained_model else ["手工参数模型"]
    source_mode = st.radio("分析引擎", options, horizontal=True)
    inputs = _trained_inputs(database) if source_mode == "已训练联赛模型" else _manual_inputs()
    if inputs is None:
        st.error("训练模型文件不可用，请到模型中心重新训练。")
        return
    home_team, away_team, home_xg, away_xg, rho, model_version = inputs

    with st.expander("SP与让球设置", expanded=True):
        odds_col1, odds_col2, odds_col3, handicap_col = st.columns(4)
        odds_home = odds_col1.number_input("主胜SP", 1.01, 50.0, 1.95, 0.01)
        odds_draw = odds_col2.number_input("平局SP", 1.01, 50.0, 3.35, 0.01)
        odds_away = odds_col3.number_input("客胜SP", 1.01, 50.0, 3.75, 0.01)
        handicap_value = handicap_col.selectbox("主队让球", [-3, -2, -1, 0, 1, 2, 3], index=2)

    result = service.analyze(
        home_xg=home_xg, away_xg=away_xg,
        odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away,
        handicap=handicap_value, rho=rho,
    )
    summary = result.summary
    st.caption(f"当前模型：{model_version} ｜ {home_team} vs {away_team}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("主队预期进球", f"{home_xg:.2f}")
    m2.metric("客队预期进球", f"{away_xg:.2f}")
    m3.metric("最高概率赛果", max(summary.probabilities, key=summary.probabilities.get))
    m4.metric("最高概率比分", summary.top_scores[0][0], f"{summary.top_scores[0][1]:.1%}")

    chart_col, table_col = st.columns([1.15, 1])
    with chart_col:
        st.subheader("模型与市场概率")
        st.plotly_chart(probability_chart(result.comparison), width="stretch")
    with table_col:
        st.subheader("价值差异")
        display = format_percent_columns(result.comparison, ["模型概率", "市场概率", "概率差", "理论EV"])
        display["公平SP"] = display["公平SP"].map(lambda value: f"{value:.2f}")
        st.dataframe(display, hide_index=True, width="stretch")
        best = result.comparison.loc[result.comparison["理论EV"].idxmax()]
        if best["理论EV"] >= 0.05:
            st.success(f"统计观察：{best['结果']} 理论EV {best['理论EV']:.1%}，仍需样本外回测。")
        else:
            st.info("当前输入下没有达到5%理论EV观察阈值的结果。")

    heat_col, side_col = st.columns([1.35, 1])
    with heat_col:
        st.subheader("比分概率矩阵")
        st.plotly_chart(score_heatmap(result.matrix), width="stretch")
    with side_col:
        st.subheader("最可能比分")
        score_frame = pd.DataFrame(summary.top_scores[:6], columns=["比分", "概率"])
        score_frame["概率"] = score_frame["概率"].map(lambda value: f"{value:.1%}")
        st.dataframe(score_frame, hide_index=True, width="stretch")
        st.subheader("让球胜平负")
        st.caption(f"主队让球：{handicap_value:+d}")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("胜", f"{result.handicap['胜']:.1%}")
        hc2.metric("平", f"{result.handicap['平']:.1%}")
        hc3.metric("负", f"{result.handicap['负']:.1%}")
        st.subheader("总进球")
        totals = pd.DataFrame({"总进球": summary.total_goals.keys(), "概率": summary.total_goals.values()})
        totals_figure = px.bar(totals, x="总进球", y="概率", color_discrete_sequence=["#3b82f6"])
        totals_figure.update_layout(
            height=235, margin=dict(l=10, r=10, t=5, b=10),
            yaxis_tickformat=".0%",
        )
        plotly_theme(totals_figure)
        st.plotly_chart(totals_figure, width="stretch")

    render_risk_note("开发样本中的市场赔率不是中国体彩官方SP。理论EV不等于实际收益，必须通过严格滚动回测。")
