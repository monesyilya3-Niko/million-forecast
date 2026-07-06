from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from football_model.core import TEAM_MAPS, competition_for_league
from football_model.data import (
    LocalDatabase,
    MatchRepository,
    ModelRepository,
    PredictionRepository,
    SportteryRepository,
)
from football_model.engine import infer_expected_goals_from_market, market_comparison, score_matrix, summarize_market
from football_model.models import DixonColesModel, MatchIntelligence, build_match_intelligence
from football_model.services import LiveContextService, SportteryLiveService
from football_model.services.ensemble_service import EnsembleAnalysisService
from football_model.ui.components import (
    empty_state,
    format_percent,
    format_percent_columns,
    hero_pro,
    metric_card,
    plotly_theme,
    probability_chart,
    render_confidence_meter,
    render_risk_note,
    score_heatmap,
    section_header,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def _load_dc_model(path: str, modified_at: float) -> DixonColesModel:
    del modified_at
    return DixonColesModel.load(path)


@st.cache_data(ttl=300)
def _training_frame(database_path: str, competition: str) -> pd.DataFrame:
    return MatchRepository(LocalDatabase(database_path)).training_frame(competition)


def _competition_for_league(league_name: str) -> str | None:
    return competition_for_league(league_name)




def _series_get(row: pd.Series, field: str, default: object = None) -> object:
    """Read a field from a pandas Series without raising when optional columns are absent."""
    try:
        return row.get(field, default)
    except AttributeError:
        return default

def _probabilities(matrix: np.ndarray) -> dict[str, float]:
    return {
        "主胜": float(np.tril(matrix, k=-1).sum()),
        "平局": float(np.trace(matrix)),
        "客胜": float(np.triu(matrix, k=1).sum()),
    }


def _confidence(
    intelligence: MatchIntelligence | None,
    independent_matrix: np.ndarray | None,
    market_matrix: np.ndarray,
    history: pd.DataFrame,
) -> tuple[int, str, list[str]]:
    score = 20.0
    risks: list[str] = []
    if intelligence:
        sample = min(intelligence.home.matches, intelligence.away.matches)
        recent = min(intelligence.home.recent_matches, intelligence.away.recent_matches)
        score += min(sample / 40, 1) * 20 + min(recent / 8, 1) * 10
        if recent < 6:
            risks.append("近期有效样本不足6场")
    else:
        risks.append("缺少球队状态与Elo数据")
    if independent_matrix is not None:
        score += 20
        model_probs = np.array(list(_probabilities(independent_matrix).values()))
        market_probs = np.array(list(_probabilities(market_matrix).values()))
        disagreement = float(np.abs(model_probs - market_probs).max())
        score += max(0, 15 * (1 - disagreement / 0.18))
        if disagreement > 0.10:
            risks.append(f"独立模型与市场最大分歧达到{disagreement:.1%}")
    else:
        risks.append("没有可用的独立进球模型")
    snapshots = history["captured_at"].nunique() if not history.empty else 0
    score += min(snapshots / 6, 1) * 15
    if snapshots < 3:
        risks.append("盘口快照过少，无法判断稳定趋势")
    final = int(np.clip(round(score), 0, 100))
    grade = "高" if final >= 75 else "中" if final >= 55 else "低"
    return final, grade, risks


def _verdict(probabilities: dict[str, float], comparison: pd.DataFrame, confidence: int) -> tuple[str, str]:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    leader, leader_probability = ordered[0]
    gap = leader_probability - ordered[1][1]
    value_row = comparison.loc[comparison["概率差"].idxmax()]
    strength = "明显" if gap >= 0.14 else "轻微" if gap >= 0.06 else "不明显"
    headline = f"{leader}概率最高（{leader_probability:.1%}），优势{strength}"
    value_result = value_row["结果"]
    value_diff = value_row["概率差"]
    explanation = (
        f"第一与第二结果相差{gap:.1%}；模型相对官方市场分歧最大的是「{value_result}」"
        f"（{value_diff:+.1%}）。当前可信度{confidence}/100，结论应按概率而不是确定赛果理解。"
    )
    return headline, explanation


def _team_table(intelligence: MatchIntelligence) -> pd.DataFrame:
    rows = []
    for label, team in [("主队", intelligence.home), ("客队", intelligence.away)]:
        rows.append({
            "球队": f"{label} · {team.team}",
            "Elo": round(team.elo),
            "Elo排名": team.elo_rank,
            "历史场次": team.matches,
            "近8场": f"{team.wins}胜 {team.draws}平 {team.losses}负",
            "场均积分": team.points_per_game,
            "场均进球": team.goals_for,
            "场均失球": team.goals_against,
            "近5场序列": team.form,
        })
    return pd.DataFrame(rows)


def _independent_weight(competition: str | None, metrics: dict[str, object]) -> float:
    log_loss = metrics.get("holdout_log_loss")
    if not isinstance(log_loss, (float, int)):
        return 0.52 if competition == "瑞典超级联赛" else 0.42
    validation_score = float(np.clip((1.16 - float(log_loss)) / 0.28, 0, 1))
    if competition == "瑞典超级联赛":
        return 0.52 + 0.18 * validation_score
    return 0.42 + 0.16 * validation_score


def _recent_match_rows(training: pd.DataFrame, team: str, cutoff: object, limit: int = 8) -> pd.DataFrame:
    data = training.loc[
        (training["kickoff"] < pd.to_datetime(cutoff))
        & ((training["home_team"] == team) | (training["away_team"] == team))
    ].tail(limit)
    rows = []
    for match in data.iloc[::-1].itertuples(index=False):
        is_home = match.home_team == team
        goals_for = int(match.home_goals if is_home else match.away_goals)
        goals_against = int(match.away_goals if is_home else match.home_goals)
        rows.append({
            "日期": pd.to_datetime(match.kickoff).strftime("%Y-%m-%d"),
            "场地": "主" if is_home else "客",
            "对手": match.away_team if is_home else match.home_team,
            "比分": f"{goals_for}:{goals_against}",
            "结果": "胜" if goals_for > goals_against else "平" if goals_for == goals_against else "负",
        })
    return pd.DataFrame(rows)


def _head_to_head_rows(training: pd.DataFrame, home_team: str, away_team: str, cutoff: object) -> pd.DataFrame:
    teams = {home_team, away_team}
    data = training.loc[
        (training["kickoff"] < pd.to_datetime(cutoff))
        & training["home_team"].isin(teams)
        & training["away_team"].isin(teams)
    ].tail(10)
    if data.empty:
        return pd.DataFrame(columns=["日期", "比赛", "比分"])
    return pd.DataFrame({
        "日期": pd.to_datetime(data["kickoff"]).dt.strftime("%Y-%m-%d"),
        "比赛": data["home_team"] + " vs " + data["away_team"],
        "比分": data["home_goals"].astype(int).astype(str) + ":" + data["away_goals"].astype(int).astype(str),
    }).iloc[::-1]


def _market_movement(history: pd.DataFrame, market: str = "HAD") -> pd.DataFrame:
    data = history.loc[history["market"] == market].sort_values("captured_at")
    if data.empty:
        return pd.DataFrame()
    labels = {"H": "主胜", "D": "平局", "A": "客胜"}
    rows = []
    first_time = data["captured_at"].min()
    last_time = data["captured_at"].max()
    first = data.loc[data["captured_at"] == first_time].set_index("selection")["odds"]
    last = data.loc[data["captured_at"] == last_time].set_index("selection")["odds"]
    common = [s for s in ["H", "D", "A"] if s in first and s in last]
    first_implied = 1 / first[common].astype(float)
    last_implied = 1 / last[common].astype(float)
    first_prob = first_implied / first_implied.sum()
    last_prob = last_implied / last_implied.sum()
    for selection in common:
        odds_change = float(last[selection] - first[selection])
        probability_change = float(last_prob[selection] - first_prob[selection])
        selection_odds = data.loc[data["selection"] == selection, "odds"].astype(float)
        volatility = float(selection_odds.std(ddof=0) / selection_odds.mean()) if len(selection_odds) > 1 else 0.0
        rows.append({
            "结果": labels[selection],
            "初始SP": float(first[selection]),
            "最新SP": float(last[selection]),
            "SP变化": odds_change,
            "SP波动率": volatility,
            "市场概率变化": probability_change,
            "方向": "增强" if probability_change > 0.002 else "减弱" if probability_change < -0.002 else "稳定",
        })
    return pd.DataFrame(rows)


def render_match_analysis(
    database: LocalDatabase,
    live_service: SportteryLiveService,
    ensemble_service: EnsembleAnalysisService,
) -> None:
    match_id = st.session_state.get("analysis_match_id")
    if not match_id:
        hero_pro("比赛分析", "多模型集成 · 市场概率 · 球队情报 · 风险校验", "MATCH INTELLIGENCE", meta=["等待选择比赛", "本地模型报告"])
        empty_state("没有选中的比赛", "请先从今日竞彩页面选择一场比赛，再进入模型报告。", "📊")
        return

    selected = None
    for business_date in live_service.dates():
        found = live_service.matches_for_date(business_date).loc[lambda frame: frame["match_id"] == match_id]
        if not found.empty:
            selected = found.iloc[0]
            break
    if selected is None:
        empty_state("找不到该比赛", "本地数据库中没有该 match_id，请返回今日竞彩刷新后重试。", "⚠️")
        return

    kickoff_at = pd.to_datetime(_series_get(selected, "kickoff", pd.Timestamp.now()))
    hero_pro(
        title="比赛分析",
        subtitle=(
            f"{_series_get(selected, 'home_team', '主队')} vs {_series_get(selected, 'away_team', '客队')} · "
            f"{_series_get(selected, 'match_number', '')} · {_series_get(selected, 'league_name', '未知联赛')} · "
            f"{kickoff_at:%Y-%m-%d %H:%M}"
        ),
        eyebrow="MATCH INTELLIGENCE",
        meta=["多模型集成", "市场概率", "球队情报", "风险校验"],
    )

    if st.button("📋 查看比赛详情（阵容/伤停/赛况）", key=f"match-analysis-detail-{match_id}"):
        st.session_state["detail_match_id"] = match_id
        st.session_state["requested_page"] = "📄 比赛详情"
        st.rerun()

    if any(pd.isna(_series_get(selected, column)) for column in ["had_h", "had_d", "had_a"]):
        empty_state("官方胜平负 SP 尚未发布", "缺少 had_h / had_d / had_a 时不会执行完整模型报告。", "🟡")
        return

    odds = {
        "主胜": float(_series_get(selected, "had_h")),
        "平局": float(_series_get(selected, "had_d")),
        "客胜": float(_series_get(selected, "had_a")),
    }
    market_xg = infer_expected_goals_from_market(*odds.values())
    market_matrix = score_matrix(*market_xg)
    history = SportteryRepository(database).odds_history(match_id)
    competition = _competition_for_league(str(_series_get(selected, "league_name", "")))
    sync_col, provider_col = st.columns([1, 3])
    with sync_col:
        if st.button("更新伤停与首发", key=f"match-analysis-sync-context-{match_id}", width="stretch"):
            sync_result = LiveContextService(database).sync_match(selected, competition or "")
            st.session_state["context_sync_message"] = sync_result.message or str(sync_result)
            st.rerun()
    with provider_col:
        st.caption(
            st.session_state.get("context_sync_message", "实时阵容：等待供应商同步；通常在开赛前20–40分钟发布。")
        )
    model_record = (
        ModelRepository(database).latest_for_competition(competition, model_type="Dixon-Coles League")
        if competition
        else None
    )
    mapped_home = mapped_away = None
    dc_xg = form_xg = None
    dc_matrix = form_matrix = independent_matrix = None
    intelligence = None
    training: pd.DataFrame | None = None
    model_version = "未匹配"
    model_metrics: dict[str, object] = {}
    model: DixonColesModel | None = None
    with database.connection(read_only=True) as connection:
        provider_row = connection.execute(
            "SELECT venue FROM provider_fixtures WHERE match_id=?", [match_id]
        ).fetchone()
    provider_venue = provider_row[0] if provider_row else None
    production_prediction = ensemble_service.predict(
        home_team=str(_series_get(selected, "home_team", "")),
        away_team=str(_series_get(selected, "away_team", "")),
        competition=competition or "",
        league_name=str(_series_get(selected, "league_name", "")),
        odds_home=odds["主胜"],
        odds_draw=odds["平局"],
        odds_away=odds["客胜"],
        kickoff=kickoff_at,
        venue=provider_venue,
        match_id=match_id,
    )
    if production_prediction is not None and pd.Timestamp.now() < kickoff_at:
        cutoff_at = _series_get(selected, "odds_updated_at")
        if pd.isna(cutoff_at):
            cutoff_at = _series_get(selected, "last_update", pd.Timestamp.now())
        PredictionRepository(database).save(
            match_id=match_id,
            model_version=production_prediction.model_version,
            cutoff_at=cutoff_at,
            home_probability=production_prediction.home_win,
            draw_probability=production_prediction.draw,
            away_probability=production_prediction.away_win,
            home_xg=production_prediction.home_xg,
            away_xg=production_prediction.away_xg,
            confidence=production_prediction.confidence,
            components=production_prediction.components,
            input_odds=odds,
        )
    if model_record is not None and competition:
        mapping = TEAM_MAPS.get(competition, {})
        mapped_home = mapping.get(str(_series_get(selected, "home_team", "")))
        mapped_away = mapping.get(str(_series_get(selected, "away_team", "")))
        artifact = Path(model_record["artifact_path"])
        model = _load_dc_model(str(artifact), artifact.stat().st_mtime)
        if mapped_home in model.teams and mapped_away in model.teams:
            dc_xg = model.expected_goals(mapped_home, mapped_away)
            dc_matrix = score_matrix(*dc_xg, rho=model.rho)
            model_version = str(model_record["version"])
            model_metrics = json.loads(model_record["metrics_json"])
            training = _training_frame(str(database.path), competition)
            intelligence = build_match_intelligence(
                training, mapped_home, mapped_away,
                as_of=kickoff_at,
                neutral_venue=competition == "世界杯国家队",
            )
            form_xg = (intelligence.home_xg, intelligence.away_xg)
            form_matrix = score_matrix(*form_xg, rho=model.rho)
            independent_matrix = 0.62 * dc_matrix + 0.38 * form_matrix
            independent_matrix /= independent_matrix.sum()
    if production_prediction is not None:
        ensemble_matrix = production_prediction.score_matrix
        ensemble_xg = (production_prediction.home_xg, production_prediction.away_xg)
        mapped_home = production_prediction.mapped_home
        mapped_away = production_prediction.mapped_away
        model_version = production_prediction.model_version
        component_text = " · ".join(f"{name} {weight:.0%}" for name, weight in production_prediction.weights.items())
    elif independent_matrix is not None and dc_xg and form_xg:
        independent_weight = _independent_weight(competition, model_metrics)
        ensemble_matrix = independent_weight * independent_matrix + (1 - independent_weight) * market_matrix
        component_text = (
            f"DC {independent_weight * 0.62:.0%} · Elo状态 {independent_weight * 0.38:.0%} · "
            f"市场 {1 - independent_weight:.0%}"
        )
        ensemble_xg = (
            independent_weight * (0.62 * dc_xg[0] + 0.38 * form_xg[0]) + (1 - independent_weight) * market_xg[0],
            independent_weight * (0.62 * dc_xg[1] + 0.38 * form_xg[1]) + (1 - independent_weight) * market_xg[1],
        )
    else:
        ensemble_matrix = market_matrix
        ensemble_xg = market_xg
        component_text = "仅市场先验"
    ensemble_matrix /= ensemble_matrix.sum()
    summary = summarize_market(ensemble_matrix, *ensemble_xg)
    comparison = market_comparison(summary.probabilities, odds)
    if production_prediction is not None:
        confidence = production_prediction.confidence
        quality = production_prediction.confidence_label
        risks = list(production_prediction.risks)
        if history["captured_at"].nunique() < 3 if not history.empty else True:
            risks.append("盘口快照不足3个，趋势判断不稳定")
    else:
        confidence, quality, risks = _confidence(intelligence, independent_matrix, market_matrix, history)
    headline, explanation = _verdict(summary.probabilities, comparison, confidence)

    # ── Top-level insight cards ──
    section_header("核心结论区", "先看模型倾向、可信度、预期进球和主要风险。", f"模型结构：{component_text}")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    leader = max(summary.probabilities, key=summary.probabilities.get)
    top_score, top_score_prob = summary.top_scores[0]
    max_gap_row = comparison.loc[comparison["概率差"].abs().idxmax()] if "概率差" in comparison.columns and not comparison.empty else None
    max_gap_text = f"{max_gap_row['结果']} {max_gap_row['概率差']:+.1%}" if max_gap_row is not None else "-"
    with s1:
        metric_card("可信度", f"{confidence}/100", delta=quality, variant="positive" if confidence >= 75 else "warning" if confidence >= 55 else "danger")
    with s2:
        metric_card("模型倾向", leader, variant="info")
    with s3:
        metric_card("预期进球", f"{sum(ensemble_xg):.2f}", delta=f"{ensemble_xg[0]:.2f} : {ensemble_xg[1]:.2f}", variant="purple")
    with s4:
        metric_card("首选比分", top_score.replace(":", "–"), delta=f"{top_score_prob:.1%}", variant="info")
    with s5:
        metric_card("最大分歧", max_gap_text, variant="warning")
    with s6:
        metric_card("风险数量", str(len(risks)), variant="danger" if risks else "positive")
    render_confidence_meter(confidence, "报告可信度")
    render_risk_note("；".join(risks) if risks else "当前未发现显著的数据完整性或模型分歧风险，但结论仍应按概率信号理解。")

    # ── Tabs ──
    verdict_tab, model_tab, market_tab, form_tab, prev_tab, lineup_tab, tactical_tab, data_tab = st.tabs(
        ["核心结论", "概率模型", "赔率市场", "球队状态", "上场复盘", "首发伤停", "技战术解读", "数据质量"]
    )

    # Alias for backward compatibility
    score_tab = model_tab  # Score heatmap goes in model tab now

    with verdict_tab:
        section_header("核心结论", headline, "概率报告")
        st.write(explanation)

        # Value analysis
        from football_model.services.value_analysis import ValueAnalysisService
        value_service = ValueAnalysisService()
        value_report = value_service.analyze_match(
            match_id=match_id,
            odds_home=odds["主胜"],
            odds_draw=odds["平局"],
            odds_away=odds["客胜"],
            model_home=summary.probabilities["主胜"],
            model_draw=summary.probabilities["平局"],
            model_away=summary.probabilities["客胜"],
            confidence_score=confidence,
        )

        left, right = st.columns([1.25, 1])
        with left:
            st.plotly_chart(probability_chart(comparison), width="stretch")
        with right:
            display = format_percent_columns(comparison, ["模型概率", "市场概率", "概率差", "理论EV"])
            display["公平SP"] = display["公平SP"].map(lambda value: f"{value:.2f}")
            st.dataframe(display, hide_index=True, width="stretch")

        # Value analysis summary
        section_header("价值分析", "EV和Kelly仓位参考。", f"整体风险: {value_report.overall_risk}")
        v1, v2, v3, v4 = st.columns(4)
        if value_report.best_value:
            v1.metric("最佳机会", value_report.best_value.label)
            v2.metric("EV", f"{value_report.best_value.ev:+.1%}")
            v3.metric("Kelly", f"{value_report.best_value.kelly_fraction:.1%}")
            v4.metric("建议仓位", f"{value_report.best_value.recommended_stake_pct:.1%}")
        else:
            v1.metric("最佳机会", "无")
            v2.metric("EV", "-")
            v3.metric("Kelly", "-")
            v4.metric("建议仓位", "-")

        # Value details
        for va in value_report.outcomes:
            ev_color = "#22c55e" if va.ev > 0 else "#ef4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:1rem;padding:0.3rem 0'>"
                f"<span style='font-weight:600;width:3rem'>{va.label}</span>"
                f"<span>赔率 {va.odds:.2f}</span>"
                f"<span>市场 {va.market_prob:.1%}</span>"
                f"<span>模型 {va.model_prob:.1%}</span>"
                f"<span style='color:{ev_color};font-weight:600'>EV {va.ev:+.1%}</span>"
                f"<span>Kelly {va.kelly_fraction:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if va.risk_warning:
                st.caption(f"  ⚠️ {va.risk_warning}")

        render_risk_note(value_report.disclaimer)

        st.markdown("#### 主要风险")
        if risks:
            for risk in risks:
                render_risk_note(risk)
        else:
            render_risk_note("当前未发现显著的数据完整性或模型分歧风险，但不代表赛果确定。")
        st.markdown("#### 关键因素")
        factors: list[str] = []
        if intelligence is not None:
            elo_gap = intelligence.home.elo - intelligence.away.elo
            factors.append(
                f"Elo实力差：{intelligence.home.team} {intelligence.home.elo:.0f}，"
                f"{intelligence.away.team} {intelligence.away.elo:.0f}（主队差值{elo_gap:+.0f}）。"
            )
            factors.append(
                f"近8场加权积分：主队{intelligence.home.points_per_game:.2f}，"
                f"客队{intelligence.away.points_per_game:.2f}。"
            )
        if dc_xg and form_xg:
            factors.append(
                f"长期模型预期进球{dc_xg[0]:.2f}:{dc_xg[1]:.2f}；近期状态模型{form_xg[0]:.2f}:{form_xg[1]:.2f}。"
            )
        movement = _market_movement(history)
        if not movement.empty:
            strongest = movement.iloc[movement["市场概率变化"].abs().argmax()]
            factors.append(
                f"官方市场变化最大：{strongest['结果']}概率{strongest['市场概率变化']:+.1%}，"
                f"方向为{strongest['方向']}。"
            )
        for factor in factors:
            st.markdown(f"- {factor}")

    with form_tab:
        if intelligence is None:
            empty_state("未匹配球队历史状态", "当前比赛没有可用的 Elo、近期战绩或历史交锋数据。", "🧠")
        else:
            st.dataframe(_team_table(intelligence), hide_index=True, width="stretch")
            e1, e2, e3 = st.columns(3)
            e1.metric("Elo倾向主胜", f"{intelligence.elo_home_probability:.1%}")
            e2.metric("Elo倾向平局", f"{intelligence.elo_draw_probability:.1%}")
            e3.metric("Elo倾向客胜", f"{intelligence.elo_away_probability:.1%}")
            st.caption(f"状态数据截止 {intelligence.data_cutoff:%Y-%m-%d}；世界杯按中立场处理。")
            if training is not None and mapped_home and mapped_away:
                home_recent, away_recent, h2h = st.tabs(
                    [f"{_series_get(selected, 'home_team', '主队')}近期", f"{_series_get(selected, 'away_team', '客队')}近期", "历史交锋"]
                )
                with home_recent:
                    st.dataframe(
                        _recent_match_rows(training, mapped_home, kickoff_at),
                        hide_index=True, width="stretch",
                    )
                with away_recent:
                    st.dataframe(
                        _recent_match_rows(training, mapped_away, kickoff_at),
                        hide_index=True, width="stretch",
                    )
                with h2h:
                    meetings = _head_to_head_rows(training, mapped_home, mapped_away, kickoff_at)
                    if meetings.empty:
                        empty_state("暂无历史交锋", "当前训练数据集中没有双方近期直接交锋。", "📭")
                    else:
                        st.dataframe(meetings, hide_index=True, width="stretch")

    with lineup_tab:
        from football_model.ui.pages.match_detail import POS_MAP, _render_formation_pitch, _render_player_table

        with database.connection(read_only=True) as connection:
            fixture_context = connection.execute(
                "SELECT provider_fixture_id, venue, status, resolved_at FROM provider_fixtures WHERE match_id=?",
                [match_id],
            ).df()
            lineups = connection.execute(
                """SELECT team_side, is_current, formation, confirmed, players_json, captured_at
                FROM lineup_snapshots WHERE match_id=?
                ORDER BY is_current DESC, captured_at DESC""",
                [match_id],
            ).df()
            injuries = connection.execute(
                """SELECT team_side, players_json, captured_at FROM injury_snapshots
                WHERE match_id=?
                QUALIFY ROW_NUMBER() OVER (PARTITION BY team_side ORDER BY captured_at DESC)=1""",
                [match_id],
            ).df()

        if fixture_context.empty:
            empty_state("尚未匹配实时供应商", "点击页面上方“更新伤停与首发”后可尝试同步阵容和伤停。", "📡")
        else:
            context = fixture_context.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("供应商状态", context["status"])
            c2.metric("比赛场馆", context["venue"] or "未知")
            c3.metric("供应商比赛ID", str(context["provider_fixture_id"]))

        # Organize lineups: current first, then previous
        current_lineups = {}
        previous_lineups = {}
        if not lineups.empty:
            for _, row in lineups.iterrows():
                side = row["team_side"]
                if row["is_current"] and side not in current_lineups:
                    current_lineups[side] = row
                elif not row["is_current"] and side not in previous_lineups:
                    previous_lineups[side] = row

        # Display lineups with formation view
        for side, label in [("home", "主队"), ("away", "客队")]:
            if side in current_lineups:
                row = current_lineups[side]
                players = json.loads(row["players_json"])
                _render_formation_pitch(
                    team_name=label,
                    formation=row["formation"] or "4-3-3",
                    players=players,
                    is_confirmed=row["confirmed"],
                    source_label="本场首发",
                )
                with st.expander("📋 球员名单", expanded=False):
                    _render_player_table(players, is_confirmed=True)
            elif side in previous_lineups:
                row = previous_lineups[side]
                players = json.loads(row["players_json"])
                _render_formation_pitch(
                    team_name=label,
                    formation=row["formation"] or "4-3-3",
                    players=players,
                    is_confirmed=False,
                    source_label="上一场首发（本场待更新）",
                )
                with st.expander("📋 球员名单", expanded=False):
                    _render_player_table(players, is_confirmed=False)
            else:
                empty_state(f"{label}首发尚未发布", "通常在开赛前20–40分钟可用；未确认阵容会降低报告可信度。", "👥")

        # Injuries
        st.markdown("#### 伤停名单")
        if injuries.empty:
            empty_state("暂无伤停快照", "供应商尚未返回该场比赛的伤停数据。", "🩺")
        else:
            for side, label in [("home", "主队"), ("away", "客队")]:
                row = injuries.loc[injuries["team_side"] == side]
                if not row.empty:
                    players = json.loads(row.iloc[0]["players_json"])
                    st.markdown(f"**{label}**")
                    if not players:
                        st.caption("供应商未报告伤停")
                    else:
                        for p in players:
                            name = p.get("name", "未知")
                            pos = p.get("position", "")
                            reason = p.get("injury_type", p.get("reason", "未知"))
                            pos_cn = POS_MAP.get(pos.upper(), pos) if pos else ""
                            st.markdown(f"- **{name}** {pos_cn} · {reason}")

    with model_tab:
        components = []
        component_labels = {"market": "官方市场先验", "dixon_coles": "Dixon–Coles", "poisson": "Poisson状态特征"}
        if production_prediction is not None:
            for name, probabilities in production_prediction.components.items():
                components.append({
                    "模型": component_labels.get(name, name),
                    "主胜": probabilities["home_win"],
                    "平局": probabilities["draw"],
                    "客胜": probabilities["away_win"],
                })
        else:
            components.append({"模型": "官方市场先验", **_probabilities(market_matrix)})
            if dc_matrix is not None:
                components.append({"模型": "Dixon–Coles长期强度", **_probabilities(dc_matrix)})
            if form_matrix is not None:
                components.append({"模型": "Elo + 近8场状态", **_probabilities(form_matrix)})
        components.append({"模型": "最终组合", **summary.probabilities})
        component_frame = pd.DataFrame(components)
        chart_data = component_frame.melt(id_vars="模型", var_name="赛果", value_name="概率")
        figure = px.bar(chart_data, x="模型", y="概率", color="赛果", barmode="group", text_auto=".1%")
        figure.update_layout(yaxis_tickformat=".0%", height=350)
        figure = plotly_theme(figure)
        st.plotly_chart(figure, width="stretch")
        st.dataframe(
            format_percent_columns(component_frame, ["主胜", "平局", "客胜"]), hide_index=True, width="stretch"
        )
        render_risk_note("长期强度、近期状态和市场先验分别计算后再组合；市场赔率不参与独立模型训练。")

    with score_tab:
        score_col, totals_col = st.columns([1.35, 1])
        with score_col:
            figure = score_heatmap(ensemble_matrix, max_goals=7)
            st.plotly_chart(figure, width="stretch")
        with totals_col:
            top_scores = pd.DataFrame(summary.top_scores[:10], columns=["比分", "概率"])
            st.dataframe(format_percent_columns(top_scores, ["概率"]), hide_index=True, width="stretch")
            over_25 = sum(
                ensemble_matrix[h, a]
                for h in range(ensemble_matrix.shape[0])
                for a in range(ensemble_matrix.shape[1])
                if h + a >= 3
            )
            btts = sum(
                ensemble_matrix[h, a]
                for h in range(1, ensemble_matrix.shape[0])
                for a in range(1, ensemble_matrix.shape[1])
            )
            metric_card("大于2.5球", format_percent(over_25, 1), variant="info")
            metric_card("双方进球", format_percent(btts, 1), variant="purple")

    with market_tab:
        snapshot_count = history["captured_at"].nunique() if not history.empty else 0
        if snapshot_count < 3:
            render_risk_note("盘口快照仍少；系统每60秒保存变化，样本增加后趋势判断才有意义。")
        movement = _market_movement(history)
        if not movement.empty:
            st.markdown("#### 去水后的市场概率变化")
            movement_display = movement.copy()
            movement_display["市场概率变化"] = movement_display["市场概率变化"].map(lambda value: f"{value:+.2%}")
            movement_display["SP变化"] = movement_display["SP变化"].map(lambda value: f"{value:+.2f}")
            movement_display["SP波动率"] = movement_display["SP波动率"].map(lambda value: f"{value:.2%}")
            st.dataframe(movement_display, hide_index=True, width="stretch")
        if not history.empty:
            labels = {"H": "主胜", "D": "平局", "A": "客胜"}
            history["选项"] = history["selection"].map(labels)
            for market, title in [("HAD", "官方胜平负SP变化"), ("HHAD", "官方让球胜平负SP变化")]:
                market_history = history.loc[history["market"] == market].copy()
                if market_history.empty:
                    continue
                if market == "HHAD":
                    market_history["选项"] = market_history.apply(
                        lambda row: (
                            f"{row['选项']} ({row['goal_line']})" if pd.notna(row["goal_line"]) else row["选项"]
                        ),
                        axis=1,
                    )
                figure = px.line(market_history, x="captured_at", y="odds", color="选项", markers=True, title=title)
                figure.update_layout(height=300)
                figure = plotly_theme(figure)
                st.plotly_chart(figure, width="stretch")
        st.dataframe(history, hide_index=True, width="stretch")

    with data_tab:
        training_cutoff = model.training_cutoff if model is not None else "不可用"
        details = pd.DataFrame(
            [
                ["模型赛事", competition or "无"],
                ["模型版本", model_version],
                ["训练比赛", model_metrics.get("matches", "不可用")],
                ["训练球队", model_metrics.get("teams", "不可用")],
                ["训练截止", training_cutoff],
                ["时间留出样本", model_metrics.get("holdout_matches", "待重新训练")],
                ["留出集Log Loss", f"{model_metrics['holdout_log_loss']:.3f}" if "holdout_log_loss" in model_metrics else "待重新训练"],
                ["留出集Brier", f"{model_metrics['holdout_brier']:.3f}" if "holdout_brier" in model_metrics else "待重新训练"],
                ["留出集赛果命中率", f"{model_metrics['holdout_accuracy']:.1%}" if "holdout_accuracy" in model_metrics else "待重新训练"],
                ["球队映射", f"{mapped_home or '-'} vs {mapped_away or '-'}"],
                ["市场先验xG", f"{market_xg[0]:.2f} / {market_xg[1]:.2f}"],
                ["长期强度xG", f"{dc_xg[0]:.2f} / {dc_xg[1]:.2f}" if dc_xg else "不可用"],
                ["状态模型xG", f"{form_xg[0]:.2f} / {form_xg[1]:.2f}" if form_xg else "不可用"],
                ["生产集成xG", f"{ensemble_xg[0]:.2f} / {ensemble_xg[1]:.2f}"],
                ["生产权重", component_text],
                ["官方数据更新时间", str(_series_get(selected, "last_update", "未知"))],
                ["盘口快照数", snapshot_count],
            ],
            columns=["项目", "值"],
        )
        details["值"] = details["值"].astype(str)
        st.dataframe(details, hide_index=True, width="stretch")
        render_risk_note("当前仍未接入亚洲盘、大小球和精确场馆天气。这些字段缺失时，系统不会把可信度标成满分。")

    # ── 上场复盘 Tab ──
    with prev_tab:
        _render_previous_match_tab(database, mapped_home, mapped_away, competition, selected)

    # ── 技战术解读 Tab ──
    with tactical_tab:
        _render_tactical_tab(database, match_id, str(_series_get(selected, "home_team", "")), str(_series_get(selected, "away_team", "")), competition, kickoff_at)


def _render_previous_match_tab(
    database: LocalDatabase,
    mapped_home: str | None,
    mapped_away: str | None,
    competition: str | None,
    selected: pd.Series,
) -> None:
    """Render previous match analysis tab."""
    from football_model.services.previous_match import PreviousMatchService
    from football_model.ui.components import safe_html

    if not competition or not mapped_home or not mapped_away:
        empty_state("数据不足", "球队映射未完成，无法分析上场比赛。", "📊")
        return

    service = PreviousMatchService(database)
    kickoff_at = pd.to_datetime(_series_get(selected, "kickoff", pd.Timestamp.now()))

    home_prev, away_prev = service.get_both_previous_matches(mapped_home, mapped_away, competition, kickoff_at)

    col1, col2 = st.columns(2)

    with col1:
        section_header(f"{safe_html(_series_get(selected, 'home_team', '主队'))}", "最近一场比赛表现。")
        if home_prev:
            result_color = "#22c55e" if home_prev.result == "W" else "#eab308" if home_prev.result == "D" else "#ef4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem'>"
                f"<span style='background:{result_color};color:#fff;padding:0.2rem 0.6rem;border-radius:4px;font-weight:700;font-size:1.1rem'>{home_prev.result}</span>"
                f"<span style='font-size:1rem;font-weight:600'>{safe_html(home_prev.venue)} {safe_html(home_prev.opponent)}</span>"
                f"<span style='font-size:1.2rem;font-weight:800'>{safe_html(home_prev.score)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("日期", home_prev.match_date)
            m2.metric("天数", f"{home_prev.days_ago}天前")
            m3.metric("体能", home_prev.fatigue_level)

            st.markdown("**比赛特征**")
            features = []
            if home_prev.is_clean_sheet:
                features.append("✅ 零封对手")
            if home_prev.is_btts:
                features.append("⚽ 双方进球")
            if home_prev.is_over25:
                features.append("📈 大2.5球")
            for f in features:
                st.caption(f)

            st.markdown("**对本场影响**")
            st.caption(home_prev.impact_on_next)
        else:
            empty_state("暂无上场比赛数据", "等待历史数据同步。", "📭")

    with col2:
        section_header(f"{safe_html(_series_get(selected, 'away_team', '客队'))}", "最近一场比赛表现。")
        if away_prev:
            result_color = "#22c55e" if away_prev.result == "W" else "#eab308" if away_prev.result == "D" else "#ef4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem'>"
                f"<span style='background:{result_color};color:#fff;padding:0.2rem 0.6rem;border-radius:4px;font-weight:700;font-size:1.1rem'>{away_prev.result}</span>"
                f"<span style='font-size:1rem;font-weight:600'>{safe_html(away_prev.venue)} {safe_html(away_prev.opponent)}</span>"
                f"<span style='font-size:1.2rem;font-weight:800'>{safe_html(away_prev.score)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("日期", away_prev.match_date)
            m2.metric("天数", f"{away_prev.days_ago}天前")
            m3.metric("体能", away_prev.fatigue_level)

            st.markdown("**比赛特征**")
            features = []
            if away_prev.is_clean_sheet:
                features.append("✅ 零封对手")
            if away_prev.is_btts:
                features.append("⚽ 双方进球")
            if away_prev.is_over25:
                features.append("📈 大2.5球")
            for f in features:
                st.caption(f)

            st.markdown("**对本场影响**")
            st.caption(away_prev.impact_on_next)
        else:
            empty_state("暂无上场比赛数据", "等待历史数据同步。", "📭")


def _render_tactical_tab(
    database: LocalDatabase,
    match_id: str,
    home_team: str,
    away_team: str,
    competition: str | None,
    kickoff: object,
) -> None:
    """Render tactical analysis tab."""
    from football_model.services.tactical_analysis import TacticalAnalysisService
    from football_model.ui.components import safe_html

    if not competition:
        empty_state("数据不足", "赛事信息不完整，无法生成技战术分析。", "📋")
        return

    service = TacticalAnalysisService(database)

    try:
        report = service.generate_analysis(match_id, home_team, away_team, competition, kickoff)
    except Exception as e:
        logger.warning("Tactical analysis failed: %s", e)
        empty_state("技战术分析数据不足", "等待更多比赛数据积累。", "📋")
        return

    # Formation comparison
    section_header("阵型对比", "双方预计阵型和上一场阵型。")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"**{safe_html(home_team)}**")
        st.caption(f"预计阵型: {safe_html(report.home_formation)}")
    with f2:
        st.markdown(f"**{safe_html(away_team)}**")
        st.caption(f"预计阵型: {safe_html(report.away_formation)}")

    # Style comparison
    section_header("战术风格", "双方进攻和防守方式。")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{safe_html(home_team)}**")
        st.caption(f"进攻风格: {safe_html(report.home_attack_style)}")
        st.caption(f"防守风格: {safe_html(report.home_defense_style)}")
        st.caption(f"压迫强度: {safe_html(report.home_pressing)}")
    with s2:
        st.markdown(f"**{safe_html(away_team)}**")
        st.caption(f"进攻风格: {safe_html(report.away_attack_style)}")
        st.caption(f"防守风格: {safe_html(report.away_defense_style)}")
        st.caption(f"压迫强度: {safe_html(report.away_pressing)}")

    # Strengths
    section_header("强度指标", "反击、边路、中场、定位球能力。")
    str_data = pd.DataFrame([
        ["反击能力", f"{report.home_counter_attack:.0%}", f"{report.away_counter_attack:.0%}"],
        ["边路强度", f"{report.home_wing_strength:.0%}", f"{report.away_wing_strength:.0%}"],
        ["中场控制", f"{report.home_midfield_control:.0%}", f"{report.away_midfield_control:.0%}"],
        ["定位球威胁", f"{report.home_set_piece_threat:.0%}", f"{report.away_set_piece_threat:.0%}"],
    ], columns=["指标", home_team, away_team])
    st.dataframe(str_data, hide_index=True, use_container_width=True)

    # Weaknesses
    section_header("弱点分析", "双方防守薄弱环节。")
    w1, w2 = st.columns(2)
    with w1:
        st.caption(f"**{safe_html(home_team)}**: {safe_html(report.home_defensive_weakness)}")
    with w2:
        st.caption(f"**{safe_html(away_team)}**: {safe_html(report.away_defensive_weakness)}")

    # Key matchups
    section_header("关键对位", "影响比赛走向的关键因素。")
    for matchup in report.key_matchups:
        st.caption(f"- {safe_html(matchup)}")

    # Tactical advantage
    section_header("战术优势", "综合评估。")
    st.caption(f"**战术优势**: {safe_html(report.tactical_advantage)}")
    st.caption(f"**预期变化**: {safe_html(report.expected_changes)}")
    st.caption(f"**概率影响**: {safe_html(report.probability_impact)}")

    render_risk_note("技战术分析基于历史数据推断，实际比赛可能因临场调整、意外事件等因素产生偏差。")
