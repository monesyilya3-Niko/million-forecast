"""Value radar page with enterprise-grade UI."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from football_model.core import competition_for_league
from football_model.data import LocalDatabase
from football_model.services import EnsembleAnalysisService, SportteryLiveService
from football_model.ui.components import (
    empty_state,
    format_number,
    format_percent,
    hero_pro,
    metric_card,
    render_badge,
    render_risk_note,
    safe_html,
    section_header,
)

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _kelly_fraction(probability: float, odds: float) -> float:
    """Return the theoretical Kelly fraction for decimal odds."""
    probability = _safe_float(probability)
    odds = _safe_float(odds)
    if odds <= 1.0:
        return 0.0
    fraction = (odds * probability - 1.0) / (odds - 1.0)
    return max(0.0, fraction)


def _confidence_score(best_ev: float, edge: float, model_available: bool) -> tuple[str, int]:
    if model_available and best_ev > 0.10 and edge > 0.05:
        return "高", 82
    if best_ev > 0.05 or edge > 0.03:
        return "中", 62
    return "低", 38


def _confidence_variant(label: str) -> str:
    return "success" if label == "高" else "warning" if label == "中" else "danger"


def render_recommendations(
    database: LocalDatabase,
    live_service: SportteryLiveService,
    ensemble_service: EnsembleAnalysisService,
) -> None:
    hero_pro(
        title="价值发现",
        subtitle="模型与市场分歧分析 · 高亮潜在机会",
        eyebrow="VALUE RADAR",
        meta=["EV 扫描", "Kelly 理论仓位", "风险优先"],
    )

    dates = live_service.dates()
    if not dates:
        empty_state("暂无比赛数据", "请先在今日竞彩页面刷新本地赛程。", "📡")
        render_risk_note()
        return

    section_header("扫描范围", "仅扫描销售中的胜平负赔率，结果按 EV 降序排列。")
    today = live_service.today_string()
    business_date = st.selectbox(
        "日期",
        dates,
        index=dates.index(today) if today in dates else 0,
        key="value-radar-date",
    )
    matches = live_service.matches_for_date(business_date)

    if matches.empty:
        empty_state("该日期无比赛", "请选择其他日期或刷新赛程数据。", "📭")
        render_risk_note()
        return

    if "sell_status" in matches.columns:
        matches = matches[matches["sell_status"].astype(str) == "1"].copy()
    if matches.empty:
        empty_state("无销售中的比赛", "当前日期没有可扫描的销售中赛事。", "🟡")
        render_risk_note()
        return

    results: list[dict[str, Any]] = []
    scanned_count = 0
    for _, match in matches.iterrows():
        if any(pd.isna(match.get(column)) for column in ["had_h", "had_d", "had_a"]):
            continue

        scanned_count += 1
        home = str(match.get("home_team", "主队"))
        away = str(match.get("away_team", "客队"))
        league = str(match.get("league_name", "未知联赛"))
        odds_h = _safe_float(match.get("had_h"))
        odds_d = _safe_float(match.get("had_d"))
        odds_a = _safe_float(match.get("had_a"))
        if min(odds_h, odds_d, odds_a) <= 1.0:
            continue

        raw = {"home_win": 1 / odds_h, "draw": 1 / odds_d, "away_win": 1 / odds_a}
        raw_total = sum(raw.values())
        market = {key: value / raw_total for key, value in raw.items()}

        model = market.copy()
        model_source = "仅市场"
        competition = competition_for_league(league)
        if competition:
            prediction = ensemble_service.predict(
                home_team=home,
                away_team=away,
                competition=competition,
                league_name=league,
                odds_home=odds_h,
                odds_draw=odds_d,
                odds_away=odds_a,
                kickoff=pd.to_datetime(match.get("kickoff")),
                match_id=str(match.get("match_id", "")),
            )
            if prediction is not None:
                model = {"home_win": prediction.home_win, "draw": prediction.draw, "away_win": prediction.away_win}
                model_source = prediction.model_version

        model_total = sum(model.values())
        if model_total <= 0:
            continue
        model = {key: value / model_total for key, value in model.items()}

        labels = ["主胜", "平局", "客胜"]
        outcomes = ["home_win", "draw", "away_win"]
        odds_list = [odds_h, odds_d, odds_a]

        best_ev = -999.0
        best_idx = 0
        for i, outcome in enumerate(outcomes):
            ev = model[outcome] * odds_list[i] - 1
            if ev > best_ev:
                best_ev = ev
                best_idx = i

        best_outcome = outcomes[best_idx]
        edge = model[best_outcome] - market[best_outcome]
        model_available = model_source != "仅市场"
        confidence, confidence_numeric = _confidence_score(best_ev, edge, model_available)
        kelly = _kelly_fraction(model[best_outcome], odds_list[best_idx])

        results.append({
            "比赛": f"{home} vs {away}",
            "联赛": league,
            "推荐": labels[best_idx],
            "赔率": odds_list[best_idx],
            "模型概率": model[best_outcome],
            "市场概率": market[best_outcome],
            "概率差": edge,
            "EV": best_ev,
            "Kelly理论仓位": kelly,
            "模型": model_source,
            "置信度": confidence,
            "置信度分": confidence_numeric,
        })

    if not results:
        empty_state("无法生成价值信号", "当前销售中赛事缺少完整胜平负赔率或模型输入。", "🔎")
        render_risk_note()
        return

    df = pd.DataFrame(results).sort_values("EV", ascending=False).reset_index(drop=True)
    positive_count = int((df["EV"] > 0).sum())
    max_ev = float(df["EV"].max()) if not df.empty else 0.0
    model_available_rate = float((df["模型"] != "仅市场").mean()) if not df.empty else 0.0

    section_header("机会雷达", "正 EV 表示模型概率与赔率之间存在分歧，但不是确定性结论。")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("扫描场次", str(scanned_count), variant="info")
    with c2:
        metric_card("正 EV 数量", str(positive_count), variant="positive" if positive_count else "neutral")
    with c3:
        metric_card("最大 EV", format_percent(max_ev, 1), variant="positive" if max_ev > 0 else "danger")
    with c4:
        metric_card("模型可用率", format_percent(model_available_rate, 0), caption="非市场先验结果占比", variant="purple")

    section_header("重点机会", "仅展示排序靠前的观察信号；完整结果见下方表格。")
    for _, row in df.head(8).iterrows():
        ev = float(row["EV"])
        ev_variant = "success" if ev > 0.05 else "warning" if ev > 0 else "danger"
        conf_variant = _confidence_variant(str(row["置信度"]))
        kelly_warning = "Kelly 为理论仓位，不是投注指令"
        st.markdown(
            f"""
<div class="match-card anim-fade">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
    <div>
      <div class="match-title">{safe_html(row['比赛'])}</div>
      <div class="match-meta">{safe_html(row['联赛'])} · {safe_html(row['模型'])}</div>
    </div>
    <div style="display:flex;gap:0.4rem;flex-wrap:wrap;justify-content:flex-end">
      {render_badge(str(row['推荐']), 'info')}
      {render_badge(str(row['置信度']), conf_variant)}
      {render_badge('正EV' if ev > 0 else '观察', ev_variant)}
    </div>
  </div>
  <div class="odds-grid">
    <div class="odds-box"><div class="odds-label">赔率</div><div class="odds-value">{safe_html(format_number(row['赔率'], 2))}</div></div>
    <div class="odds-box"><div class="odds-label">模型概率</div><div class="odds-value">{safe_html(format_percent(row['模型概率']))}</div></div>
    <div class="odds-box"><div class="odds-label">市场概率</div><div class="odds-value">{safe_html(format_percent(row['市场概率']))}</div></div>
  </div>
  <div class="odds-grid">
    <div class="odds-box"><div class="odds-label">概率差</div><div class="odds-value">{safe_html(format_percent(row['概率差']))}</div></div>
    <div class="odds-box"><div class="odds-label">EV</div><div class="odds-value">{safe_html(format_percent(row['EV']))}</div></div>
    <div class="odds-box"><div class="odds-label">Kelly理论</div><div class="odds-value">{safe_html(format_percent(row['Kelly理论仓位']))}</div></div>
  </div>
  <div class="match-meta" style="margin-top:0.7rem;color:#fcd34d">风险提示：{safe_html(kelly_warning)}；赛前阵容和盘口变化可能改变信号强度。</div>
</div>
""",
            unsafe_allow_html=True,
        )

    section_header("完整扫描结果", "保留 st.dataframe，便于排序、复制和二次检查。")
    display_df = df.copy()
    for column in ["模型概率", "市场概率", "概率差", "EV", "Kelly理论仓位"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: format_percent(value, 1))
    if "赔率" in display_df.columns:
        display_df["赔率"] = display_df["赔率"].map(lambda value: format_number(value, 2))
    st.dataframe(display_df, hide_index=True, width="stretch")

    render_risk_note(
        "模型推荐仅供概率分析参考，不构成投注建议。Kelly 仓位只是基于当前概率和赔率的理论计算，"
        "不能作为实际投注指令；足球比赛存在临场阵容、伤停、状态和盘口波动风险。"
    )
