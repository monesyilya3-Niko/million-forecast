from __future__ import annotations

import logging
import os as _os
from typing import Any

import pandas as pd
import streamlit as st

from football_model.services import SportteryLiveService
from football_model.ui.components import (
    empty_state,
    format_number,
    hero_pro,
    metric_card,
    render_badge,
    render_risk_note,
    render_status_dot,
    safe_html,
    section_header,
)

logger = logging.getLogger(__name__)

MODEL_LEAGUES = {"世界杯", "世界杯国家队", "瑞典超级联赛", "英格兰超级联赛"}


def _open_analysis(match_id: str) -> None:
    """Navigate to match analysis."""
    from football_model.ui.navigation import navigate_to
    navigate_to("🔍 比赛分析", context={"analysis_match_id": match_id})


def _row_get(row: Any, field: str, default: Any = None) -> Any:
    if hasattr(row, "_asdict"):
        return row._asdict().get(field, default)
    if hasattr(row, "get"):
        return row.get(field, default)
    return getattr(row, field, default)


def _has_column(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns


def _safe_series(frame: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _is_value(value: Any) -> bool:
    try:
        return value is not None and pd.notna(value)
    except (TypeError, ValueError):
        return value is not None


def _fmt_odds(value: Any) -> str:
    return format_number(value, digits=2, default="--")


def _fmt_time(value: Any) -> str:
    if not _is_value(value):
        return "--"
    try:
        return f"{pd.to_datetime(value):%m-%d %H:%M}"
    except (TypeError, ValueError):
        return str(value)


def _sell_status_text(value: Any) -> tuple[str, str, str]:
    status = str(value) if _is_value(value) else ""
    if status == "1":
        return "销售中", "success", "live"
    if status == "2":
        return "已暂停", "warning", "warning"
    if status in {"0", "3"}:
        return "未开售", "neutral", "neutral"
    return "状态未知", "neutral", "neutral"


def _is_model_supported(league: Any) -> bool:
    return str(league) in MODEL_LEAGUES


def render_live_matches(live_service: SportteryLiveService) -> None:
    hero_pro(
        title="今日竞彩 / Today's Matches",
        subtitle="官方赛程 · 实时赔率 · 模型分析 / Official Schedule · Live Odds · Model Analysis",
        eyebrow="LIVE MATCH CENTER",
        meta=["本地运行 / Local", "自动刷新 / Auto Refresh 30s", "数据源 / Source: 竞彩"],
    )
    _live_panel(live_service)


@st.fragment(run_every="30s")
def _live_panel(live_service: SportteryLiveService) -> None:
    refresh_error = None
    refresh = None

    # HuggingFace 等海外环境：跳过 API 刷新，直接用缓存
    _is_cloud = bool(_os.environ.get("SPACE_ID") or _os.environ.get("DOCKER_CONTAINER"))

    if not _is_cloud:
        # 本地环境：自动重试机制
        for attempt in range(3):
            try:
                refresh = live_service.refresh()
                break
            except Exception as error:
                refresh_error = str(error)
                if attempt < 2:
                    import time
                    time.sleep(1)

    if refresh is None:
        logger.warning("Live service refresh failed after 3 attempts: %s", refresh_error)

    dates = live_service.dates()
    if not dates:
        empty_state("暂无竞彩赛程", "请检查本地数据源或稍后刷新。", "⚽")
        if refresh_error:
            render_risk_note("接口暂不可用，当前页面无法同步最新竞彩赛程；请以官方数据源为准。")
        return

    today = live_service.today_string()
    default_index = dates.index(today) if today in dates else max(0, len(dates) - 1)

    # 实时状态指示器
    if refresh:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">'
            f'<span class="status-dot live"></span>'
            f'<span style="font-size:0.78rem;color:#22c55e">实时同步中 · {safe_html(str(refresh.last_update))} · {refresh.total_count}场</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">'
            '<span class="status-dot warning"></span>'
            '<span style="font-size:0.78rem;color:#eab308">使用缓存数据 · 接口暂不可用</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    section_header("筛选条件", "按日期、联赛和销售状态过滤实时赛事。", "30s 自动刷新")
    with st.container():
        st.markdown('<div class="card anim-fade">', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([1.45, 1.45, 1])
        business_date = f1.selectbox("日期", dates, index=default_index, key="live-business-date")
        matches = live_service.matches_for_date(business_date)

        if matches.empty:
            leagues = ["全部"]
        elif _has_column(matches, "league_name"):
            leagues = ["全部"] + sorted(matches["league_name"].dropna().astype(str).unique().tolist())
        else:
            leagues = ["全部"]

        selected_league = f2.selectbox("联赛", leagues, key="live-league")
        only_selling = f3.toggle("仅销售中", value=False, key="live-only-selling")
        st.markdown("</div>", unsafe_allow_html=True)

    if matches.empty:
        empty_state("该日期暂无比赛", "换一个日期，或刷新本地竞彩数据源。", "📭")
        return

    base_matches = matches.copy()
    if selected_league != "全部" and _has_column(matches, "league_name"):
        matches = matches.loc[matches["league_name"].astype(str) == selected_league]
    if only_selling and _has_column(matches, "sell_status"):
        matches = matches.loc[matches["sell_status"].astype(str) == "1"]

    sell_series = _safe_series(base_matches, "sell_status", "")
    single_series = _safe_series(base_matches, "had_single", False)
    league_series = _safe_series(base_matches, "league_name", "")
    last_update = getattr(refresh, "last_update", None) if refresh else None
    total_count = getattr(refresh, "total_count", None) if refresh else len(base_matches)

    section_header("实时总览", "赛事数量、销售状态和模型覆盖概览。")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("总场次", str(total_count or len(base_matches)), variant="info")
    with m2:
        metric_card("销售中", str(int((sell_series.astype(str) == "1").sum())), variant="positive")
    with m3:
        metric_card("单关", str(int(single_series.fillna(False).astype(bool).sum())), variant="purple")
    with m4:
        metric_card("有模型", str(int(league_series.apply(_is_model_supported).sum())), variant="info")
    with m5:
        metric_card("最近同步", safe_html(last_update) or "--", caption="接口不可用时显示本地缓存", variant="warning" if refresh_error else "neutral")

    if refresh_error:
        render_risk_note("竞彩接口当前不可用，页面展示本地缓存数据。赔率和销售状态可能滞后，请以官方信息为准。")

    section_header("赛事列表", "点击详情查看阵容/伤停，点击分析进入模型报告。", f"当前筛选 {len(matches)} 场")
    if matches.empty:
        empty_state("当前筛选条件下无比赛", "取消“仅销售中”或切换联赛后再查看。", "🔎")
        return

    for match in matches.itertuples(index=False):
        match_id = str(_row_get(match, "match_id", f"row-{getattr(match, 'Index', '')}"))
        league = _row_get(match, "league_name", "未知联赛")
        home_team = _row_get(match, "home_team", "主队")
        away_team = _row_get(match, "away_team", "客队")
        kickoff = _fmt_time(_row_get(match, "kickoff"))
        match_number = _row_get(match, "match_number", "")
        sell_status, sell_variant, dot_status = _sell_status_text(_row_get(match, "sell_status"))
        had_single = bool(_row_get(match, "had_single", False)) if _is_value(_row_get(match, "had_single", False)) else False
        model_supported = _is_model_supported(league)
        goal_line = _row_get(match, "goal_line", "--")

        badges = [render_badge(sell_status, sell_variant)]
        if had_single:
            badges.append(render_badge("单关", "purple"))
        if model_supported:
            badges.append(render_badge("有模型", "info"))

        st.markdown(
            f"""
<div class="match-card anim-fade">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
    <div>
      <div class="match-title">{safe_html(home_team)} <span style="color:var(--text-faint);font-weight:600">vs</span> {safe_html(away_team)}</div>
      <div class="match-meta">{render_status_dot(dot_status)}{safe_html(match_number)} · {safe_html(league)} · 开赛 {safe_html(kickoff)}</div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:flex-end">{''.join(badges)}</div>
  </div>
  <div class="odds-grid">
    <div class="odds-box"><div class="odds-label">主胜</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'had_h')))}</div></div>
    <div class="odds-box"><div class="odds-label">平局</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'had_d')))}</div></div>
    <div class="odds-box"><div class="odds-label">客胜</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'had_a')))}</div></div>
  </div>
  <div class="odds-grid">
    <div class="odds-box"><div class="odds-label">让主 {safe_html(goal_line or '--')}</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'hhad_h')))}</div></div>
    <div class="odds-box"><div class="odds-label">让平</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'hhad_d')))}</div></div>
    <div class="odds-box"><div class="odds-label">让客</div><div class="odds-value">{safe_html(_fmt_odds(_row_get(match, 'hhad_a')))}</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        b1, b2, _ = st.columns([1, 1, 5])
        with b1:
            if st.button("📋 详情", key=f"live-detail-{match_id}", type="secondary"):
                from football_model.ui.navigation import navigate_to
                navigate_to("📄 比赛详情", context={"detail_match_id": match_id})
        with b2:
            analysis_disabled = not all(_is_value(_row_get(match, col)) for col in ["had_h", "had_d", "had_a"])
            if st.button("🔍 分析", key=f"live-analysis-{match_id}", disabled=analysis_disabled):
                _open_analysis(match_id)
                st.rerun(scope="app")
