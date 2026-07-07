"""Parlay / combination analysis page.

Allows users to combine multiple match predictions into parlay bets
and calculate expected value, risk, and optimal stake sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from football_model.core import competition_for_league
from football_model.data import LocalDatabase
from football_model.services import EnsembleAnalysisService, SportteryLiveService
from football_model.ui.components import hero_pro, render_risk_note

logger = logging.getLogger(__name__)


@dataclass
class ParlayLeg:
    match_id: str
    match_label: str
    selection: str  # "主胜", "平局", "客胜"
    odds: float
    model_prob: float
    market_prob: float


@dataclass
class ParlayResult:
    legs: list[ParlayLeg]
    combined_odds: float
    combined_prob: float
    combined_market_prob: float
    ev: float
    kelly_fraction: float
    confidence: str


def _kelly_fraction(prob: float, odds: float) -> float:
    """Calculate Kelly criterion fraction."""
    if odds <= 1.0 or prob <= 0:
        return 0.0
    edge = prob * odds - 1
    if edge <= 0:
        return 0.0
    return max(0.0, edge / (odds - 1))


def render_parlay(
    database: LocalDatabase,
    live_service: SportteryLiveService,
    ensemble_service: EnsembleAnalysisService,
) -> None:
    hero_pro("过关分析", "组合多场比赛，计算期望价值和最优仓位。", "PARLAY BUILDER", ["EV", "Kelly", "风险控制"])

    dates = live_service.dates()
    if not dates:
        st.info("暂无比赛数据")
        return

    today = live_service.today_string()
    business_date = st.selectbox("选择日期", dates, index=dates.index(today) if today in dates else 0)
    matches = live_service.matches_for_date(business_date)
    if "sell_status" in matches.columns:
        matches = matches[matches["sell_status"].fillna("").astype(str) == "1"]
    else:
        st.info("缺少销售状态字段，无法筛选")
        return

    if matches.empty:
        st.info("无销售中的比赛")
        return

    # Session state for parlay legs
    if "parlay_legs" not in st.session_state:
        st.session_state["parlay_legs"] = []

    # Match selector
    st.markdown("#### 添加比赛")
    match_options = []
    for _, m in matches.iterrows():
        h = m.get("had_h")
        d = m.get("had_d")
        a = m.get("had_a")
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            match_options.append({
                "id": str(m["match_id"]),
                "label": f"{m['home_team']} vs {m['away_team']} ({m['league_name']})",
                "home": str(m["home_team"]),
                "away": str(m["away_team"]),
                "league": str(m["league_name"]),
                "had_h": float(h),
                "had_d": float(d),
                "had_a": float(a),
            })

    if not match_options:
        st.info("无可分析的比赛")
        return

    col_match, col_selection = st.columns([2, 1])
    with col_match:
        selected_idx = st.selectbox(
            "选择比赛",
            range(len(match_options)),
            format_func=lambda i: match_options[i]["label"],
            key="parlay_match_select",
        )
    with col_selection:
        selected_match = match_options[selected_idx]
        sel_options = {
            "主胜": selected_match["had_h"],
            "平局": selected_match["had_d"],
            "客胜": selected_match["had_a"],
        }
        selection = st.selectbox("选择赛果", list(sel_options.keys()), key="parlay_selection")

    if st.button("➕ 添加到过关", type="primary", use_container_width=True):
        match = match_options[selected_idx]
        odds = sel_options[selection]

        # Get model prediction
        model_prob = 1.0 / odds / sum(1.0 / v for v in [match["had_h"], match["had_d"], match["had_a"]])
        competition = competition_for_league(match["league"])
        if competition:
            pred = ensemble_service.predict(
                home_team=match["home"], away_team=match["away"],
                competition=competition, league_name=match["league"],
                odds_home=match["had_h"], odds_draw=match["had_d"], odds_away=match["had_a"],
                kickoff=pd.Timestamp.now(),
            )
            if pred is not None:
                probs = {"主胜": pred.home_win, "平局": pred.draw, "客胜": pred.away_win}
                model_prob = probs[selection]

        market_prob = 1.0 / odds / sum(1.0 / v for v in [match["had_h"], match["had_d"], match["had_a"]])

        leg = ParlayLeg(
            match_id=match["id"],
            match_label=match["label"],
            selection=selection,
            odds=odds,
            model_prob=model_prob,
            market_prob=market_prob,
        )
        # Check for duplicate
        existing_ids = {(leg.match_id, leg.selection) for leg in st.session_state["parlay_legs"]}
        if (match["id"], selection) in existing_ids:
            st.warning("该比赛该赛果已在过关中")
        else:
            st.session_state["parlay_legs"].append(leg)
            st.rerun()

    # Display current legs
    legs = st.session_state["parlay_legs"]
    if legs:
        st.divider()
        st.markdown(f"#### 当前过关（{len(legs)} 场）")

        for i, leg in enumerate(legs):
            col_info, col_remove = st.columns([5, 1])
            with col_info:
                ev_color = "#22c55e" if leg.model_prob * leg.odds - 1 > 0 else "#ef4444"
                st.markdown(
                    f"""
                <div class="match-card" style="margin-bottom:0.4rem;padding:0.8rem 1rem">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="color:#f0f6fc;font-weight:600">{leg.match_label}</span>
                            <span style="color:#8b949e;margin-left:0.5rem">·</span>
                            <span style="color:#3b82f6;font-weight:700;margin-left:0.5rem">{leg.selection}</span>
                        </div>
                        <div style="text-align:right">
                            <span style="color:#f0f6fc;font-weight:700">赔率 {leg.odds:.2f}</span>
                            <span style="color:#8b949e;margin-left:0.8rem">模型 {leg.model_prob:.1%}</span>
                            <span style="color:{ev_color};margin-left:0.8rem;font-weight:600">EV {leg.model_prob * leg.odds - 1:+.1%}</span>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col_remove:
                if st.button("❌", key=f"remove-{i}", use_container_width=True):
                    st.session_state["parlay_legs"].pop(i)
                    st.rerun()

        if len(legs) >= 2:
            # Calculate parlay result
            combined_odds = 1.0
            combined_prob = 1.0
            combined_market = 1.0
            for leg in legs:
                combined_odds *= leg.odds
                combined_prob *= leg.model_prob
                combined_market *= leg.market_prob

            ev = combined_prob * combined_odds - 1
            kelly = _kelly_fraction(combined_prob, combined_odds)

            if ev > 0.10 and all(leg.model_prob * leg.odds - 1 > 0.02 for leg in legs):
                confidence = "高"
            elif ev > 0.02:
                confidence = "中"
            else:
                confidence = "低"

            st.divider()
            st.markdown("#### 过关汇总")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("组合赔率", f"{combined_odds:.2f}")
            r2.metric("组合概率", f"{combined_prob:.1%}")
            r3.metric("期望价值", f"{ev:+.1%}", delta_color="normal" if ev > 0 else "inverse")
            r4.metric("Kelly仓位", f"{kelly:.1%}")

            conf_variant = "success" if confidence == "高" else "warning" if confidence == "中" else "danger"
            st.markdown(
                f'<span class="badge badge-{conf_variant}">置信度：{confidence}</span>',
                unsafe_allow_html=True,
            )

            # Risk analysis
            st.markdown("#### 风险分析")
            risk_col1, risk_col2 = st.columns(2)
            with risk_col1:
                st.markdown("**单项EV分析**")
                for leg in legs:
                    leg_ev = leg.model_prob * leg.odds - 1
                    color = "#22c55e" if leg_ev > 0 else "#ef4444"
                    st.markdown(
                        f"- {leg.selection}：EV <span style='color:{color};font-weight:600'>{leg_ev:+.1%}</span>",
                        unsafe_allow_html=True,
                    )
            with risk_col2:
                st.markdown("**破产概率估算**")
                lose_prob = 1 - combined_prob
                st.markdown(f"- 单次命中率：{combined_prob:.1%}")
                st.markdown(f"- 单次亏损概率：{lose_prob:.1%}")
                streak_5 = lose_prob ** 5
                st.markdown(f"- 连亏5次概率：{streak_5:.1%}")

            # Correlation risk
            st.markdown("#### 相关性风险")
            leagues = [leg.match_label.split("(")[-1].rstrip(")") if "(" in leg.match_label else "未知" for leg in legs]
            unique_leagues = set(leagues)

            if len(unique_leagues) == 1 and len(leagues) > 1:
                render_risk_note(f"⚠️ 同联赛风险：{len(legs)}场比赛均来自同一联赛，相关性较高，风险叠加。")
            elif len(legs) >= 4:
                render_risk_note("⚠️ 过关场次过多：4场以上过关风险极高，爆冷概率大幅上升。")
            elif len(legs) >= 3:
                render_risk_note("⚠️ 3场过关风险较高，建议控制仓位。")

            # Max recommended stake
            max_stake = min(kelly * 0.25, 0.03)  # Quarter Kelly or 3% max
            st.markdown(f"**建议最大仓位**: {max_stake:.1%}（Quarter Kelly或3%取较小值）")

            if st.button("🗑️ 清空过关", key="clear_parlay"):
                st.session_state["parlay_legs"] = []
                st.rerun()

            render_risk_note(
                "过关分析仅供参考。组合赔率越高，命中率越低。Kelly仓位为理论最优，实际使用建议减半或更小。"
                "任何模型都无法保证盈利，请理性投注。"
            )
        elif len(legs) == 1:
            st.info("请添加至少2场比赛以计算过关。")
