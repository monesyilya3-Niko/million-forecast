from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from football_model.core import get_settings  # noqa: E402
from football_model.data import LocalDatabase  # noqa: E402
from football_model.services import AnalysisService, SportteryLiveService  # noqa: E402
from football_model.services.ensemble_service import EnsembleAnalysisService  # noqa: E402
from football_model.ui.components import apply_theme, render_status_dot, safe_html  # noqa: E402
from football_model.ui.navigation import (  # noqa: E402
    get_current_page,
    navigate_to,
)
from football_model.ui.pages import (  # noqa: E402
    render_backtest,
    render_batch,
    render_data_center,
    render_live_matches,
    render_match_analysis,
    render_match_detail,
    render_model_center,
    render_parlay,
    render_recommendations,
    render_results,
    render_single_match,
    render_system_status,
)
from football_model.ui.pages.p3_analysis import render_p3_analysis  # noqa: E402
from football_model.ui.pages.p3_history import render_p3_history  # noqa: E402
from football_model.ui.pages.p3_generator import render_p3_generator  # noqa: E402
from football_model.ui.pages.dlt_analysis import render_dlt_analysis  # noqa: E402
from football_model.ui.pages.dlt_history import render_dlt_history  # noqa: E402
from football_model.ui.pages.dlt_generator import render_dlt_generator  # noqa: E402
from football_model.ui.pages.lottery_backtest import render_lottery_backtest  # noqa: E402

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="百万竞猜 · Million Forecast Terminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Services ─────────────────────────────────────────────────────
settings = get_settings(ROOT)
database = LocalDatabase(settings.database_path)
database.initialize()
analysis_service = AnalysisService()
sporttery_live_service = SportteryLiveService(database)
ensemble_service = EnsembleAnalysisService(database, settings)

# ── Auto-fetch data on first run (Hugging Face Spaces) ──────────
try:
    with database.connection(read_only=True) as conn:
        _match_count = conn.execute("SELECT COUNT(*) FROM sporttery_matches").fetchone()[0]
    if _match_count == 0:
        try:
            sporttery_live_service.refresh()
        except Exception:
            # Fallback: load cached data if API not accessible (e.g., Hugging Face)
            import json
            _cache_path = Path(__file__).parent / "data" / "sporttery_matches_cache.json"
            if _cache_path.exists():
                _cached = json.loads(_cache_path.read_text(encoding="utf-8"))
                with database.connection() as conn:
                    for _m in _cached:
                        try:
                            _kickoff = _m.get("kickoff")
                            if isinstance(_kickoff, (int, float)):
                                from datetime import datetime, timezone
                                _kickoff = datetime.fromtimestamp(_kickoff / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                            _biz = _m.get("business_date")
                            if isinstance(_biz, (int, float)):
                                from datetime import datetime, timezone
                                _biz = datetime.fromtimestamp(_biz / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                            conn.execute(
                                """INSERT OR IGNORE INTO sporttery_matches
                                (match_id, official_match_id, business_date, match_number, kickoff,
                                 weekday, league_id, league_name, home_team_id, home_team,
                                 away_team_id, away_team, sell_status, match_status, remark,
                                 had_single, hhad_single, available_pools, last_update, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [
                                    _m.get("match_id"), _m.get("official_match_id", 0),
                                    _biz, _m.get("match_number", ""), _kickoff,
                                    _m.get("weekday", ""), _m.get("league_id", ""), _m.get("league_name", ""),
                                    _m.get("home_team_id", 0), _m.get("home_team", ""),
                                    _m.get("away_team_id", 0), _m.get("away_team", ""),
                                    _m.get("sell_status", ""), _m.get("match_status", ""), _m.get("remark", ""),
                                    _m.get("had_single", False), _m.get("hhad_single", False),
                                    _m.get("available_pools", ""), _m.get("last_update", _kickoff),
                                    _m.get("source", "sporttery.cn"),
                                ],
                            )
                        except Exception:
                            pass
except Exception:
    pass

# ── Page Registry ────────────────────────────────────────────────
SECTIONS = {
    "实时中心": {
        "⚽ 今日竞彩": lambda: render_live_matches(sporttery_live_service),
        "📊 价值发现": lambda: render_recommendations(database, sporttery_live_service, ensemble_service),
        "📋 比赛结果": lambda: render_results(database),
    },
    "足球分析": {
        "🔍 比赛分析": lambda: render_match_analysis(database, sporttery_live_service, ensemble_service),
        "📄 比赛详情": lambda: render_match_detail(database, st.session_state.get("detail_match_id", "")),
        "🎯 单场分析": lambda: render_single_match(analysis_service, database),
        "🎰 过关分析": lambda: render_parlay(database, sporttery_live_service, ensemble_service),
        "📦 批量分析": lambda: render_batch(analysis_service),
        "🧠 足球模型中心": lambda: render_model_center(database, settings),
    },
    "数字彩票": {
        "🎲 排列三分析": lambda: render_p3_analysis(database),
        "📋 排列三历史": lambda: render_p3_history(database),
        "🔧 排列三组合": lambda: render_p3_generator(database),
        "🎱 大乐透分析": lambda: render_dlt_analysis(database),
        "📋 大乐透历史": lambda: render_dlt_history(database),
        "🔧 大乐透组合": lambda: render_dlt_generator(database),
        "📊 彩票回测": lambda: render_lottery_backtest(database),
    },
    "研究回测": {
        "📈 足球回测": render_backtest,
    },
    "数据中心": {
        "💾 数据导入": lambda: render_data_center(database),
    },
    "系统": {
        "⚙️ 系统状态": lambda: render_system_status(settings, database),
    },
}

ALL_PAGES: dict[str, object] = {}
for section_pages in SECTIONS.values():
    ALL_PAGES.update(section_pages)


def _data_source_summary() -> str:
    """Return a small local-cache status string without mutating live data."""
    try:
        dates = sporttery_live_service.dates()
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("Failed to read live dates: %s", error)
        return "竞彩缓存不可读"
    return f"竞彩缓存 {len(dates)} 日" if dates else "竞彩缓存待同步"


def _get_display_url() -> str:
    """Get the display URL based on deployment environment."""
    import os
    # Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        return "Hugging Face Spaces"
    # Docker / local
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = os.environ.get("APP_PORT", "8502")
    return f"{host}:{port}"


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    db_ok = database.health_check()
    source_summary = _data_source_summary()
    status_text = "系统运行中" if db_ok else "数据库异常"
    status_variant = "live" if db_ok else "danger"

    st.markdown(
        f"""
<div class="sidebar-brand">
  <div class="sidebar-logo">🎯</div>
  <div class="sidebar-title">百万竞猜</div>
  <div class="sidebar-subtitle">Million Forecast Terminal</div>
  <div class="sidebar-tagline">足球竞彩 · 排列三 · 大乐透</div>
  <div class="sidebar-meta">
    <div class="sidebar-meta-row"><span>终端状态</span><strong>{render_status_dot(status_variant)}{safe_html(status_text)}</strong></div>
    <div class="sidebar-meta-row"><span>运行地址</span><strong>{safe_html(_get_display_url())}</strong></div>
    <div class="sidebar-meta-row"><span>数据源</span><strong>{safe_html(source_summary)}</strong></div>
    <div class="sidebar-meta-row"><span>版本</span><strong>v{safe_html(settings.version)}</strong></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Navigation with state management
    requested_page = st.session_state.pop("requested_page", None)
    if requested_page in ALL_PAGES:
        navigate_to(requested_page, set_return=False)
        st.rerun()

    for section_name, section_pages in SECTIONS.items():
        st.markdown(f'<div class="sidebar-section">{safe_html(section_name)}</div>', unsafe_allow_html=True)
        for page_name in section_pages:
            is_active = get_current_page() == page_name
            if st.button(
                page_name,
                key=f"nav-{section_name}-{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                navigate_to(page_name, set_return=False)
                st.rerun()

# ── Main Content ─────────────────────────────────────────────────
selected_page = get_current_page()
if selected_page not in ALL_PAGES:
    selected_page = "⚽ 今日竞彩"
    st.session_state["nav_page"] = selected_page

ALL_PAGES[selected_page]()
