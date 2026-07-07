"""Enterprise-grade UI components — Million Forecast Terminal.

Includes football, lottery (P3/DLT), and shared components.
"""

from __future__ import annotations

import html as _html
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "bg_primary": "#050816",
    "bg_secondary": "#0b1020",
    "bg_card": "rgba(15, 23, 42, 0.78)",
    "border": "rgba(148, 163, 184, 0.18)",
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "text_muted": "#94a3b8",
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "green": "#22c55e",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "purple": "#8b5cf6",
    "p3_front": "#3b82f6",
    "p3_mid": "#22c55e",
    "p3_back": "#a855f7",
    "dlt_front": "#3b82f6",
    "dlt_back": "#ef4444",
}

RISK_TEXT = (
    "本系统仅用于历史数据分析、概率研究和风险评估，不构成任何收益承诺。"
    "彩票和竞彩均存在高风险，请理性参与，严格控制预算。"
)


# ═══════════════════════════════════════════════════════════════════
# SAFE FORMATTERS
# ═══════════════════════════════════════════════════════════════════

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        return False
    return False


def safe_html(value: Any) -> str:
    if _is_missing(value):
        return ""
    return _html.escape(str(value), quote=True)


def format_number(value: Any, digits: int = 2, default: str = "-") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return default


def format_percent(value: Any, digits: int = 1, default: str = "-") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════

def apply_theme() -> None:
    st.markdown("""
<style>
:root {
    --bg-primary: #050816;
    --bg-secondary: #0b1020;
    --bg-tertiary: #111827;
    --bg-card: rgba(15, 23, 42, 0.78);
    --bg-card-hover: rgba(30, 41, 59, 0.88);
    --border: rgba(148, 163, 184, 0.18);
    --border-strong: rgba(148, 163, 184, 0.32);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --text-faint: #64748b;
    --blue: #3b82f6;
    --cyan: #06b6d4;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
    --purple: #8b5cf6;
}
html { color-scheme: dark; }
.stApp {
    background: radial-gradient(circle at 20% 0%, rgba(59,130,246,0.12), transparent 32rem),
                radial-gradient(circle at 90% 10%, rgba(6,182,212,0.08), transparent 30rem),
                var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1440px; }
h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; letter-spacing: -0.02em; font-weight: 700 !important; }
p, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text-secondary); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11,16,32,0.96), rgba(5,8,22,0.98)) !important;
    border-right: 1px solid var(--border) !important;
}
.sidebar-brand { padding: 0.9rem; border: 1px solid var(--border); border-radius: 18px; background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(30,41,59,0.64)); margin-bottom: 0.85rem; }
.sidebar-logo { width: 42px; height: 42px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: radial-gradient(circle at 30% 20%, rgba(168,85,247,0.36), rgba(59,130,246,0.20)); border: 1px solid rgba(168,85,247,0.35); font-size: 1.35rem; margin-bottom: 0.7rem; }
.sidebar-title { font-size: 1.08rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; }
.sidebar-subtitle { font-size: 0.71rem; color: var(--text-muted); margin-top: 0.18rem; letter-spacing: 0.04em; text-transform: uppercase; }
.sidebar-tagline { font-size: 0.68rem; color: var(--text-faint); margin-top: 0.3rem; }
.sidebar-meta { display: grid; gap: 0.35rem; margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid var(--border); font-size: 0.72rem; color: var(--text-muted); }
.sidebar-meta-row { display:flex; align-items:center; justify-content:space-between; gap:0.5rem; }
.sidebar-section { margin: 0.85rem 0 0.35rem; font-size: 0.68rem; color: var(--text-faint); font-weight: 800; letter-spacing: 0.11em; text-transform: uppercase; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; justify-content: flex-start; border-radius: 12px; border: 1px solid transparent;
    background: transparent; color: var(--text-secondary); box-shadow: none; padding: 0.58rem 0.72rem; font-size: 0.86rem;
    transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease, transform 0.16s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(59,130,246,0.10); border-color: rgba(59,130,246,0.24); color: var(--text-primary); transform: translateX(2px); box-shadow: none; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: linear-gradient(90deg, rgba(59,130,246,0.24), rgba(6,182,212,0.13)); border-color: rgba(59,130,246,0.36); color: var(--text-primary); box-shadow: inset 3px 0 0 var(--blue); }

/* Buttons */
.stButton > button { border-radius: 12px; border: 1px solid rgba(59,130,246,0.32); background: linear-gradient(135deg, rgba(59,130,246,0.92), rgba(6,182,212,0.72)); color: #fff; font-weight: 700; font-size: 0.85rem; box-shadow: 0 10px 24px rgba(37,99,235,0.22); transition: border-color 0.16s ease, transform 0.16s ease, box-shadow 0.16s ease; }
.stButton > button:hover { border-color: rgba(125,211,252,0.58); transform: translateY(-1px); box-shadow: 0 12px 30px rgba(37,99,235,0.30); }
.stButton > button[kind="secondary"] { background: rgba(15,23,42,0.82); color: var(--text-secondary); border-color: var(--border); box-shadow: none; }

/* Metrics */
[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 0.9rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.74rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 800 !important; }

/* Inputs */
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: rgba(15,23,42,0.84) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text-primary) !important;
}
.stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
    border-color: rgba(59,130,246,0.72) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: rgba(15,23,42,0.52); border: 1px solid var(--border); border-bottom: none; border-radius: 12px 12px 0 0; color: var(--text-muted); font-weight: 700; padding: 0.65rem 1rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(180deg, rgba(59,130,246,0.25), rgba(15,23,42,0.78)); color: var(--text-primary) !important; border-color: rgba(59,130,246,0.34); }

/* DataFrames */
.stDataFrame, [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 16px !important; overflow: hidden; background: rgba(15,23,42,0.62) !important; }

/* Animations */
@keyframes fadeIn { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.48; } }
.anim-fade { animation: fadeIn 0.32s ease-out; }
@media (prefers-reduced-motion: reduce) {
    .anim-fade { animation: none; }
    .status-dot.live, .status-dot.success { animation: none; }
}

/* Cards */
.card, .match-card, .insight-card, .metric-card, .risk-panel, .empty-state, .hero-pro {
    border: 1px solid var(--border); background: var(--bg-card); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 50px rgba(0,0,0,0.18); -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
}
.card, .insight-card { border-radius: 18px; padding: 1.05rem; }
.card:hover, .match-card:hover, .insight-card:hover, .metric-card:hover { border-color: rgba(96,165,250,0.38); background: var(--bg-card-hover); }
.hero-pro { position: relative; overflow: hidden; border-radius: 24px; padding: 1.35rem 1.45rem; margin: 0.35rem 0 1.1rem; background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(30,41,59,0.72)), radial-gradient(circle at 85% 18%, rgba(6,182,212,0.18), transparent 24rem); }
.hero-eyebrow { color: var(--cyan); font-size: 0.72rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.45rem; }
.hero-title { font-size: clamp(1.75rem, 2.8vw, 2.65rem); line-height: 1.08; margin: 0; color: var(--text-primary); }
.hero-subtitle { max-width: 760px; margin: 0.55rem 0 0; color: var(--text-secondary); font-size: 0.96rem; }
.hero-meta { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.9rem; }
.section-header { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin: 1.25rem 0 0.75rem; }
.section-title { font-size: 1.08rem; font-weight: 800; color: var(--text-primary); }
.section-subtitle { color: var(--text-muted); font-size: 0.82rem; margin-top: 0.16rem; }
.section-right { color: var(--text-muted); font-size: 0.78rem; }
.metric-card { border-radius: 18px; padding: 0.95rem 1rem; min-height: 6.1rem; }
.metric-label { color: var(--text-muted); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value { color: var(--text-primary); font-size: 1.55rem; font-weight: 900; line-height: 1.15; margin-top: 0.45rem; }
.metric-delta { font-size: 0.78rem; font-weight: 800; margin-top: 0.35rem; }
.metric-card.positive .metric-delta, .metric-card.positive .metric-value { color: var(--green); }
.metric-card.warning .metric-delta, .metric-card.warning .metric-value { color: var(--yellow); }
.metric-card.danger .metric-delta, .metric-card.danger .metric-value { color: var(--red); }
.metric-card.info .metric-delta, .metric-card.info .metric-value { color: var(--cyan); }
.metric-card.purple .metric-delta, .metric-card.purple .metric-value { color: var(--purple); }
.match-card { border-radius: 20px; padding: 1rem 1.05rem; margin: 0.75rem 0; }
.match-title { color: var(--text-primary); font-size: 1.04rem; font-weight: 900; letter-spacing: -0.01em; }
.match-meta { color: var(--text-muted); font-size: 0.76rem; margin-top: 0.25rem; }
.odds-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.55rem; margin-top: 0.75rem; }
.odds-box { background: rgba(2,6,23,0.36); border: 1px solid rgba(148,163,184,0.16); border-radius: 14px; padding: 0.62rem 0.55rem; text-align: center; }
.odds-label { color: var(--text-muted); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.05em; }
.odds-value { color: var(--text-primary); font-size: 1.08rem; font-weight: 900; margin-top: 0.18rem; }
.badge { display: inline-flex; align-items: center; gap: 0.28rem; padding: 0.22rem 0.58rem; border-radius: 999px; font-size: 0.70rem; font-weight: 850; letter-spacing: 0.02em; white-space: nowrap; }
.badge-success { background: rgba(34,197,94,0.13); color: #86efac; border: 1px solid rgba(34,197,94,0.26); }
.badge-warning { background: rgba(245,158,11,0.13); color: #fcd34d; border: 1px solid rgba(245,158,11,0.30); }
.badge-danger { background: rgba(239,68,68,0.13); color: #fca5a5; border: 1px solid rgba(239,68,68,0.28); }
.badge-info { background: rgba(59,130,246,0.13); color: #93c5fd; border: 1px solid rgba(59,130,246,0.28); }
.badge-purple { background: rgba(139,92,246,0.13); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.28); }
.badge-neutral { background: rgba(148,163,184,0.10); color: var(--text-secondary); border: 1px solid rgba(148,163,184,0.20); }
.status-dot { display:inline-block; width: 0.46rem; height: 0.46rem; border-radius: 50%; margin-right: 0.35rem; vertical-align: middle; }
.status-dot.live, .status-dot.success { background: var(--green); box-shadow: 0 0 14px rgba(34,197,94,0.65); animation: pulse 2s infinite; }
.status-dot.warning { background: var(--yellow); box-shadow: 0 0 12px rgba(245,158,11,0.48); }
.status-dot.error, .status-dot.danger { background: var(--red); box-shadow: 0 0 12px rgba(239,68,68,0.48); }
.status-dot.neutral { background: var(--text-faint); }
.risk-panel, .risk-note { border-radius: 16px; padding: 0.9rem 1rem; margin: 0.85rem 0; background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(15,23,42,0.70)); border: 1px solid rgba(245,158,11,0.28); color: #fde68a; font-size: 0.83rem; line-height: 1.65; }
.empty-state { border-radius: 22px; text-align: center; padding: 2.4rem 1.2rem; margin: 1rem 0; }
.empty-icon { font-size: 2.25rem; margin-bottom: 0.75rem; }
.empty-title { color: var(--text-primary); font-size: 1.05rem; font-weight: 900; }
.empty-desc { color: var(--text-muted); font-size: 0.84rem; margin-top: 0.35rem; }
.confidence-meter { margin: 0.8rem 0; }
.confidence-head { display:flex; justify-content:space-between; color: var(--text-muted); font-size:0.74rem; font-weight:800; margin-bottom:0.42rem; }
.confidence-bar { height: 0.56rem; border-radius:999px; overflow:hidden; background: rgba(51,65,85,0.86); border: 1px solid rgba(148,163,184,0.14); }
.confidence-fill { height:100%; border-radius:999px; transition: width 0.35s ease-in-out; }
.confidence-fill.success { background: linear-gradient(90deg, var(--green), var(--cyan)); }
.confidence-fill.warning { background: linear-gradient(90deg, var(--yellow), var(--cyan)); }
.confidence-fill.danger { background: linear-gradient(90deg, var(--red), var(--yellow)); }

/* Lottery number balls */
.lottery-ball {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2.2rem; height: 2.2rem; border-radius: 50%;
    font-size: 0.95rem; font-weight: 800; color: #fff;
    margin: 0.15rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.lottery-ball.p3 { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.lottery-ball.p3-pos1 { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
.lottery-ball.p3-pos2 { background: linear-gradient(135deg, #22c55e, #16a34a); }
.lottery-ball.p3-pos3 { background: linear-gradient(135deg, #a855f7, #7c3aed); }
.lottery-ball.dlt-front { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
.lottery-ball.dlt-back { background: linear-gradient(135deg, #ef4444, #dc2626); }
.lottery-ball.hot { border: 2px solid #ef4444; }
.lottery-ball.cold { border: 2px solid #3b82f6; }
.lottery-ball.warm { border: 2px solid #6b7280; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# LAYOUT COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def hero_pro(title: str, subtitle: str = "", eyebrow: str = "", meta: list[str] | None = None) -> None:
    parts = ['<div class="hero-pro anim-fade">']
    if eyebrow:
        parts.append(f'<div class="hero-eyebrow">{safe_html(eyebrow)}</div>')
    parts.append(f'<h1 class="hero-title">{safe_html(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="hero-subtitle">{safe_html(subtitle)}</p>')
    if meta:
        badges = "".join(render_badge(item, "neutral") for item in meta if item)
        parts.append(f'<div class="hero-meta">{badges}</div>')
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    hero_pro(title=title, subtitle=subtitle, eyebrow=eyebrow)


def metric_card(label: str, value: str, delta: str | None = None, variant: str = "neutral", caption: str | None = None) -> None:
    allowed = {"positive", "warning", "danger", "neutral", "info", "purple"}
    variant = variant if variant in allowed else "neutral"
    delta_html = f'<div class="metric-delta">{safe_html(delta)}</div>' if delta else ""
    caption_html = f'<div class="metric-caption">{safe_html(caption)}</div>' if caption else ""
    st.markdown(f"""
<div class="metric-card {variant} anim-fade">
  <div class="metric-label">{safe_html(label)}</div>
  <div class="metric-value">{safe_html(value)}</div>
  {delta_html}
  {caption_html}
</div>
""", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", right: str = "") -> None:
    subtitle_html = f'<div class="section-subtitle">{safe_html(subtitle)}</div>' if subtitle else ""
    right_html = f'<div class="section-right">{safe_html(right)}</div>' if right else ""
    st.markdown(f"""
<div class="section-header">
  <div>
    <div class="section-title">{safe_html(title)}</div>
    {subtitle_html}
  </div>
  {right_html}
</div>
""", unsafe_allow_html=True)


def render_badge(text: str, variant: str = "info") -> str:
    allowed = {"success", "warning", "danger", "info", "purple", "neutral"}
    variant = variant if variant in allowed else "info"
    return f'<span class="badge badge-{variant}">{safe_html(text)}</span>'


def render_status_dot(status: str = "live") -> str:
    allowed = {"live", "success", "warning", "error", "danger", "neutral"}
    status = status if status in allowed else "neutral"
    labels = {"live": "在线", "success": "成功", "warning": "警告", "error": "错误", "danger": "危险", "neutral": "中性"}
    label = labels.get(status, status)
    return f'<span class="status-dot {status}" aria-label="{label}"></span>'


def render_confidence_meter(value: int, label: str = "可信度") -> None:
    try:
        score = int(np.clip(int(value), 0, 100))
    except (TypeError, ValueError):
        score = 0
    color_class = "success" if score >= 75 else "warning" if score >= 55 else "danger"
    st.markdown(f"""
<div class="confidence-meter">
  <div class="confidence-head"><span>{safe_html(label)}</span><span>{score}/100</span></div>
  <div class="confidence-bar"><div class="confidence-fill {color_class}" style="width:{score}%"></div></div>
</div>
""", unsafe_allow_html=True)


def render_risk_note(text: str | None = None) -> None:
    content = text or RISK_TEXT
    st.markdown(f'<div class="risk-panel">{safe_html(content)}</div>', unsafe_allow_html=True)


def empty_state(title: str, description: str = "", icon: str = "📭") -> None:
    desc = f'<div class="empty-desc">{safe_html(description)}</div>' if description else ""
    st.markdown(f"""
<div class="empty-state anim-fade">
  <div class="empty-icon">{safe_html(icon)}</div>
  <div class="empty-title">{safe_html(title)}</div>
  {desc}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# LOTTERY COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def lottery_number_ball(number: int | str, ball_type: str = "p3", size: str = "normal") -> str:
    """Render a lottery number ball.

    Args:
        number: The number to display
        ball_type: "p3", "p3-pos1", "p3-pos2", "p3-pos3", "dlt-front", "dlt-back"
        size: "normal" or "small"
    """
    size_px = "1.8rem" if size == "small" else "2.2rem"
    font_size = "0.8rem" if size == "small" else "0.95rem"
    num_text = str(number).zfill(2)
    return f'<span class="lottery-ball {ball_type}" style="width:{size_px};height:{size_px};font-size:{font_size}" aria-label="号码{num_text}">{safe_html(num_text)}</span>'


def render_p3_numbers(d1: int, d2: int, d3: int, show_label: bool = True) -> None:
    """Render P3 draw numbers with styled balls."""
    balls = (
        lottery_number_ball(d1, "p3-pos1") +
        lottery_number_ball(d2, "p3-pos2") +
        lottery_number_ball(d3, "p3-pos3")
    )
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.3rem">{balls}</div>', unsafe_allow_html=True)


def render_dlt_numbers(front: list[int], back: list[int], show_label: bool = True) -> None:
    """Render DLT draw numbers with styled balls."""
    front_balls = "".join(lottery_number_ball(n, "dlt-front") for n in front)
    back_balls = "".join(lottery_number_ball(n, "dlt-back") for n in back)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.3rem">'
        f'{front_balls}'
        f'<span style="color:var(--text-muted);margin:0 0.3rem;font-weight:700">|</span>'
        f'{back_balls}'
        f'</div>',
        unsafe_allow_html=True,
    )


def lottery_draw_card(issue_no: str, draw_date: str, numbers: str, lottery_type: str = "p3") -> None:
    """Render a lottery draw card."""
    st.markdown(f"""
<div class="match-card anim-fade" style="padding:0.8rem 1rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:0.75rem;color:var(--text-muted)">第{safe_html(issue_no)}期</div>
      <div style="font-size:0.85rem;color:var(--text-secondary)">{safe_html(draw_date)}</div>
    </div>
    <div style="font-size:1.3rem;font-weight:900;color:var(--text-primary);letter-spacing:0.15em">{safe_html(numbers)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def model_score_card(score: float, label: str = "模型评分", max_score: float = 100) -> None:
    """Render a model score card with progress bar."""
    pct = min(100, max(0, score / max_score * 100))
    color_class = "success" if pct >= 70 else "warning" if pct >= 40 else "danger"
    st.markdown(f"""
<div style="margin:0.5rem 0">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
    <span style="font-size:0.78rem;color:var(--text-muted)">{safe_html(label)}</span>
    <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary)">{score:.1f}</span>
  </div>
  <div class="confidence-bar">
    <div class="confidence-fill {color_class}" style="width:{pct}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)


def data_quality_badge(score: float) -> str:
    """Render a data quality badge based on score (0-100)."""
    if score >= 90:
        return render_badge("数据质量: 高", "success")
    elif score >= 70:
        return render_badge("数据质量: 中", "info")
    elif score >= 40:
        return render_badge("数据质量: 低", "warning")
    else:
        return render_badge("数据质量: 不可用", "danger")


def data_quality_panel(score: float, issues: list[str] | None = None) -> None:
    """Render a data quality panel with score and issues."""
    badge = data_quality_badge(score)
    st.markdown(f"""
<div class="card" style="margin:0.5rem 0">
  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
    {badge}
    <span style="font-size:0.85rem;color:var(--text-secondary)">评分: {score:.0f}/100</span>
  </div>
</div>
""", unsafe_allow_html=True)
    if issues:
        for issue in issues:
            st.caption(f"⚠️ {issue}")


def missing_data_warning(fields: list[str]) -> None:
    """Render a warning for missing data fields."""
    if not fields:
        return
    field_list = "、".join(fields)
    st.warning(f"缺少数据字段: {field_list}，部分分析结果可能不完整。")


def provider_status_panel(status: dict[str, bool]) -> None:
    """Render provider status panel."""
    for provider, available in status.items():
        dot = render_status_dot("live" if available else "error")
        label = "可用" if available else "不可用"
        st.markdown(f"{dot} {safe_html(provider)}: {label}", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ═══════════════════════════════════════════════════════════════════

def plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_secondary"]),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def format_percent_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for col in columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: format_percent(x, 1, "—"))
    return formatted


def probability_chart(comparison: pd.DataFrame) -> go.Figure:
    def to_float(val: object) -> float:
        if isinstance(val, str):
            return float(val.rstrip("%")) / 100
        return float(val) if not _is_missing(val) else 0.0

    fig = go.Figure()
    if comparison.empty:
        return plotly_theme(fig)

    # Support both "赛果" and "结果" column names
    x_col = "赛果" if "赛果" in comparison.columns else "结果"

    fig.add_trace(go.Bar(x=comparison[x_col], y=comparison["模型概率"].apply(to_float), name="模型概率", marker_color=COLORS["blue"]))
    fig.add_trace(go.Bar(x=comparison[x_col], y=comparison["市场概率"].apply(to_float), name="市场概率", marker_color=COLORS["text_muted"], opacity=0.62))
    fig.update_layout(height=290, yaxis_tickformat=".0%", bargap=0.32)
    return plotly_theme(fig)


def score_heatmap(matrix: np.ndarray, max_goals: int = 7) -> go.Figure:
    import plotly.express as _px
    fig = _px.imshow(
        matrix[:max_goals, :max_goals],
        x=[str(v) for v in range(max_goals)],
        y=[str(v) for v in range(max_goals)],
        labels={"x": "客队进球", "y": "主队进球", "color": "概率"},
        text_auto=".1%",
        color_continuous_scale=[[0, "#0b1020"], [0.52, "#06b6d4"], [1, "#22c55e"]],
    )
    fig.update_layout(height=400)
    return plotly_theme(fig)


def render_match_card(home_team: str, away_team: str, league: str, match_time: str, odds_home: str, odds_draw: str, odds_away: str, status: str = "selling", match_number: str = "") -> None:
    status_map = {"selling": ("销售中", "success"), "paused": ("已暂停", "warning"), "waiting": ("待开售", "info"), "ended": ("已结束", "neutral")}
    text, variant = status_map.get(status, ("未知", "neutral"))
    num_part = f"{safe_html(match_number)} · " if match_number else ""
    st.markdown(f"""
<div class="match-card anim-fade">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
    <div>
      <div class="match-title">{safe_html(home_team)} <span style="color:var(--text-faint);font-weight:600">vs</span> {safe_html(away_team)}</div>
      <div class="match-meta">{render_status_dot(status == 'selling' and 'live' or 'neutral')}{num_part}{safe_html(league)} · 开赛 {safe_html(match_time)}</div>
    </div>
    {render_badge(text, variant)}
  </div>
  <div class="odds-grid">
    <div class="odds-box"><div class="odds-label">主胜</div><div class="odds-value">{safe_html(odds_home)}</div></div>
    <div class="odds-box"><div class="odds-label">平局</div><div class="odds-value">{safe_html(odds_draw)}</div></div>
    <div class="odds-box"><div class="odds-label">客胜</div><div class="odds-value">{safe_html(odds_away)}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
