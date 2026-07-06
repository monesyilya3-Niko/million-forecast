"""Match detail page — shows team history, lineups in formation view, injuries, and recent form."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)

# 队名映射
TEAM_NAME_MAP = {
    "埃尔夫斯堡": "Elfsborg", "哈马比": "Hammarby", "IFK哥德堡": "Goteborg",
    "AIK索尔纳": "AIK", "哥德堡盖斯": "GAIS", "北雪平": "Norrkoping",
    "赫根": "Hacken", "佐加顿斯": "Djurgarden", "天狼星": "Sirius",
    "米亚尔比": "Mjallby", "卡尔马": "Kalmar", "布鲁马波卡纳": "Brommapojkarna",
    "代格福什": "Degerfors", "瓦纳默": "Varnamo", "韦斯特罗斯": "Vasteras SK",
    "厄尔格里特": "Orgryte",
    "葡萄牙": "Portugal", "西班牙": "Spain", "巴西": "Brazil",
    "阿根廷": "Argentina", "法国": "France", "德国": "Germany",
    "英格兰": "England", "荷兰": "Netherlands", "比利时": "Belgium",
    "日本": "Japan", "韩国": "South Korea", "摩洛哥": "Morocco",
    "瑞士": "Switzerland", "哥伦比亚": "Colombia", "克罗地亚": "Croatia",
    "挪威": "Norway", "墨西哥": "Mexico", "美国": "USA",
    "沙特": "Saudi Arabia", "伊朗": "Iran", "澳大利亚": "Australia",
    "加拿大": "Canada", "塞内加尔": "Senegal", "乌拉圭": "Uruguay",
    "埃及": "Egypt", "加纳": "Ghana", "突尼斯": "Tunisia",
    "厄瓜多尔": "Ecuador", "卡塔尔": "Qatar", "沙特阿拉伯": "Saudi Arabia",
}

# 位置缩写映射
POS_MAP = {
    "G": "门将", "GK": "门将",
    "D": "后卫", "CB": "中后卫", "CD-L": "左中卫", "CD-R": "右中卫",
    "LB": "左后卫", "RB": "右后卫", "LCB": "左中卫", "RCB": "右中卫",
    "LWB": "左翼卫", "RWB": "右翼卫", "WB": "翼卫",
    "M": "中场", "CM": "中场", "CDM": "后腰", "CAM": "前腰",
    "AM": "前腰", "AM-L": "左前腰", "AM-R": "右前腰",
    "DM": "后腰", "LM": "左中场", "RM": "右中场",
    "LCM": "左中前卫", "RCM": "右中前卫",
    "F": "前锋", "ST": "中锋", "CF": "前锋", "SS": "影锋",
    "LW": "左边锋", "RW": "右边锋",
    "SUB": "替补",
}


def _normalize_position(pos: str) -> str:
    """Normalize ESPN position abbreviations to standard."""
    pos = pos.upper().strip()
    # ESPN specific mappings
    if pos in ("G", "GK"):
        return "GK"
    if pos.startswith("CD"):  # CD-L, CD-R -> CB
        return "CB"
    if pos in ("LB", "LWB", "LCB"):
        return "LB"
    if pos in ("RB", "RWB", "RCB"):
        return "RB"
    if pos in ("CB", "D"):
        return "CB"
    if pos.startswith("AM"):  # AM, AM-L, AM-R -> CAM
        return "CAM"
    if pos in ("CAM", "CM", "CDM", "DM"):
        return pos
    if pos in ("LM", "LW"):
        return "LM"
    if pos in ("RM", "RW"):
        return "RM"
    if pos in ("M"):
        return "CM"
    if pos in ("F", "ST", "CF", "SS"):
        return "ST"
    return pos


def _group_players_by_position(players: list[dict], formation: str = "4-3-3") -> dict[str, list[dict]]:
    """Group players by position line: GK, DEF, MID, FWD.

    Adjusts grouping to fit the formation by moving excess midfielders
    to forward line if needed.
    """
    groups = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    # Filter out substitutes first
    starters = [p for p in players if p.get("position", "").upper() != "SUB"]
    subs = [p for p in players if p.get("position", "").upper() == "SUB"]

    # If not enough starters, use subs to fill
    if len(starters) < 11:
        starters.extend(subs[:11 - len(starters)])

    for p in starters[:11]:  # Only first 11
        pos = _normalize_position(p.get("position", "M"))
        if pos in ("GK",):
            groups["GK"].append(p)
        elif pos in ("CB", "LB", "RB", "D"):
            groups["DEF"].append(p)
        elif pos in ("ST", "CF", "F", "SS", "LW", "RW"):
            groups["FWD"].append(p)
        elif pos in ("CM", "CDM", "CAM", "LM", "RM", "AM", "DM", "M"):
            groups["MID"].append(p)
        else:
            groups["MID"].append(p)

    # Parse formation to get expected counts
    lines = _formation_to_lines(formation)
    expected_fwd = lines[2] if len(lines) >= 3 else 3
    expected_mid = lines[1] if len(lines) >= 2 else 3
    expected_def = lines[0] if len(lines) >= 1 else 4

    # If FWD is short and MID has excess, move attacking midfielders to FWD
    while len(groups["FWD"]) < expected_fwd and len(groups["MID"]) > expected_mid:
        # Move the last midfielder (usually attacking) to forward
        groups["FWD"].append(groups["MID"].pop())

    # If DEF is short and MID has excess, move defensive midfielders to DEF
    while len(groups["DEF"]) < expected_def and len(groups["MID"]) > expected_mid:
        groups["DEF"].append(groups["MID"].pop(0))  # Move first midfielder (usually defensive)

    return groups


def _formation_to_lines(formation: str) -> list[int]:
    """Parse formation string like '4-3-3' into list of counts [4, 3, 3]."""
    if not formation or "-" not in formation:
        return [4, 3, 3]
    try:
        return [int(x) for x in formation.split("-")]
    except ValueError:
        return [4, 3, 3]


def _render_formation_pitch(
    team_name: str,
    formation: str,
    players: list[dict],
    is_confirmed: bool = False,
    source_label: str = "",
) -> None:
    """Render a football formation using Streamlit columns."""
    if not players:
        st.info(f"{team_name}: 无阵容数据")
        return

    # Status badge
    if is_confirmed:
        status = "✅ 已确认首发"
        badge_class = "badge-success"
    else:
        status = "⏳ 参考阵容"
        badge_class = "badge-warning"

    source_html = f' · <span style="color:#94a3b8;font-size:0.78rem">{source_label}</span>' if source_label else ""

    st.markdown(
        f'<div style="margin-bottom:0.5rem">'
        f'<span style="font-size:1.05rem;font-weight:700;color:#0f172a">{team_name}</span> '
        f'<span style="font-size:0.9rem;color:#2563eb;font-weight:600;margin-left:0.5rem">{formation}</span> '
        f'<span class="badge {badge_class}" style="margin-left:0.5rem">{status}</span>'
        f'{source_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Group players by position
    groups = _group_players_by_position(players, formation)
    lines = _formation_to_lines(formation)

    # Build formation display: FWD -> MID -> DEF -> GK (top to bottom)
    position_groups = []
    if len(lines) >= 3:
        position_groups = [
            ("FWD", lines[2], groups["FWD"]),
            ("MID", lines[1], groups["MID"]),
            ("DEF", lines[0], groups["DEF"]),
        ]
    else:
        position_groups = [
            ("FWD", 3, groups["FWD"]),
            ("MID", 3, groups["MID"]),
            ("DEF", 4, groups["DEF"]),
        ]

    # Render pitch background with CSS
    st.markdown(
        """<div style="
            background: linear-gradient(180deg, #166534 0%, #15803d 50%, #166534 100%);
            border-radius: 10px;
            padding: 1rem 0.5rem;
            border: 2px solid #22c55e;
            margin-bottom: 0.5rem;
        ">""",
        unsafe_allow_html=True,
    )

    # Render each line from front to back
    for group_name, count, group_players in position_groups:
        # Pad or trim to expected count
        display_players = group_players[:count] if group_players else []
        while len(display_players) < count:
            display_players.append({"name": "—", "position": group_name, "number": ""})

        cols = st.columns(max(count, 1))
        for i, player in enumerate(display_players):
            col_idx = min(i, len(cols) - 1)
            with cols[col_idx]:
                name = player.get("name", "—")
                number = player.get("number", player.get("jersey", ""))
                pos = player.get("position", "")
                pos_cn = POS_MAP.get(pos.upper(), pos) if pos else ""

                # Truncate long names
                if len(name) > 12:
                    name = name[:11] + "…"

                st.markdown(
                    f'<div style="text-align:center;padding:0.3rem">'
                    f'<div style="'
                    f'width:38px;height:38px;'
                    f'background:{"#2563eb" if is_confirmed else "#64748b"};'
                    f'border-radius:50%;'
                    f'display:inline-flex;align-items:center;justify-content:center;'
                    f'border:2px solid white;'
                    f'box-shadow:0 2px 4px rgba(0,0,0,0.2);'
                    f'">'
                    f'<span style="color:white;font-size:0.8rem;font-weight:700">{number}</span>'
                    f'</div>'
                    f'<div style="font-size:0.75rem;font-weight:600;color:white;margin-top:0.2rem;'
                    f'text-shadow:0 1px 2px rgba(0,0,0,0.5)">{name}</div>'
                    f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.8)">{pos_cn}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # GK line
    gk_players = groups["GK"][:1]
    if not gk_players:
        gk_players = [{"name": "—", "position": "G", "number": ""}]
    gk = gk_players[0]
    gk_name = gk.get("name", "—")
    gk_number = gk.get("number", gk.get("jersey", ""))
    if len(gk_name) > 12:
        gk_name = gk_name[:11] + "…"

    st.markdown(
        f'<div style="text-align:center;padding:0.5rem 0 0.2rem">'
        f'<div style="'
        f'width:38px;height:38px;'
        f'background:{"#ca8a04" if is_confirmed else "#94a3b8"};'
        f'border-radius:50%;'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'border:2px solid white;'
        f'box-shadow:0 2px 4px rgba(0,0,0,0.2);'
        f'">'
        f'<span style="color:white;font-size:0.8rem;font-weight:700">{gk_number}</span>'
        f'</div>'
        f'<div style="font-size:0.75rem;font-weight:600;color:white;margin-top:0.2rem;'
        f'text-shadow:0 1px 2px rgba(0,0,0,0.5)">{gk_name}</div>'
        f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.8)">门将</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Close pitch div
    st.markdown("</div>", unsafe_allow_html=True)

    # Show substitutes
    subs = [p for p in players if p.get("position", "").upper() == "SUB"]
    if subs:
        with st.expander(f"🔄 替补席 ({len(subs)}人)", expanded=False):
            sub_cols = st.columns(min(len(subs), 4))
            for i, p in enumerate(subs):
                with sub_cols[i % len(sub_cols)]:
                    name = p.get("name", "—")
                    number = p.get("number", p.get("jersey", ""))
                    if len(name) > 15:
                        name = name[:14] + "…"
                    st.markdown(
                        f'<div style="text-align:center;padding:0.2rem">'
                        f'<span style="font-size:0.75rem;color:#64748b">#{number}</span> '
                        f'<span style="font-size:0.82rem;font-weight:500">{name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def _render_player_table(players: list[dict], is_confirmed: bool = False) -> None:
    """Render player list as a table below the pitch."""
    if not players:
        return

    # Separate starters and substitutes
    starters = [p for p in players if p.get("position", "").upper() != "SUB"]
    subs = [p for p in players if p.get("position", "").upper() == "SUB"]

    rows = []
    for p in starters:
        pos = p.get("position", "M")
        pos_cn = POS_MAP.get(pos.upper(), pos)
        rows.append({
            "类型": "首发",
            "号码": p.get("number", p.get("jersey", "?")),
            "姓名": p.get("name", "未知"),
            "位置": pos_cn,
        })
    for p in subs:
        rows.append({
            "类型": "替补",
            "号码": p.get("number", p.get("jersey", "?")),
            "姓名": p.get("name", "未知"),
            "位置": "替补",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("号码")
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_match_detail(database: LocalDatabase, match_id: str) -> None:
    """Render detailed match information page."""
    from football_model.ui.components import hero_pro, section_header, empty_state, safe_html

    if not match_id:
        empty_state("未选择比赛", "请先从今日竞彩页面选择一场比赛", "⚽")
        return

    with database.connection(read_only=True) as conn:
        match = conn.execute(
            """
            SELECT s.match_id, s.home_team, s.away_team, s.kickoff, s.league_name,
                   s.match_number, s.sell_status, s.business_date,
                   r.status, r.home_goals, r.away_goals, r.halftime_home_goals, r.halftime_away_goals
            FROM sporttery_matches s
            LEFT JOIN match_results r USING (match_id)
            WHERE s.match_id = ?
            """,
            [match_id],
        ).fetchone()

    if not match:
        st.error("比赛未找到")
        return

    home_team, away_team = match[1], match[2]
    league_name = match[4]
    kickoff = match[3]

    # Convert league_name to competition name for model lookup
    from football_model.core import competition_for_league
    competition = competition_for_league(str(league_name)) or str(league_name)

    hero_pro(
        f"{home_team} vs {away_team}",
        f"{match[5]} · {league_name} · {pd.to_datetime(kickoff):%Y-%m-%d %H:%M}",
        "MATCH DETAIL",
        ["比赛情报档案"],
    )

    # Back button
    from football_model.ui.navigation import render_back_button
    render_back_button("← 返回上一页", key="detail-back")

    # 完场比分
    if match[8] == "FT":
        section_header("完场比分", "比赛最终结果。")
        c1, c2, c3 = st.columns([1, 0.3, 1])
        with c1:
            st.markdown(f"<div style='text-align:right;font-size:2rem;font-weight:800'>{safe_html(home_team)}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='text-align:center;font-size:2.5rem;font-weight:800;color:var(--blue)'>{safe_html(str(match[9]))}:{safe_html(str(match[10]))}</div>", unsafe_allow_html=True)
            if match[11] is not None:
                st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:var(--text-muted)'>半场 {safe_html(str(match[11]))}:{safe_html(str(match[12]))}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='text-align:left;font-size:2rem;font-weight:800'>{safe_html(away_team)}</div>", unsafe_allow_html=True)

    # 球队画像
    section_header("球队画像", "双方近期状态、Elo评分和主客场表现。")
    _render_team_profiles(database, home_team, away_team, competition, kickoff)

    # 上场比赛分析
    section_header("上场比赛分析", "双方最近一场比赛的表现。")
    _render_previous_matches(database, home_team, away_team, competition, kickoff)

    # 近期战绩
    section_header("近期战绩", "双方近5场比赛结果。")
    col1, col2 = st.columns(2)
    with col1:
        _render_team_form(database, home_team, kickoff, "主场")
    with col2:
        _render_team_form(database, away_team, kickoff, "客场")

    # 历史交锋
    section_header("历史交锋", "双方近期直接对话记录。")
    _render_h2h(database, home_team, away_team, competition, kickoff)

    # 阵容（本场 + 上一场）
    section_header("阵容信息", "首发阵容和替补席。")
    _render_lineups(database, match_id, home_team, away_team, kickoff, league_name)

    # 伤停
    section_header("伤停信息", "伤病和停赛情况。")
    _render_injuries(database, match_id)

    # 技战术分析
    section_header("技战术分析", "双方战术风格和关键对位。")
    _render_tactical_analysis(database, match_id, home_team, away_team, competition, kickoff)

    # 赔率
    section_header("赔率变化", "官方赔率走势。")
    _render_odds_history(database, match_id)


def _render_team_form(database: LocalDatabase, team: str, before_date: datetime, label: str) -> None:
    """Render recent form for a team."""
    en_team = TEAM_NAME_MAP.get(team, team)

    with database.connection(read_only=True) as conn:
        recent = conn.execute(
            """
            SELECT m.kickoff, m.home_team, m.away_team, m.home_goals, m.away_goals
            FROM matches m
            WHERE (m.home_team IN (?, ?) OR m.away_team IN (?, ?))
              AND m.kickoff < ?
              AND m.home_goals IS NOT NULL
            ORDER BY m.kickoff DESC
            LIMIT 5
            """,
            [team, en_team, team, en_team, before_date],
        ).fetchall()

    if not recent:
        st.info(f"{team} 暂无近期比赛记录")
        return

    st.markdown(f"**{team}** ({label})")
    for match in recent[:5]:
        home, away, hg, ag = match[1], match[2], match[3], match[4]
        is_home = (home == team or home == en_team)
        opponent = away if is_home else home
        gf = hg if is_home else ag
        ga = ag if is_home else hg

        if gf > ga:
            result, color = "W", "#16a34a"
        elif gf == ga:
            result, color = "D", "#ca8a04"
        else:
            result, color = "L", "#dc2626"

        venue = "主" if is_home else "客"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0'>"
            f"<span style='background:{color};color:#fff;padding:0.1rem 0.4rem;border-radius:3px;font-size:0.75rem;font-weight:700'>{result}</span>"
            f"<span style='font-size:0.8rem;color:#94a3b8'>{pd.to_datetime(match[0]):%m-%d}</span>"
            f"<span style='font-size:0.85rem'>{venue} {opponent}</span>"
            f"<span style='font-size:0.85rem;font-weight:600'>{gf}:{ga}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _get_prev_lineup(database: LocalDatabase, team: str, en_team: str, before_date: datetime, side: str) -> dict | None:
    """Get previous match lineup for a team."""
    with database.connection(read_only=True) as conn:
        prev = conn.execute(
            """
            SELECT l.players_json, l.formation, l.captured_at, s.home_team, s.away_team
            FROM lineup_snapshots l
            JOIN sporttery_matches s ON l.match_id = s.match_id
            WHERE l.team_side = ?
              AND s.kickoff < ?
              AND (s.home_team IN (?, ?) OR s.away_team IN (?, ?))
              AND l.players_json IS NOT NULL
            ORDER BY s.kickoff DESC
            LIMIT 1
            """,
            [side, before_date, team, en_team, team, en_team],
        ).fetchone()

    if prev and prev[0]:
        try:
            players = json.loads(prev[0])
            return {"players": players, "formation": prev[1], "date": prev[2], "home": prev[3], "away": prev[4]}
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _render_lineups(database: LocalDatabase, match_id: str, home_team: str, away_team: str, kickoff: datetime, league_name: str) -> None:
    """Render lineup information — previous match as default, current when available."""
    with database.connection(read_only=True) as conn:
        lineups = conn.execute(
            """
            SELECT team_side, formation, confirmed, players_json, captured_at, is_current
            FROM lineup_snapshots
            WHERE match_id = ?
            ORDER BY is_current DESC, captured_at DESC
            """,
            [match_id],
        ).fetchall()

    # Organize lineups: current first, then previous
    current = {}
    previous = {}
    for lineup in lineups:
        side = lineup[0]
        is_current = lineup[5]
        if is_current and side not in current:
            current[side] = {
                "formation": lineup[1],
                "confirmed": lineup[2],
                "players": lineup[3],
                "captured_at": lineup[4],
            }
        elif not is_current and side not in previous:
            previous[side] = {
                "formation": lineup[1],
                "confirmed": lineup[2],
                "players": lineup[3],
                "captured_at": lineup[4],
            }

    en_home = TEAM_NAME_MAP.get(home_team, home_team)
    en_away = TEAM_NAME_MAP.get(away_team, away_team)

    for side, team, en_team in [("home", home_team, en_home), ("away", away_team, en_away)]:
        side_label = "主队" if side == "home" else "客队"

        # Priority: current lineup > previous lineup in DB > ESPN fallback
        if side in current and current[side]["players"]:
            # Show current match lineup (confirmed)
            players = json.loads(current[side]["players"]) if isinstance(current[side]["players"], str) else current[side]["players"]
            _render_formation_pitch(
                team_name=f"{side_label} {team}",
                formation=current[side]["formation"] or "4-3-3",
                players=players,
                is_confirmed=True,
                source_label="本场首发",
            )
            with st.expander("📋 球员名单", expanded=False):
                _render_player_table(players, is_confirmed=True)

        elif side in previous and previous[side]["players"]:
            # Show previous match lineup as reference
            players = json.loads(previous[side]["players"]) if isinstance(previous[side]["players"], str) else previous[side]["players"]
            _render_formation_pitch(
                team_name=f"{side_label} {team}",
                formation=previous[side]["formation"] or "4-3-3",
                players=players,
                is_confirmed=False,
                source_label="上一场首发（本场待更新）",
            )
            with st.expander("📋 球员名单", expanded=False):
                _render_player_table(players, is_confirmed=False)

        else:
            # Try ESPN fallback
            espn_players = _fetch_espn_roster(league_name, team, en_team)
            if espn_players and len(espn_players) >= 5:
                _render_formation_pitch(
                    team_name=f"{side_label} {team}",
                    formation="4-3-3",
                    players=espn_players,
                    is_confirmed=False,
                    source_label=f"ESPN阵容参考（{len(espn_players)}人）",
                )
                with st.expander("📋 球员名单", expanded=False):
                    _render_player_table(espn_players, is_confirmed=False)
            else:
                st.info(f"{side_label} {team} 暂无阵容数据，等待开赛前更新")


def _fetch_espn_roster(league: str, team: str, en_team: str) -> list[dict] | None:
    """Try to fetch team roster from ESPN API."""
    try:
        from football_model.data.adapters.espn import ESPNAdapter
        adapter = ESPNAdapter()
        players = adapter.get_team_roster(league, en_team)
        if players:
            return [{"name": p.name, "position": p.position, "number": p.jersey or "?", "is_starter": True} for p in players]
    except Exception as e:
        logger.debug("ESPN roster fetch failed for %s: %s", team, e)
    return None


def _render_injuries(database: LocalDatabase, match_id: str) -> None:
    """Render injury information."""
    with database.connection(read_only=True) as conn:
        injuries = conn.execute(
            """
            SELECT team_side, players_json, captured_at
            FROM injury_snapshots
            WHERE match_id = ?
            ORDER BY captured_at DESC
            """,
            [match_id],
        ).fetchall()

    if not injuries:
        st.info("暂无伤停数据")
        return

    for injury in injuries:
        side = "主队" if injury[0] == "home" else "客队"
        try:
            players = json.loads(injury[1]) if injury[1] else []
            if players:
                st.markdown(f"**{side}伤停:**")
                for p in players:
                    name = p.get("name", "未知")
                    pos = p.get("position", "")
                    reason = p.get("injury_type", p.get("reason", "未知"))
                    pos_cn = POS_MAP.get(pos.upper(), pos) if pos else ""
                    st.markdown(f"- **{name}** {pos_cn} · {reason}")
            else:
                st.markdown(f"**{side}**: 无伤停报告")
        except (json.JSONDecodeError, TypeError):
            st.info(f"{side}伤停数据格式异常")


def _render_odds_history(database: LocalDatabase, match_id: str) -> None:
    """Render odds history chart."""
    with database.connection(read_only=True) as conn:
        odds = conn.execute(
            """
            SELECT captured_at, market, selection, odds
            FROM odds_snapshots
            WHERE match_id = ?
            ORDER BY captured_at
            """,
            [match_id],
        ).fetchall()

    if not odds:
        st.info("暂无赔率历史数据")
        return

    df = pd.DataFrame(odds, columns=["时间", "市场", "选项", "赔率"])

    had = df[df["市场"].str.contains("1x2", na=False)]
    if not had.empty:
        import plotly.express as px
        fig = px.line(had, x="时间", y="赔率", color="选项", title="胜平负赔率变化", markers=True)
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            font=dict(color="#475569"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    latest = df.groupby(["市场", "选项"]).last().reset_index()
    if not latest.empty:
        st.dataframe(latest[["市场", "选项", "赔率"]].style.format({"赔率": "{:.2f}"}), hide_index=True, use_container_width=True)


def _render_team_profiles(database: LocalDatabase, home_team: str, away_team: str, league: str, kickoff: object) -> None:
    """Render team profile comparison."""
    from football_model.services.team_profile import TeamProfileService
    from football_model.ui.components import safe_html

    service = TeamProfileService(database)
    kickoff_ts = pd.to_datetime(kickoff)

    home_en = TEAM_NAME_MAP.get(home_team, home_team)
    away_en = TEAM_NAME_MAP.get(away_team, away_team)

    home_profile = service.get_team_profile(home_en, league, kickoff_ts)
    away_profile = service.get_team_profile(away_en, league, kickoff_ts)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{safe_html(home_team)}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Elo", f"{home_profile.elo_rating:.0f}")
        m2.metric("场均进球", f"{home_profile.goals_for:.2f}")
        m3.metric("场均失球", f"{home_profile.goals_against:.2f}")

        if home_profile.matches_played > 0:
            st.caption(f"近5场: {safe_html(home_profile.form_last_5)} | {home_profile.wins}胜{home_profile.draws}平{home_profile.losses}负")
            st.caption(f"主场: {home_profile.home_wins}胜{home_profile.home_draws}平{home_profile.home_losses}负")
        else:
            st.info("暂无历史数据")

    with col2:
        st.markdown(f"**{safe_html(away_team)}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Elo", f"{away_profile.elo_rating:.0f}")
        m2.metric("场均进球", f"{away_profile.goals_for:.2f}")
        m3.metric("场均失球", f"{away_profile.goals_against:.2f}")

        if away_profile.matches_played > 0:
            st.caption(f"近5场: {safe_html(away_profile.form_last_5)} | {away_profile.wins}胜{away_profile.draws}平{away_profile.losses}负")
            st.caption(f"客场: {away_profile.away_wins}胜{away_profile.away_draws}平{away_profile.away_losses}负")
        else:
            st.info("暂无历史数据")


def _render_previous_matches(database: LocalDatabase, home_team: str, away_team: str, league: str, kickoff: object) -> None:
    """Render previous match analysis for both teams."""
    from football_model.services.previous_match import PreviousMatchService
    from football_model.ui.components import safe_html

    service = PreviousMatchService(database)
    kickoff_ts = pd.to_datetime(kickoff)

    home_en = TEAM_NAME_MAP.get(home_team, home_team)
    away_en = TEAM_NAME_MAP.get(away_team, away_team)

    home_prev, away_prev = service.get_both_previous_matches(home_en, away_en, league, kickoff_ts)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{safe_html(home_team)}**")
        if home_prev:
            result_color = "#22c55e" if home_prev.result == "W" else "#eab308" if home_prev.result == "D" else "#ef4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem'>"
                f"<span style='background:{result_color};color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-weight:700'>{home_prev.result}</span>"
                f"<span>{safe_html(home_prev.venue)} {safe_html(home_prev.opponent)}</span>"
                f"<span style='font-weight:700'>{safe_html(home_prev.score)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"日期: {home_prev.match_date} | {home_prev.days_ago}天前")
            st.caption(f"体能: {home_prev.fatigue_level}")
            st.caption(home_prev.impact_on_next)
        else:
            st.info("暂无上场比赛数据")

    with col2:
        st.markdown(f"**{safe_html(away_team)}**")
        if away_prev:
            result_color = "#22c55e" if away_prev.result == "W" else "#eab308" if away_prev.result == "D" else "#ef4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem'>"
                f"<span style='background:{result_color};color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-weight:700'>{away_prev.result}</span>"
                f"<span>{safe_html(away_prev.venue)} {safe_html(away_prev.opponent)}</span>"
                f"<span style='font-weight:700'>{safe_html(away_prev.score)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"日期: {away_prev.match_date} | {away_prev.days_ago}天前")
            st.caption(f"体能: {away_prev.fatigue_level}")
            st.caption(away_prev.impact_on_next)
        else:
            st.info("暂无上场比赛数据")


def _render_h2h(database: LocalDatabase, home_team: str, away_team: str, league: str, kickoff: object) -> None:
    """Render head-to-head record."""
    from football_model.services.team_profile import TeamProfileService
    from football_model.ui.components import safe_html

    service = TeamProfileService(database)
    kickoff_ts = pd.to_datetime(kickoff)

    home_en = TEAM_NAME_MAP.get(home_team, home_team)
    away_en = TEAM_NAME_MAP.get(away_team, away_team)

    h2h = service.get_h2h(home_en, away_en, league, kickoff_ts)

    if h2h.empty:
        st.info("暂无历史交锋记录")
        return

    # Summary
    home_wins = len(h2h[(h2h["主队"] == home_en) & (h2h["主队进球"] > h2h["客队进球"])])
    away_wins = len(h2h[(h2h["主队"] == away_en) & (h2h["主队进球"] > h2h["客队进球"])])
    draws = len(h2h[h2h["主队进球"] == h2h["客队进球"]])

    st.caption(f"近{len(h2h)}场交锋: {safe_html(home_team)} {home_wins}胜 {draws}平 {away_wins}负 {safe_html(away_team)}")
    st.dataframe(h2h, hide_index=True, use_container_width=True)


def _render_tactical_analysis(database: LocalDatabase, match_id: str, home_team: str, away_team: str, league: str, kickoff: object) -> None:
    """Render tactical analysis."""
    from football_model.services.tactical_analysis import TacticalAnalysisService
    from football_model.ui.components import safe_html

    service = TacticalAnalysisService(database)

    try:
        report = service.generate_analysis(match_id, home_team, away_team, league, kickoff)
    except Exception as e:
        logger.warning("Tactical analysis failed: %s", e)
        st.info("技战术分析数据不足，等待更多比赛数据积累。")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{safe_html(home_team)}**")
        st.caption(f"阵型: {safe_html(report.home_formation)}")
        st.caption(f"进攻风格: {safe_html(report.home_attack_style)}")
        st.caption(f"防守风格: {safe_html(report.home_defense_style)}")
        st.caption(f"压迫强度: {safe_html(report.home_pressing)}")

    with col2:
        st.markdown(f"**{safe_html(away_team)}**")
        st.caption(f"阵型: {safe_html(report.away_formation)}")
        st.caption(f"进攻风格: {safe_html(report.away_attack_style)}")
        st.caption(f"防守风格: {safe_html(report.away_defense_style)}")
        st.caption(f"压迫强度: {safe_html(report.away_pressing)}")

    st.markdown("**战术分析**")
    st.caption(f"战术优势: {safe_html(report.tactical_advantage)}")
    for matchup in report.key_matchups:
        st.caption(f"- {safe_html(matchup)}")
    st.caption(f"预期变化: {safe_html(report.expected_changes)}")
    st.caption(f"概率影响: {safe_html(report.probability_impact)}")
