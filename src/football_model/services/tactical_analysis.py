"""Enhanced tactical analysis service.

Provides comprehensive tactical analysis based on team profiles,
historical data, and formation patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from football_model.data import LocalDatabase
from football_model.services.team_profile import TeamProfile, TeamProfileService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TacticalReport:
    """Comprehensive tactical analysis for a match."""

    match_id: str

    # 阵型
    home_formation: str
    away_formation: str
    home_prev_formation: str
    away_prev_formation: str

    # 进攻风格
    home_attack_style: str
    away_attack_style: str
    home_attack_strength: float  # 0-1
    away_attack_strength: float

    # 防守风格
    home_defense_style: str
    away_defense_style: str
    home_defense_strength: float  # 0-1
    away_defense_strength: float

    # 压迫强度
    home_pressing: str
    away_pressing: str
    home_pressing_intensity: float  # 0-1
    away_pressing_intensity: float

    # 反击能力
    home_counter_attack: float  # 0-1
    away_counter_attack: float

    # 边路/中路
    home_wing_strength: float  # 0-1
    away_wing_strength: float
    home_central_control: float  # 0-1
    away_central_control: float

    # 定位球
    home_set_piece_threat: float  # 0-1
    away_set_piece_threat: float

    # 防线弱点
    home_defensive_weakness: str
    away_defensive_weakness: str

    # 关键对位
    key_matchups: list[str]

    # 战术优势
    tactical_advantage: str
    advantage_score: float  # 负数=客队优势, 正数=主队优势

    # 预期变化
    expected_changes: str

    # 概率影响
    probability_impact: str
    impact_score: float  # 0-1


class TacticalAnalysisService:
    """Service for generating comprehensive tactical analysis."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database
        self.profile_service = TeamProfileService(database)

    def generate_analysis(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        kickoff: object,
    ) -> TacticalReport:
        """Generate comprehensive tactical analysis."""
        import pandas as pd
        kickoff_ts = pd.to_datetime(kickoff)

        # 获取球队画像
        home_profile = self.profile_service.get_team_profile(home_team, league, kickoff_ts)
        away_profile = self.profile_service.get_team_profile(away_team, league, kickoff_ts)

        # 获取阵型
        home_formation, home_prev = self._get_formations(match_id, "home")
        away_formation, away_prev = self._get_formations(match_id, "away")

        # 分析进攻风格
        home_attack, home_atk_str = self._analyze_attack(home_profile)
        away_attack, away_atk_str = self._analyze_attack(away_profile)

        # 分析防守风格
        home_defense, home_def_str = self._analyze_defense(home_profile)
        away_defense, away_def_str = self._analyze_defense(away_profile)

        # 分析压迫强度
        home_pressing, home_press_int = self._analyze_pressing(home_profile)
        away_pressing, away_press_int = self._analyze_pressing(away_profile)

        # 计算各项能力
        home_counter = self._compute_counter(home_profile, away_profile)
        away_counter = self._compute_counter(away_profile, home_profile)

        home_wing = self._compute_wing_strength(home_profile)
        away_wing = self._compute_wing_strength(away_profile)

        home_central = self._compute_central_control(home_profile, away_profile)
        away_central = self._compute_central_control(away_profile, home_profile)

        home_setpiece = self._compute_setpiece(home_profile)
        away_setpiece = self._compute_setpiece(away_profile)

        # 弱点分析
        home_weakness = self._identify_weakness(home_profile, "home")
        away_weakness = self._identify_weakness(away_profile, "away")

        # 关键对位
        matchups = self._identify_matchups(home_profile, away_profile, home_team, away_team)

        # 战术优势
        advantage, adv_score = self._assess_advantage(home_profile, away_profile, home_team, away_team)

        # 预期变化
        changes = self._predict_changes(home_profile, away_profile)

        # 概率影响
        impact, impact_score = self._assess_impact(home_profile, away_profile)

        return TacticalReport(
            match_id=match_id,
            home_formation=home_formation,
            away_formation=away_formation,
            home_prev_formation=home_prev,
            away_prev_formation=away_prev,
            home_attack_style=home_attack,
            away_attack_style=away_attack,
            home_attack_strength=home_atk_str,
            away_attack_strength=away_atk_str,
            home_defense_style=home_defense,
            away_defense_style=away_defense,
            home_defense_strength=home_def_str,
            away_defense_strength=away_def_str,
            home_pressing=home_pressing,
            away_pressing=away_pressing,
            home_pressing_intensity=home_press_int,
            away_pressing_intensity=away_press_int,
            home_counter_attack=home_counter,
            away_counter_attack=away_counter,
            home_wing_strength=home_wing,
            away_wing_strength=away_wing,
            home_central_control=home_central,
            away_central_control=away_central,
            home_set_piece_threat=home_setpiece,
            away_set_piece_threat=away_setpiece,
            home_defensive_weakness=home_weakness,
            away_defensive_weakness=away_weakness,
            key_matchups=matchups,
            tactical_advantage=advantage,
            advantage_score=adv_score,
            expected_changes=changes,
            probability_impact=impact,
            impact_score=impact_score,
        )

    def _get_formations(self, match_id: str, side: str) -> tuple[str, str]:
        """Get current and previous formation."""
        with self.database.connection(read_only=True) as conn:
            # 当前阵型
            row = conn.execute(
                """SELECT formation FROM lineup_snapshots
                WHERE match_id = ? AND team_side = ? AND is_current = true
                ORDER BY captured_at DESC LIMIT 1""",
                [match_id, side],
            ).fetchone()
            current = row[0] if row and row[0] else "未知"

            # 上一场阵型
            row = conn.execute(
                """SELECT formation FROM lineup_snapshots
                WHERE match_id = ? AND team_side = ? AND is_current = false
                ORDER BY captured_at DESC LIMIT 1""",
                [match_id, side],
            ).fetchone()
            prev = row[0] if row and row[0] else "未知"

        return current, prev

    def _analyze_attack(self, profile: TeamProfile) -> tuple[str, float]:
        """Analyze attacking style and strength."""
        if profile.matches_played == 0:
            return "数据不足", 0.5

        gf = profile.goals_for

        if gf >= 2.5:
            style = "进攻型"
            strength = min(1.0, 0.7 + (gf - 2.5) * 0.15)
        elif gf >= 1.8:
            style = "均衡进攻"
            strength = 0.55 + (gf - 1.8) * 0.2
        elif gf >= 1.2:
            style = "均衡型"
            strength = 0.4 + (gf - 1.2) * 0.25
        elif gf >= 0.8:
            style = "防守反击型"
            strength = 0.25 + (gf - 0.8) * 0.35
        else:
            style = "保守型"
            strength = 0.15 + gf * 0.12

        return style, float(np.clip(strength, 0.1, 1.0))

    def _analyze_defense(self, profile: TeamProfile) -> tuple[str, float]:
        """Analyze defensive style and strength."""
        if profile.matches_played == 0:
            return "数据不足", 0.5

        ga = profile.goals_against

        if ga <= 0.6:
            style = "稳固防守"
            strength = 0.9
        elif ga <= 1.0:
            style = "组织型防守"
            strength = 0.7 + (1.0 - ga) * 0.5
        elif ga <= 1.4:
            style = "均衡防守"
            strength = 0.5 + (1.4 - ga) * 0.5
        elif ga <= 1.8:
            style = "高压防守"
            strength = 0.35 + (1.8 - ga) * 0.35
        else:
            style = "防守薄弱"
            strength = 0.2 + max(0, 2.5 - ga) * 0.1

        return style, float(np.clip(strength, 0.1, 1.0))

    def _analyze_pressing(self, profile: TeamProfile) -> tuple[str, float]:
        """Analyze pressing intensity."""
        if profile.matches_played == 0:
            return "数据不足", 0.5

        # 基于胜率和进球数判断压迫强度
        win_rate = profile.wins / max(profile.matches_played, 1)
        gf = profile.goals_for

        if win_rate > 0.6 and gf > 2.0:
            return "高位逼抢", 0.85
        elif win_rate > 0.5:
            return "中高位压迫", 0.7
        elif win_rate > 0.35:
            return "中位逼抢", 0.55
        elif win_rate > 0.2:
            return "低位防守", 0.35
        else:
            return "深度防守", 0.2

    def _compute_counter(self, team: TeamProfile, opponent: TeamProfile) -> float:
        """Compute counter-attack capability."""
        if team.matches_played == 0:
            return 0.5

        score = 0.4
        # 进攻能力强
        if team.goals_for > 1.5:
            score += 0.2
        # 客场表现好
        if team.away_strength > 1.0:
            score += 0.15
        # 对手防守弱
        if opponent.goals_against > 1.3:
            score += 0.15
        # 速度快（通过进球效率推断）
        if team.goals_for > opponent.goals_for:
            score += 0.1

        return float(np.clip(score, 0.1, 1.0))

    def _compute_wing_strength(self, profile: TeamProfile) -> float:
        """Compute wing play strength."""
        if profile.matches_played == 0:
            return 0.5
        # 基于进球数推断边路能力
        return float(np.clip(0.3 + profile.goals_for * 0.15, 0.1, 0.95))

    def _compute_central_control(self, team: TeamProfile, opponent: TeamProfile) -> float:
        """Compute central midfield control."""
        if team.matches_played == 0:
            return 0.5

        score = 0.45
        if team.elo_rating > opponent.elo_rating:
            score += 0.2
        if team.goals_for > opponent.goals_for:
            score += 0.15
        if team.goals_against < opponent.goals_against:
            score += 0.15

        return float(np.clip(score, 0.1, 0.95))

    def _compute_setpiece(self, profile: TeamProfile) -> float:
        """Compute set piece threat."""
        if profile.matches_played == 0:
            return 0.4
        # 基于进球数推断定位球能力
        return float(np.clip(0.3 + profile.goals_for * 0.1, 0.15, 0.85))

    def _identify_weakness(self, profile: TeamProfile, venue: str) -> str:
        """Identify defensive weaknesses."""
        if profile.matches_played == 0:
            return "数据不足，无法评估"

        weaknesses = []

        if profile.goals_against > 1.5:
            weaknesses.append("失球过多")
        if venue == "home" and profile.home_losses > profile.home_wins:
            weaknesses.append("主场胜率低")
        if venue == "away" and profile.away_losses > profile.away_wins:
            weaknesses.append("客场能力弱")
        if profile.goals_for < 1.0:
            weaknesses.append("进攻乏力")
        if profile.clean_sheets < profile.matches_played * 0.2:
            weaknesses.append("零封率低")

        if not weaknesses:
            return "无明显弱点"

        return "、".join(weaknesses[:3])

    def _identify_matchups(
        self,
        home: TeamProfile,
        away: TeamProfile,
        home_name: str,
        away_name: str,
    ) -> list[str]:
        """Identify key tactical matchups."""
        matchups = []

        # Elo对比
        elo_diff = home.elo_rating - away.elo_rating
        if abs(elo_diff) > 100:
            stronger = home_name if elo_diff > 0 else away_name
            matchups.append(f"{stronger}实力占优（Elo差值{abs(elo_diff):.0f}）")

        # 进攻对比
        gf_diff = home.goals_for - away.goals_for
        if abs(gf_diff) > 0.5:
            better = home_name if gf_diff > 0 else away_name
            matchups.append(f"{better}进攻更强（场均进球差{abs(gf_diff):.2f}）")

        # 防守对比
        ga_diff = home.goals_against - away.goals_against
        if abs(ga_diff) > 0.3:
            better = home_name if ga_diff < 0 else away_name
            matchups.append(f"{better}防守更稳（场均失球差{abs(ga_diff):.2f}）")

        # 主场优势
        if home.home_wins > home.home_losses:
            matchups.append(f"{home_name}主场优势明显")

        # 状态对比
        if home.form_last_5 and away.form_last_5:
            home_w = home.form_last_5.count("W")
            away_w = away.form_last_5.count("W")
            if abs(home_w - away_w) >= 2:
                better = home_name if home_w > away_w else away_name
                matchups.append(f"{better}近期状态更佳")

        if not matchups:
            matchups.append("双方实力接近，战术层面势均力敌")

        return matchups

    def _assess_advantage(
        self,
        home: TeamProfile,
        away: TeamProfile,
        home_name: str,
        away_name: str,
    ) -> tuple[str, float]:
        """Assess tactical advantage."""
        score = 0.0

        # Elo
        elo_diff = home.elo_rating - away.elo_rating
        score += np.clip(elo_diff / 200, -0.3, 0.3)

        # 进攻
        gf_diff = home.goals_for - away.goals_for
        score += np.clip(gf_diff * 0.15, -0.2, 0.2)

        # 防守
        ga_diff = away.goals_against - home.goals_against
        score += np.clip(ga_diff * 0.15, -0.2, 0.2)

        # 主场优势
        score += 0.1

        # 状态
        if home.form_last_5 and away.form_last_5:
            home_w = home.form_last_5.count("W")
            away_w = away.form_last_5.count("W")
            score += (home_w - away_w) * 0.03

        score = float(np.clip(score, -0.8, 0.8))

        if score > 0.3:
            verdict = f"{home_name}战术优势明显"
        elif score > 0.1:
            verdict = f"{home_name}略有优势"
        elif score < -0.3:
            verdict = f"{away_name}战术优势明显"
        elif score < -0.1:
            verdict = f"{away_name}略有优势"
        else:
            verdict = "战术层面势均力敌"

        return verdict, score

    def _predict_changes(self, home: TeamProfile, away: TeamProfile) -> str:
        """Predict tactical changes."""
        parts = []

        if home.form_last_5 and home.form_last_5.count("L") >= 2:
            parts.append("主队近期状态不佳，可能调整战术或阵容")
        if away.form_last_5 and away.form_last_5.count("L") >= 2:
            parts.append("客队近期状态不佳，可能加强防守")

        if home.goals_for < 1.0:
            parts.append("主队进攻乏力，可能加强前场投入")
        if away.goals_for < 1.0:
            parts.append("客队进攻乏力，可能采用防守反击")

        if home.goals_against > 1.5:
            parts.append("主队防守薄弱，可能调整防线")
        if away.goals_against > 1.5:
            parts.append("客队防守薄弱，可能加强中场保护")

        if not parts:
            parts.append("预计双方按常规战术出战")

        return "；".join(parts[:3])

    def _assess_impact(self, home: TeamProfile, away: TeamProfile) -> tuple[str, float]:
        """Assess tactical impact on probabilities."""
        elo_diff = abs(home.elo_rating - away.elo_rating)
        gf_diff = abs(home.goals_for - away.goals_for)

        if elo_diff > 200 or gf_diff > 1.0:
            return "战术层面对概率影响较大", 0.7
        elif elo_diff > 100 or gf_diff > 0.5:
            return "战术层面对概率有中等影响", 0.5
        else:
            return "战术层面对概率影响有限", 0.3
