"""Tactical analysis service.

Generates tactical analysis for matches based on team profiles,
formations, and historical patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from football_model.data import LocalDatabase
from football_model.services.team_profile import TeamProfile, TeamProfileService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TacticalReport:
    """Tactical analysis for a match."""

    match_id: str
    # Formations
    home_formation: str
    away_formation: str
    # Styles
    home_attack_style: str
    away_attack_style: str
    home_defense_style: str
    away_defense_style: str
    # Intensity
    home_pressing: str
    away_pressing: str
    # Strengths
    home_counter_attack: float
    away_counter_attack: float
    home_wing_strength: float
    away_wing_strength: float
    home_midfield_control: float
    away_midfield_control: float
    home_set_piece_threat: float
    away_set_piece_threat: float
    # Weaknesses
    home_defensive_weakness: str
    away_defensive_weakness: str
    # Matchups
    key_matchups: list[str]
    tactical_advantage: str
    expected_changes: str
    probability_impact: str


class TacticalAnalysisService:
    """Service for generating tactical analysis."""

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
        """Generate tactical analysis for a match.

        Args:
            match_id: Match identifier
            home_team: Home team name
            away_team: Away team name
            league: Competition name
            kickoff: Match kickoff time

        Returns:
            TacticalReport with analysis
        """
        import pandas as pd
        kickoff_ts = pd.to_datetime(kickoff)

        # Get team profiles
        home_profile = self.profile_service.get_team_profile(home_team, league, kickoff_ts)
        away_profile = self.profile_service.get_team_profile(away_team, league, kickoff_ts)

        # Determine formations (from lineup data if available, else infer)
        home_formation = self._get_formation(match_id, "home")
        away_formation = self._get_formation(match_id, "away")

        # Analyze styles
        home_attack = self._analyze_attack_style(home_profile)
        away_attack = self._analyze_attack_style(away_profile)
        home_defense = self._analyze_defense_style(home_profile)
        away_defense = self._analyze_defense_style(away_profile)

        # Analyze pressing
        home_pressing = self._analyze_pressing(home_profile)
        away_pressing = self._analyze_pressing(away_profile)

        # Compute strengths
        home_counter = self._compute_counter_strength(home_profile, away_profile)
        away_counter = self._compute_counter_strength(away_profile, home_profile)
        home_wing = self._compute_wing_strength(home_profile)
        away_wing = self._compute_wing_strength(away_profile)
        home_midfield = self._compute_midfield_control(home_profile, away_profile)
        away_midfield = self._compute_midfield_control(away_profile, home_profile)
        home_setpiece = self._compute_setpiece_threat(home_profile)
        away_setpiece = self._compute_setpiece_threat(away_profile)

        # Weaknesses
        home_weakness = self._identify_weakness(home_profile)
        away_weakness = self._identify_weakness(away_profile)

        # Key matchups
        matchups = self._identify_matchups(home_profile, away_profile)

        # Tactical advantage
        advantage = self._assess_advantage(home_profile, away_profile)

        # Expected changes
        changes = self._predict_changes(home_profile, away_profile)

        # Probability impact
        impact = self._assess_probability_impact(home_profile, away_profile)

        return TacticalReport(
            match_id=match_id,
            home_formation=home_formation,
            away_formation=away_formation,
            home_attack_style=home_attack,
            away_attack_style=away_attack,
            home_defense_style=home_defense,
            away_defense_style=away_defense,
            home_pressing=home_pressing,
            away_pressing=away_pressing,
            home_counter_attack=home_counter,
            away_counter_attack=away_counter,
            home_wing_strength=home_wing,
            away_wing_strength=away_wing,
            home_midfield_control=home_midfield,
            away_midfield_control=away_midfield,
            home_set_piece_threat=home_setpiece,
            away_set_piece_threat=away_setpiece,
            home_defensive_weakness=home_weakness,
            away_defensive_weakness=away_weakness,
            key_matchups=matchups,
            tactical_advantage=advantage,
            expected_changes=changes,
            probability_impact=impact,
        )

    def _get_formation(self, match_id: str, side: str) -> str:
        """Get formation from lineup data."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                """SELECT formation FROM lineup_snapshots
                WHERE match_id = ? AND team_side = ? AND is_current = true
                ORDER BY captured_at DESC LIMIT 1""",
                [match_id, side],
            ).fetchone()

        return row[0] if row and row[0] else "未知"

    def _analyze_attack_style(self, profile: TeamProfile) -> str:
        """Analyze team's attacking style."""
        if profile.goals_for > 2.0:
            return "进攻型"
        elif profile.goals_for > 1.5:
            return "均衡型"
        else:
            return "防守反击型"

    def _analyze_defense_style(self, profile: TeamProfile) -> str:
        """Analyze team's defensive style."""
        if profile.goals_against < 0.8:
            return "稳固防守"
        elif profile.goals_against < 1.2:
            return "均衡防守"
        else:
            return "防守薄弱"

    def _analyze_pressing(self, profile: TeamProfile) -> str:
        """Analyze pressing intensity."""
        if profile.wins > profile.losses * 2:
            return "高位逼抢"
        elif profile.wins > profile.losses:
            return "中位逼抢"
        else:
            return "低位防守"

    def _compute_counter_strength(self, team: TeamProfile, opponent: TeamProfile) -> float:
        """Compute counter-attack strength (0-1)."""
        score = 0.5
        if team.goals_for > 1.5:
            score += 0.2
        if opponent.goals_against > 1.2:
            score += 0.15
        if team.away_strength > 1.1:
            score += 0.15
        return min(1.0, score)

    def _compute_wing_strength(self, profile: TeamProfile) -> float:
        """Compute wing play strength (0-1)."""
        # Simplified: based on goals scored
        return min(1.0, profile.goals_for / 3.0)

    def _compute_midfield_control(self, team: TeamProfile, opponent: TeamProfile) -> float:
        """Compute midfield control (0-1)."""
        score = 0.5
        if team.elo_rating > opponent.elo_rating:
            score += 0.2
        if team.goals_for > opponent.goals_for:
            score += 0.15
        if team.goals_against < opponent.goals_against:
            score += 0.15
        return min(1.0, score)

    def _compute_setpiece_threat(self, profile: TeamProfile) -> float:
        """Compute set piece threat (0-1)."""
        # Default moderate threat
        return 0.5

    def _identify_weakness(self, profile: TeamProfile) -> str:
        """Identify defensive weakness."""
        weaknesses = []
        if profile.goals_against > 1.5:
            weaknesses.append("失球过多")
        if profile.away_losses > profile.away_wins:
            weaknesses.append("客场能力弱")
        if not weaknesses:
            return "无明显弱点"
        return "、".join(weaknesses)

    def _identify_matchups(self, home: TeamProfile, away: TeamProfile) -> list[str]:
        """Identify key matchups."""
        matchups = []

        if home.elo_rating > away.elo_rating + 100:
            matchups.append(f"主队Elo优势({home.elo_rating:.0f} vs {away.elo_rating:.0f})")
        elif away.elo_rating > home.elo_rating + 100:
            matchups.append(f"客队Elo优势({away.elo_rating:.0f} vs {home.elo_rating:.0f})")

        if home.goals_for > away.goals_for + 0.5:
            matchups.append("主队进攻更强")
        elif away.goals_for > home.goals_for + 0.5:
            matchups.append("客队进攻更强")

        if not matchups:
            matchups.append("双方实力接近")

        return matchups

    def _assess_advantage(self, home: TeamProfile, away: TeamProfile) -> str:
        """Assess tactical advantage."""
        home_score = 0
        away_score = 0

        if home.elo_rating > away.elo_rating:
            home_score += 1
        else:
            away_score += 1

        if home.goals_for > away.goals_for:
            home_score += 1
        else:
            away_score += 1

        if home.goals_against < away.goals_against:
            home_score += 1
        else:
            away_score += 1

        # Home advantage
        home_score += 0.5

        if home_score > away_score + 1:
            return "主队战术优势明显"
        elif home_score > away_score:
            return "主队略有优势"
        elif away_score > home_score + 1:
            return "客队战术优势明显"
        elif away_score > home_score:
            return "客队略有优势"
        else:
            return "战术层面势均力敌"

    def _predict_changes(self, home: TeamProfile, away: TeamProfile) -> str:
        """Predict tactical changes."""
        parts = []
        if home.form_last_5.count("L") >= 2:
            parts.append("主队可能调整战术")
        if away.form_last_5.count("L") >= 2:
            parts.append("客队可能加强防守")
        if not parts:
            parts.append("预计双方按常规战术出战")
        return "；".join(parts)

    def _assess_probability_impact(self, home: TeamProfile, away: TeamProfile) -> str:
        """Assess tactical impact on probabilities."""
        elo_diff = home.elo_rating - away.elo_rating
        if elo_diff > 150:
            return "战术层面支持主队胜出"
        elif elo_diff < -150:
            return "战术层面支持客队胜出"
        else:
            return "战术层面对概率影响有限"
