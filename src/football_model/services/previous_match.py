"""Previous match analysis service.

Analyzes the most recent match for each team to provide
context for upcoming matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviousMatchReport:
    """Analysis of a team's most recent match."""

    team_name: str
    opponent: str
    venue: str  # home / away
    match_date: str
    score: str
    result: str  # W / D / L
    goals_for: int
    goals_against: int
    goal_diff: int
    is_clean_sheet: bool
    is_btts: bool
    is_over25: bool
    days_ago: int
    fatigue_level: str  # low / medium / high
    impact_on_next: str


class PreviousMatchService:
    """Service for analyzing previous matches."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def get_previous_match(
        self,
        team_name: str,
        league: str,
        before: pd.Timestamp,
    ) -> PreviousMatchReport | None:
        """Get analysis of team's most recent match.

        Args:
            team_name: Team name
            league: Competition name
            before: Cutoff date

        Returns:
            PreviousMatchReport or None if no match found
        """
        from football_model.data.repositories import MatchRepository

        training = MatchRepository(self.database).training_frame(league)
        if training.empty:
            return None

        # Filter matches before cutoff
        training["kickoff"] = pd.to_datetime(training["kickoff"])
        before_dt = pd.to_datetime(before)

        team_matches = training[
            ((training["home_team"] == team_name) | (training["away_team"] == team_name))
            & (training["kickoff"] < before_dt)
        ].sort_values("kickoff", ascending=False)

        if team_matches.empty:
            return None

        match = team_matches.iloc[0]
        is_home = match["home_team"] == team_name
        opponent = match["away_team"] if is_home else match["home_team"]
        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        if is_home:
            goals_for = hg
            goals_against = ag
        else:
            goals_for = ag
            goals_against = hg

        # Result
        if goals_for > goals_against:
            result = "W"
        elif goals_for == goals_against:
            result = "D"
        else:
            result = "L"

        # Days ago
        match_date = pd.to_datetime(match["kickoff"])
        days_ago = max(0, (before_dt - match_date).days)

        # Fatigue level
        if days_ago <= 2:
            fatigue = "high"
        elif days_ago <= 4:
            fatigue = "medium"
        else:
            fatigue = "low"

        # Impact assessment
        impact = self._assess_impact(result, goals_for, goals_against, days_ago)

        return PreviousMatchReport(
            team_name=team_name,
            opponent=opponent,
            venue="主" if is_home else "客",
            match_date=match_date.strftime("%Y-%m-%d"),
            score=f"{goals_for}:{goals_against}",
            result=result,
            goals_for=goals_for,
            goals_against=goals_against,
            goal_diff=goals_for - goals_against,
            is_clean_sheet=goals_against == 0,
            is_btts=goals_for > 0 and goals_against > 0,
            is_over25=goals_for + goals_against > 2,
            days_ago=days_ago,
            fatigue_level=fatigue,
            impact_on_next=impact,
        )

    def get_both_previous_matches(
        self,
        home_team: str,
        away_team: str,
        league: str,
        before: pd.Timestamp,
    ) -> tuple[PreviousMatchReport | None, PreviousMatchReport | None]:
        """Get previous match reports for both teams."""
        home_prev = self.get_previous_match(home_team, league, before)
        away_prev = self.get_previous_match(away_team, league, before)
        return home_prev, away_prev

    def _assess_impact(self, result: str, gf: int, ga: int, days_ago: int) -> str:
        """Assess impact on next match."""
        parts = []

        # Result momentum
        if result == "W":
            parts.append("上场获胜，士气高涨")
        elif result == "D":
            parts.append("上场平局，状态稳定")
        else:
            parts.append("上场失利，需调整心态")

        # Goal scoring
        if gf >= 3:
            parts.append("进攻状态火热")
        elif gf == 0:
            parts.append("进攻端需要改善")

        # Defense
        if ga == 0:
            parts.append("防守稳固")
        elif ga >= 3:
            parts.append("防守存在隐患")

        # Fatigue
        if days_ago <= 2:
            parts.append("体能恢复时间不足")
        elif days_ago >= 7:
            parts.append("充分休息")

        return "；".join(parts) if parts else "无特殊影响"
