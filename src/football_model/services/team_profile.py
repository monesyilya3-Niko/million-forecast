"""Team profile service.

Generates and manages team profile data including
form, strength ratings, and tactical style indicators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_model.data import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamProfile:
    """Team profile with form, strength, and style indicators."""

    team_name: str
    league: str
    ranking: int | None = None
    points: int | None = None
    goals_for: float = 0.0
    goals_against: float = 0.0
    xg_for: float | None = None
    xg_against: float | None = None
    home_strength: float = 1.0
    away_strength: float = 1.0
    form_last_5: str = ""
    form_last_10: str = ""
    elo_rating: float = 1500.0
    attack_style: str = "balanced"
    defense_style: str = "balanced"
    set_piece_strength: float = 0.5
    transition_strength: float = 0.5
    pressing_level: str = "medium"
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    home_wins: int = 0
    home_draws: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_draws: int = 0
    away_losses: int = 0
    clean_sheets: int = 0
    btts: int = 0
    over25: int = 0


class TeamProfileService:
    """Service for generating team profiles from match history."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def get_team_profile(
        self,
        team_name: str,
        league: str,
        as_of: pd.Timestamp | None = None,
    ) -> TeamProfile:
        """Generate team profile from match history.

        Args:
            team_name: Team name (training data name)
            league: Competition name
            as_of: Cutoff date (use matches before this date)

        Returns:
            TeamProfile with computed statistics
        """
        training = self._get_training_data(league, as_of)
        if training.empty:
            return TeamProfile(team_name=team_name, league=league)

        # Filter matches for this team
        team_matches = training[
            (training["home_team"] == team_name) | (training["away_team"] == team_name)
        ].copy()

        if team_matches.empty:
            return TeamProfile(team_name=team_name, league=league)

        team_matches = team_matches.sort_values("kickoff")

        # Compute basic stats
        stats = self._compute_basic_stats(team_name, team_matches)

        # Compute form
        form_5 = self._compute_form(team_name, team_matches.tail(5))
        form_10 = self._compute_form(team_name, team_matches.tail(10))

        # Compute home/away splits
        home_matches = team_matches[team_matches["home_team"] == team_name]
        away_matches = team_matches[team_matches["away_team"] == team_name]

        home_stats = self._compute_venue_stats(team_name, home_matches, "home")
        away_stats = self._compute_venue_stats(team_name, away_matches, "away")

        # Compute Elo
        elo = self._compute_elo(team_name, training, as_of)

        # Compute strength ratings
        league_avg = self._compute_league_averages(training)
        home_strength = self._compute_strength(team_name, home_matches, "home", league_avg)
        away_strength = self._compute_strength(team_name, away_matches, "away", league_avg)

        return TeamProfile(
            team_name=team_name,
            league=league,
            points=stats["points"],
            goals_for=stats["goals_for_pg"],
            goals_against=stats["goals_against_pg"],
            home_strength=home_strength,
            away_strength=away_strength,
            form_last_5=form_5,
            form_last_10=form_10,
            elo_rating=elo,
            matches_played=stats["matches"],
            wins=stats["wins"],
            draws=stats["draws"],
            losses=stats["losses"],
            home_wins=home_stats["wins"],
            home_draws=home_stats["draws"],
            home_losses=home_stats["losses"],
            away_wins=away_stats["wins"],
            away_draws=away_stats["draws"],
            away_losses=away_stats["losses"],
            clean_sheets=stats["clean_sheets"],
            btts=stats["btts"],
            over25=stats["over25"],
        )

    def get_h2h(
        self,
        home_team: str,
        away_team: str,
        league: str,
        as_of: pd.Timestamp | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get head-to-head record between two teams."""
        training = self._get_training_data(league, as_of)
        if training.empty:
            return pd.DataFrame()

        teams = {home_team, away_team}
        h2h = training[
            training["home_team"].isin(teams) & training["away_team"].isin(teams)
        ].sort_values("kickoff").tail(limit)

        if h2h.empty:
            return pd.DataFrame()

        rows = []
        for _, m in h2h.iterrows():
            rows.append({
                "日期": pd.to_datetime(m["kickoff"]).strftime("%Y-%m-%d"),
                "主队": m["home_team"],
                "客队": m["away_team"],
                "比分": f"{int(m['home_goals'])}:{int(m['away_goals'])}",
                "主队进球": int(m["home_goals"]),
                "客队进球": int(m["away_goals"]),
            })

        return pd.DataFrame(rows)

    def _get_training_data(self, league: str, as_of: pd.Timestamp | None) -> pd.DataFrame:
        """Get training data for a league."""
        from football_model.data.repositories import MatchRepository
        data = MatchRepository(self.database).training_frame(league)
        if data.empty:
            return data
        if as_of is not None:
            data = data[pd.to_datetime(data["kickoff"]) < as_of]
        return data

    def _compute_basic_stats(self, team: str, matches: pd.DataFrame) -> dict:
        """Compute basic team statistics."""
        wins = 0
        draws = 0
        losses = 0
        gf = 0
        ga = 0
        clean_sheets = 0
        btts = 0
        over25 = 0

        for _, m in matches.iterrows():
            is_home = m["home_team"] == team
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])

            if is_home:
                gf += hg
                ga += ag
                if hg > ag:
                    wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    losses += 1
            else:
                gf += ag
                ga += hg
                if ag > hg:
                    wins += 1
                elif ag == hg:
                    draws += 1
                else:
                    losses += 1

            # Clean sheet
            if (is_home and ag == 0) or (not is_home and hg == 0):
                clean_sheets += 1

            # BTTS
            if hg > 0 and ag > 0:
                btts += 1

            # Over 2.5
            if hg + ag > 2:
                over25 += 1

        n = len(matches)
        return {
            "matches": n,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": wins * 3 + draws,
            "goals_for_pg": gf / max(n, 1),
            "goals_against_pg": ga / max(n, 1),
            "clean_sheets": clean_sheets,
            "btts": btts,
            "over25": over25,
        }

    def _compute_form(self, team: str, matches: pd.DataFrame) -> str:
        """Compute form string (W/D/L sequence)."""
        if matches.empty:
            return ""

        symbols = []
        for _, m in matches.iterrows():
            is_home = m["home_team"] == team
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])

            if is_home:
                if hg > ag:
                    symbols.append("W")
                elif hg == ag:
                    symbols.append("D")
                else:
                    symbols.append("L")
            else:
                if ag > hg:
                    symbols.append("W")
                elif ag == hg:
                    symbols.append("D")
                else:
                    symbols.append("L")

        return "-".join(symbols)

    def _compute_venue_stats(self, team: str, matches: pd.DataFrame, venue: str) -> dict:
        """Compute venue-specific stats."""
        wins = 0
        draws = 0
        losses = 0

        for _, m in matches.iterrows():
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])

            if venue == "home":
                if hg > ag:
                    wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    losses += 1
            else:
                if ag > hg:
                    wins += 1
                elif ag == hg:
                    draws += 1
                else:
                    losses += 1

        return {"wins": wins, "draws": draws, "losses": losses}

    def _compute_elo(self, team: str, training: pd.DataFrame, as_of: pd.Timestamp | None) -> float:
        """Compute Elo rating for a team."""
        ratings: dict[str, float] = {}
        k = 32.0
        ha = 50.0

        for _, m in training.iterrows():
            home = m["home_team"]
            away = m["away_team"]
            hr = ratings.get(home, 1500.0)
            ar = ratings.get(away, 1500.0)

            expected_h = 1.0 / (1.0 + 10 ** ((ar - hr - ha) / 400))

            hg = int(m["home_goals"])
            ag = int(m["away_goals"])

            if hg > ag:
                actual = 1.0
            elif hg == ag:
                actual = 0.5
            else:
                actual = 0.0

            gd = abs(hg - ag)
            mult = np.log(max(gd, 1) + 1)
            change = k * mult * (actual - expected_h)

            ratings[home] = hr + change
            ratings[away] = ar - change

        return ratings.get(team, 1500.0)

    def _compute_league_averages(self, training: pd.DataFrame) -> dict:
        """Compute league average goals."""
        home_avg = float(training["home_goals"].mean())
        away_avg = float(training["away_goals"].mean())
        return {"home_avg": max(home_avg, 0.5), "away_avg": max(away_avg, 0.5)}

    def _compute_strength(
        self,
        team: str,
        matches: pd.DataFrame,
        venue: str,
        league_avg: dict,
    ) -> float:
        """Compute strength rating relative to league average."""
        if matches.empty:
            return 1.0

        gf = 0
        ga = 0

        for _, m in matches.iterrows():
            if venue == "home":
                gf += int(m["home_goals"])
                ga += int(m["away_goals"])
            else:
                gf += int(m["away_goals"])
                ga += int(m["home_goals"])

        n = len(matches)
        gf_pg = gf / max(n, 1)
        ga_pg = ga / max(n, 1)

        if venue == "home":
            avg_gf = league_avg["home_avg"]
            avg_ga = league_avg["away_avg"]
        else:
            avg_gf = league_avg["away_avg"]
            avg_ga = league_avg["home_avg"]

        attack = gf_pg / max(avg_gf, 0.1)
        defense = ga_pg / max(avg_ga, 0.1)

        return float(np.clip((attack + (2 - defense)) / 2, 0.3, 2.0))
