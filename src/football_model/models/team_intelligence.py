from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamSnapshot:
    team: str
    elo: float
    elo_rank: int
    matches: int
    recent_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: float
    goals_against: float
    points_per_game: float
    form: str


@dataclass(frozen=True)
class MatchIntelligence:
    home: TeamSnapshot
    away: TeamSnapshot
    home_xg: float
    away_xg: float
    elo_home_probability: float
    elo_draw_probability: float
    elo_away_probability: float
    league_home_goals: float
    league_away_goals: float
    data_cutoff: pd.Timestamp


def _snapshot(team: str, ratings: dict[str, float], histories: dict[str, list[dict[str, float]]]) -> TeamSnapshot:
    history = histories.get(team, [])
    recent = history[-8:]
    if recent:
        weights = np.linspace(0.65, 1.0, len(recent))
        # Vectorized calculation for better performance
        gf_array = np.array([row["gf"] for row in recent])
        ga_array = np.array([row["ga"] for row in recent])
        points_array = np.array([row["points"] for row in recent])
        goals_for = float(np.average(gf_array, weights=weights))
        goals_against = float(np.average(ga_array, weights=weights))
        points_per_game = float(np.average(points_array, weights=weights))
    else:
        goals_for = goals_against = points_per_game = 0.0
    # Vectorized counting for better performance
    points_array = np.array([row["points"] for row in recent]) if recent else np.array([])
    wins = int(np.sum(points_array == 3))
    draws = int(np.sum(points_array == 1))
    losses = len(recent) - wins - draws
    symbols = {3.0: "W", 1.0: "D", 0.0: "L"}
    form = "-".join(symbols[row["points"]] for row in recent[-5:]) if recent else "无数据"
    ranking = sorted(ratings, key=ratings.get, reverse=True)
    return TeamSnapshot(
        team=team,
        elo=float(ratings.get(team, 1500.0)),
        elo_rank=ranking.index(team) + 1 if team in ranking else len(ranking) + 1,
        matches=len(history),
        recent_matches=len(recent),
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        points_per_game=points_per_game,
        form=form,
    )


def build_match_intelligence(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    as_of: object | None = None,
    neutral_venue: bool = False,
) -> MatchIntelligence:
    required = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"状态模型缺少字段：{', '.join(sorted(missing))}")
    data = frame.dropna(subset=list(required)).copy()
    data["kickoff"] = pd.to_datetime(data["kickoff"])
    if as_of is not None:
        data = data.loc[data["kickoff"] < pd.to_datetime(as_of)]
    data = data.sort_values("kickoff")
    if data.empty:
        raise ValueError("比赛时间之前没有可用历史数据")

    ratings: dict[str, float] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    home_advantage = 0.0 if neutral_venue else 55.0
    for row in data.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_rating = ratings.setdefault(home, 1500.0)
        away_rating = ratings.setdefault(away, 1500.0)
        expected_home = 1 / (1 + 10 ** (-(home_rating + home_advantage - away_rating) / 400))
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        goal_multiplier = 1.0 + min(abs(home_goals - away_goals), 3) * 0.12
        change = 24.0 * goal_multiplier * (actual_home - expected_home)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change
        home_points = 3.0 if actual_home == 1 else 1.0 if actual_home == 0.5 else 0.0
        away_points = 3.0 if actual_home == 0 else 1.0 if actual_home == 0.5 else 0.0
        histories.setdefault(home, []).append({"gf": home_goals, "ga": away_goals, "points": home_points})
        histories.setdefault(away, []).append({"gf": away_goals, "ga": home_goals, "points": away_points})

    home = _snapshot(home_team, ratings, histories)
    away = _snapshot(away_team, ratings, histories)
    recent_league = data.tail(min(len(data), 700))
    league_home_goals = float(recent_league["home_goals"].mean())
    league_away_goals = float(recent_league["away_goals"].mean())
    league_team_goals = max((league_home_goals + league_away_goals) / 2, 0.2)

    home_attack = max(home.goals_for / league_team_goals, 0.35)
    away_defence = max(away.goals_against / league_team_goals, 0.35)
    away_attack = max(away.goals_for / league_team_goals, 0.35)
    home_defence = max(home.goals_against / league_team_goals, 0.35)
    elo_delta = home.elo + home_advantage - away.elo
    home_elo_factor = float(np.clip(np.exp(elo_delta / 900), 0.72, 1.38))
    away_elo_factor = float(np.clip(np.exp(-elo_delta / 900), 0.72, 1.38))
    home_xg = league_home_goals * np.sqrt(home_attack * away_defence) * home_elo_factor
    away_xg = league_away_goals * np.sqrt(away_attack * home_defence) * away_elo_factor

    decisive_home = 1 / (1 + 10 ** (-elo_delta / 400))
    draw_probability = float(np.clip(0.29 - abs(elo_delta) / 1800, 0.16, 0.29))
    elo_home_probability = float((1 - draw_probability) * decisive_home)
    elo_away_probability = float(1 - draw_probability - elo_home_probability)
    return MatchIntelligence(
        home=home,
        away=away,
        home_xg=float(np.clip(home_xg, 0.2, 4.2)),
        away_xg=float(np.clip(away_xg, 0.2, 4.2)),
        elo_home_probability=elo_home_probability,
        elo_draw_probability=draw_probability,
        elo_away_probability=elo_away_probability,
        league_home_goals=league_home_goals,
        league_away_goals=league_away_goals,
        data_cutoff=pd.Timestamp(data["kickoff"].max()),
    )
