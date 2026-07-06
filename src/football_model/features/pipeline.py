"""Feature engineering pipeline for football match prediction.

All features use strict cutoff_at to prevent future information leakage.
Uses vectorized pandas rolling operations for O(n) performance.

Key design: every feature must have meaningful variance across matches.
Constant features (league-level averages) are excluded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Time-aware feature generation for football match prediction."""

    required_columns = {"kickoff", "home_team", "away_team", "cutoff_at"}

    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 10,
        long_window: int = 20,
        min_matches: int = 3,
    ) -> None:
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.min_matches = min_matches

    def validate_input(self, frame: pd.DataFrame) -> None:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"特征输入缺少字段：{', '.join(sorted(missing))}")
        kickoff = pd.to_datetime(frame["kickoff"])
        cutoff = pd.to_datetime(frame["cutoff_at"])
        if (cutoff > kickoff).any():
            raise ValueError("cutoff_at 不能晚于比赛开球时间")

    def transform(
        self,
        frame: pd.DataFrame,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate features for matches using historical data."""
        required_history = {"kickoff", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required_history - set(history.columns)
        if missing:
            raise ValueError(f"历史数据缺少字段：{', '.join(sorted(missing))}")

        self.validate_input(frame)

        frame = frame.copy().reset_index(drop=True)
        history = history.copy().sort_values("kickoff").reset_index(drop=True)

        frame["kickoff"] = pd.to_datetime(frame["kickoff"])
        frame["cutoff_at"] = pd.to_datetime(frame["cutoff_at"])
        history["kickoff"] = pd.to_datetime(history["kickoff"])

        team_stats = self._build_team_rolling_stats(history)
        elo_ratings = self._compute_elo_ratings(history)
        elo_dict = self._build_elo_lookup(elo_ratings)

        feature_rows = []
        for _, match in frame.iterrows():
            cutoff = match["cutoff_at"]
            home = match["home_team"]
            away = match["away_team"]

            row = {}
            row.update(self._lookup_team_features(team_stats, home, cutoff, "home"))
            row.update(self._lookup_team_features(team_stats, away, cutoff, "away"))

            home_elo = self._lookup_elo_fast(elo_dict, home, cutoff)
            away_elo = self._lookup_elo_fast(elo_dict, away, cutoff)
            row["home_elo"] = home_elo
            row["away_elo"] = away_elo
            row["elo_diff"] = home_elo - away_elo

            row.update(self._compute_h2h_fast(history, home, away, cutoff))
            row["home_rest_days"] = self._rest_days(team_stats, home, cutoff)
            row["away_rest_days"] = self._rest_days(team_stats, away, cutoff)
            row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]

            feature_rows.append(row)

        features_df = pd.DataFrame(feature_rows)
        result = pd.concat([frame, features_df], axis=1)
        logger.info(f"Generated {len(feature_rows)} feature rows with {len(features_df.columns)} features")
        return result

    def _build_team_rolling_stats(self, history: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if history.empty:
            return {}

        home_rows = history[["kickoff", "home_team", "away_team", "home_goals", "away_goals"]].copy()
        home_rows.columns = ["match_date", "team", "opponent", "gf", "ga"]
        home_rows["is_home"] = True
        home_rows["points"] = np.where(
            home_rows["gf"] > home_rows["ga"], 3, np.where(home_rows["gf"] == home_rows["ga"], 1, 0)
        )

        away_rows = history[["kickoff", "away_team", "home_team", "away_goals", "home_goals"]].copy()
        away_rows.columns = ["match_date", "team", "opponent", "gf", "ga"]
        away_rows["is_home"] = False
        away_rows["points"] = np.where(
            away_rows["gf"] > away_rows["ga"], 3, np.where(away_rows["gf"] == away_rows["ga"], 1, 0)
        )

        all_rows = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("match_date")

        stats = {}
        for team, group in all_rows.groupby("team"):
            g = group.sort_values("match_date").reset_index(drop=True)
            g["cum_gf"] = g["gf"].expanding().sum()
            g["cum_ga"] = g["ga"].expanding().sum()
            g["cum_points"] = g["points"].expanding().sum()
            g["n"] = np.arange(1, len(g) + 1)
            g["win"] = (g["points"] == 3).astype(float)
            g["draw"] = (g["points"] == 1).astype(float)
            g["loss"] = (g["points"] == 0).astype(float)
            g["cs"] = (g["ga"] == 0).astype(float)
            g["btts"] = ((g["gf"] > 0) & (g["ga"] > 0)).astype(float)
            g["over25"] = ((g["gf"] + g["ga"]) > 2.5).astype(float)
            g["over15"] = ((g["gf"] + g["ga"]) > 1.5).astype(float)
            g["total_goals"] = g["gf"] + g["ga"]
            stats[team] = g
        return stats

    def _lookup_team_features(
        self,
        team_stats: dict[str, pd.DataFrame],
        team: str,
        cutoff: pd.Timestamp,
        prefix: str,
    ) -> dict[str, float]:
        defaults = self._default_team_features(prefix)

        if team not in team_stats:
            return defaults

        df = team_stats[team]
        mask = df["match_date"] < cutoff
        available = df[mask]

        if len(available) < self.min_matches:
            return defaults

        short = available.tail(self.short_window)
        medium = available.tail(self.medium_window)
        n = len(available)
        latest = available.iloc[-1]

        # Overall stats (expanding window)
        ppg = latest["cum_points"] / latest["n"]
        gf_pg = latest["cum_gf"] / latest["n"]
        ga_pg = latest["cum_ga"] / latest["n"]

        features = {
            f"{prefix}_ppg": float(ppg),
            f"{prefix}_gf_pg": float(gf_pg),
            f"{prefix}_ga_pg": float(ga_pg),
            f"{prefix}_gd_pg": float(gf_pg - ga_pg),
            f"{prefix}_win_rate": float(available["win"].mean()),
            f"{prefix}_loss_rate": float(available["loss"].mean()),
            f"{prefix}_cs_rate": float(available["cs"].mean()),
            f"{prefix}_btts_rate": float(available["btts"].mean()),
            f"{prefix}_over25_rate": float(available["over25"].mean()),
            f"{prefix}_avg_goals": float(available["total_goals"].mean()),
            f"{prefix}_matches": float(n),
        }

        # Short-term form (last 5)
        if len(short) >= self.min_matches:
            features[f"{prefix}_short_ppg"] = float(short["points"].mean())
            features[f"{prefix}_short_gf_pg"] = float(short["gf"].mean())
            features[f"{prefix}_short_ga_pg"] = float(short["ga"].mean())
            features[f"{prefix}_short_gd"] = float(short["gf"].mean() - short["ga"].mean())

        # Medium-term form (last 10)
        if len(medium) >= self.min_matches:
            features[f"{prefix}_med_ppg"] = float(medium["points"].mean())

        # Venue-specific performance
        venue = available[available["is_home"]] if prefix == "home" else available[~available["is_home"]]

        if len(venue) >= self.min_matches:
            features[f"{prefix}_venue_ppg"] = float(venue["points"].mean())
            features[f"{prefix}_venue_gf_pg"] = float(venue["gf"].mean())
            features[f"{prefix}_venue_ga_pg"] = float(venue["ga"].mean())
        else:
            features[f"{prefix}_venue_ppg"] = features[f"{prefix}_ppg"]
            features[f"{prefix}_venue_gf_pg"] = features[f"{prefix}_gf_pg"]
            features[f"{prefix}_venue_ga_pg"] = features[f"{prefix}_ga_pg"]

        # Momentum (last 3 vs previous 3)
        if len(available) >= 6:
            recent3 = available.tail(3)["points"].mean()
            prev3 = available.iloc[-6:-3]["points"].mean()
            features[f"{prefix}_momentum"] = float(recent3 - prev3)

        # Win/loss streak
        features[f"{prefix}_streak"] = self._compute_streak(available)

        # Scoring consistency (std of goals scored)
        if len(short) >= self.min_matches:
            features[f"{prefix}_gf_std"] = float(short["gf"].std()) if len(short) > 1 else 0.0

        # Goal difference trend
        if len(available) >= 6:
            recent_gd = available.tail(3)["gf"].mean() - available.tail(3)["ga"].mean()
            prev_gd = available.iloc[-6:-3]["gf"].mean() - available.iloc[-6:-3]["ga"].mean()
            features[f"{prefix}_gd_trend"] = float(recent_gd - prev_gd)

        return features

    def _compute_streak(self, available: pd.DataFrame) -> float:
        """Compute current streak: positive = wins, negative = losses."""
        if available.empty:
            return 0.0
        streak = 0
        for _, row in available.iloc[::-1].iterrows():
            if row["points"] == 3:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif row["points"] == 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        return float(streak)

    def _build_elo_lookup(self, elo_df: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, float]]]:
        """Build a lookup dict for Elo ratings."""
        lookup: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        for _, row in elo_df.iterrows():
            team = row["team"]
            if team not in lookup:
                lookup[team] = []
            lookup[team].append((row["match_date"], row["elo"]))
        return lookup

    def _lookup_elo_fast(self, elo_dict: dict, team: str, cutoff: pd.Timestamp) -> float:
        entries = elo_dict.get(team, [])
        if not entries:
            return 1500.0
        # Binary search for last entry before cutoff
        lo, hi = 0, len(entries) - 1
        result = 1500.0
        while lo <= hi:
            mid = (lo + hi) // 2
            if entries[mid][0] < cutoff:
                result = entries[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def _compute_elo_ratings(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty:
            return pd.DataFrame(columns=["match_date", "team", "elo"])
        ratings: dict[str, float] = {}
        rows = []
        k = 32.0
        ha = 50.0

        for _, m in history.iterrows():
            home = m["home_team"]
            away = m["away_team"]
            hr = ratings.get(home, 1500.0)
            ar = ratings.get(away, 1500.0)

            expected_h = 1.0 / (1.0 + 10 ** ((ar - hr - ha) / 400))

            if m["home_goals"] > m["away_goals"]:
                actual = 1.0
            elif m["home_goals"] == m["away_goals"]:
                actual = 0.5
            else:
                actual = 0.0

            gd = abs(int(m["home_goals"]) - int(m["away_goals"]))
            mult = np.log(max(gd, 1) + 1)
            change = k * mult * (actual - expected_h)

            ratings[home] = hr + change
            ratings[away] = ar - change

            rows.append({"match_date": m["kickoff"], "team": home, "elo": ratings[home]})
            rows.append({"match_date": m["kickoff"], "team": away, "elo": ratings[away]})

        return pd.DataFrame(rows).sort_values("match_date")

    def _compute_h2h_fast(
        self,
        history: pd.DataFrame,
        home: str,
        away: str,
        cutoff: pd.Timestamp,
    ) -> dict[str, float]:
        mask = (history["kickoff"] < cutoff) & (
            ((history["home_team"] == home) & (history["away_team"] == away))
            | ((history["home_team"] == away) & (history["away_team"] == home))
        )
        h2h = history[mask].tail(10)

        if h2h.empty:
            return {"h2h_matches": 0, "h2h_home_wins": 0.0, "h2h_draws": 0.0, "h2h_avg_goals": 2.5}

        home_wins = ((h2h["home_team"] == home) & (h2h["home_goals"] > h2h["away_goals"])).sum()
        draws = (h2h["home_goals"] == h2h["away_goals"]).sum()
        total_goals = (h2h["home_goals"] + h2h["away_goals"]).mean()

        return {
            "h2h_matches": len(h2h),
            "h2h_home_wins": float(home_wins / len(h2h)),
            "h2h_draws": float(draws / len(h2h)),
            "h2h_avg_goals": float(total_goals),
        }

    def _rest_days(self, team_stats: dict[str, pd.DataFrame], team: str, cutoff: pd.Timestamp) -> float:
        if team not in team_stats:
            return 7.0
        df = team_stats[team]
        mask = df["match_date"] < cutoff
        available = df[mask]
        if available.empty:
            return 7.0
        last_match = available.iloc[-1]["match_date"]
        delta = (cutoff - last_match).total_seconds() / 86400
        return float(min(delta, 30.0))

    def _default_team_features(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_ppg": 1.5,
            f"{prefix}_gf_pg": 1.3,
            f"{prefix}_ga_pg": 1.3,
            f"{prefix}_gd_pg": 0.0,
            f"{prefix}_win_rate": 0.33,
            f"{prefix}_loss_rate": 0.33,
            f"{prefix}_cs_rate": 0.25,
            f"{prefix}_btts_rate": 0.50,
            f"{prefix}_over25_rate": 0.50,
            f"{prefix}_avg_goals": 2.6,
            f"{prefix}_matches": 0.0,
            f"{prefix}_short_ppg": 1.5,
            f"{prefix}_short_gf_pg": 1.3,
            f"{prefix}_short_ga_pg": 1.3,
            f"{prefix}_short_gd": 0.0,
            f"{prefix}_med_ppg": 1.5,
            f"{prefix}_venue_ppg": 1.5,
            f"{prefix}_venue_gf_pg": 1.3,
            f"{prefix}_venue_ga_pg": 1.3,
            f"{prefix}_momentum": 0.0,
            f"{prefix}_streak": 0.0,
            f"{prefix}_gf_std": 0.8,
            f"{prefix}_gd_trend": 0.0,
        }


def get_feature_names() -> list[str]:
    """Return list of all feature names."""
    defaults = {}
    defaults.update(FeaturePipeline()._default_team_features("home"))
    defaults.update(FeaturePipeline()._default_team_features("away"))
    defaults.update(
        {
            "home_elo": 1500.0,
            "away_elo": 1500.0,
            "elo_diff": 0.0,
            "h2h_matches": 0,
            "h2h_home_wins": 0.0,
            "h2h_draws": 0.0,
            "h2h_avg_goals": 2.5,
            "home_rest_days": 7.0,
            "away_rest_days": 7.0,
            "rest_days_diff": 0.0,
        }
    )
    return sorted(defaults.keys())
