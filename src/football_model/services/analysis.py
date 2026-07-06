from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_model.engine import (
    MarketSummary,
    handicap_probabilities,
    market_comparison,
    score_matrix,
    summarize_market,
)


@dataclass(frozen=True)
class AnalysisResult:
    matrix: np.ndarray
    summary: MarketSummary
    comparison: pd.DataFrame
    handicap: dict[str, float]


class AnalysisService:
    model_version = "dixon-coles-baseline:0.1.0"

    def analyze(
        self,
        *,
        home_xg: float,
        away_xg: float,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        handicap: int = 0,
        rho: float = -0.08,
    ) -> AnalysisResult:
        matrix = score_matrix(home_xg, away_xg, rho=rho)
        summary = summarize_market(matrix, home_xg, away_xg)
        comparison = market_comparison(
            summary.probabilities,
            {"主胜": odds_home, "平局": odds_draw, "客胜": odds_away},
        )
        return AnalysisResult(
            matrix=matrix,
            summary=summary,
            comparison=comparison,
            handicap=handicap_probabilities(matrix, handicap),
        )

    def analyze_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for item in frame.itertuples(index=False):
            analysis = self.analyze(
                home_xg=float(item.home_xg),
                away_xg=float(item.away_xg),
                odds_home=float(item.odds_home),
                odds_draw=float(item.odds_draw),
                odds_away=float(item.odds_away),
            )
            best = analysis.comparison.loc[analysis.comparison["理论EV"].idxmax()]
            rows.append(
                {
                    "主队": item.home_team,
                    "客队": item.away_team,
                    "主胜概率": analysis.summary.home_win,
                    "平局概率": analysis.summary.draw,
                    "客胜概率": analysis.summary.away_win,
                    "最高概率比分": analysis.summary.top_scores[0][0],
                    "价值方向": best["结果"],
                    "最高理论EV": best["理论EV"],
                    "模型版本": self.model_version,
                }
            )
        return pd.DataFrame(rows)
