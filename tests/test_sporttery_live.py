from __future__ import annotations

import logging
import numpy as np

from football_model.data.adapters import SportteryAdapter
from football_model.engine import infer_expected_goals_from_market, market_comparison, score_matrix, summarize_market

logger = logging.getLogger(__name__)


def test_sporttery_adapter_normalizes_live_match(monkeypatch) -> None:
    """Test that sporttery adapter normalizes live match data correctly."""
    logger.info("Testing sporttery adapter normalization")
    payload = {
        "errorCode": "0",
        "value": {
            "lastUpdateTime": "2026-07-05 18:32:11",
            "totalCount": 1,
            "matchInfoList": [
                {
                    "subMatchList": [
                        {
                            "matchId": 123,
                            "businessDate": "2026-07-05",
                            "matchNumStr": "周日001",
                            "matchDate": "2026-07-05",
                            "matchTime": "20:00",
                            "weekday": "周日",
                            "leagueId": "58",
                            "leagueAllName": "瑞典超级联赛",
                            "homeTeamId": 1,
                            "homeTeamAllName": "主队",
                            "awayTeamId": 2,
                            "awayTeamAllName": "客队",
                            "sellStatus": "1",
                            "matchStatus": "Selling",
                            "remark": "",
                            "poolList": [
                                {"poolCode": "HAD", "cbtSingle": 1},
                                {"poolCode": "HHAD", "cbtSingle": 0},
                            ],
                            "oddsList": [
                                {"poolCode": "HAD", "h": "1.80", "d": "3.20", "a": "4.10"},
                                {
                                    "poolCode": "HHAD",
                                    "h": "3.10",
                                    "d": "3.30",
                                    "a": "2.00",
                                    "goalLine": "-1.00",
                                },
                            ],
                        }
                    ]
                }
            ],
        },
    }
    adapter = SportteryAdapter()
    monkeypatch.setattr(adapter, "_request_json", lambda: payload)
    snapshot = adapter.fetch()
    assert len(snapshot.matches) == 1
    assert len(snapshot.odds) == 6
    assert snapshot.matches.iloc[0]["had_single"]
    assert snapshot.odds.loc[snapshot.odds["market"] == "HHAD", "goal_line"].iloc[0] == "-1.00"


def test_market_prior_reproduces_normalized_probabilities() -> None:
    odds = {"主胜": 1.8, "平局": 3.4, "客胜": 4.5}
    home_xg, away_xg = infer_expected_goals_from_market(1.8, 3.4, 4.5)
    summary = summarize_market(score_matrix(home_xg, away_xg), home_xg, away_xg)
    comparison = market_comparison(summary.probabilities, odds)
    assert np.max(np.abs(comparison["概率差"].to_numpy())) < 0.015
