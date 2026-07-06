import logging
import pandas as pd

from football_model.ui.pages.match_analysis import (
    TEAM_MAPS,
    _competition_for_league,
    _independent_weight,
    _market_movement,
)

logger = logging.getLogger(__name__)


def test_only_today_competitions_are_routed_to_independent_models() -> None:
    """Test that only today's competitions are routed to independent models."""
    logger.info("Testing competition routing")
    assert _competition_for_league("世界杯") == "世界杯国家队"
    assert _competition_for_league("瑞典超级联赛") == "瑞典超级联赛"
    assert _competition_for_league("英格兰超级联赛") is None


def test_today_team_names_map_to_training_names() -> None:
    assert TEAM_MAPS["世界杯国家队"]["巴西"] == "Brazil"
    assert TEAM_MAPS["世界杯国家队"]["英格兰"] == "England"
    assert TEAM_MAPS["瑞典超级联赛"]["卡尔马"] == "Kalmar"
    assert TEAM_MAPS["瑞典超级联赛"]["哈马比"] == "Hammarby"
    assert TEAM_MAPS["世界杯国家队"]["法国"] == "France"
    assert TEAM_MAPS["世界杯国家队"]["摩洛哥"] == "Morocco"
    assert TEAM_MAPS["瑞典超级联赛"]["赫根"] == "Hacken"


def test_validation_metrics_control_independent_weight() -> None:
    weak = _independent_weight("瑞典超级联赛", {"holdout_log_loss": 1.16})
    strong = _independent_weight("瑞典超级联赛", {"holdout_log_loss": 0.90})
    assert strong > weak
    assert 0.5 <= weak <= 0.7
    assert 0.5 <= strong <= 0.7


def test_market_movement_uses_normalized_implied_probability() -> None:
    frame = pd.DataFrame(
        [
            {"captured_at": "2026-07-05 10:00", "market": "HAD", "selection": "H", "odds": 2.00},
            {"captured_at": "2026-07-05 10:00", "market": "HAD", "selection": "D", "odds": 3.20},
            {"captured_at": "2026-07-05 10:00", "market": "HAD", "selection": "A", "odds": 3.80},
            {"captured_at": "2026-07-05 11:00", "market": "HAD", "selection": "H", "odds": 1.80},
            {"captured_at": "2026-07-05 11:00", "market": "HAD", "selection": "D", "odds": 3.40},
            {"captured_at": "2026-07-05 11:00", "market": "HAD", "selection": "A", "odds": 4.20},
        ]
    )
    movement = _market_movement(frame)
    home = movement.loc[movement["结果"] == "主胜"].iloc[0]
    assert home["市场概率变化"] > 0
    assert home["方向"] == "增强"
    assert home["SP波动率"] > 0
