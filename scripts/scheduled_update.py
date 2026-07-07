"""Scheduled data update script.

Run this script periodically (e.g., every 6 hours) to keep data fresh.
Can be used with Windows Task Scheduler or cron.

Usage:
    python scripts/scheduled_update.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

from football_model.data import LocalDatabase
from football_model.core import get_settings
from football_model.services.data_update import DataUpdateService
from football_model.services.sporttery_live import SportteryLiveService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings(Path(__file__).resolve().parent.parent)
    database = LocalDatabase(settings.database_path)

    logger.info("=== 百万竞猜 · 定时数据更新 ===")
    logger.info("数据库: %s", settings.database_path)

    # 1. 同步竞彩数据
    logger.info("[1/3] 同步竞彩赛程...")
    try:
        live_service = SportteryLiveService(database)
        result = live_service.refresh()
        logger.info("  竞彩同步完成: %d 场比赛", result.total_count)
    except Exception as e:
        logger.warning("  竞彩同步失败: %s", e)

    # 2. 更新阵容/赛果/球队资料
    logger.info("[2/3] 更新阵容和赛果...")
    try:
        update_service = DataUpdateService(database)
        results = update_service.update_all()
        logger.info("  阵容更新: %d 条", results["lineups_updated"])
        logger.info("  赛果更新: %d 条", results["results_updated"])
        logger.info("  球队资料: %d 条", results["profiles_updated"])
    except Exception as e:
        logger.warning("  数据更新失败: %s", e)

    # 3. 数据质量检查
    logger.info("[3/3] 数据质量检查...")
    try:
        from football_model.lottery.validators import calculate_lottery_data_quality
        p3q = calculate_lottery_data_quality(database, "p3")
        dltq = calculate_lottery_data_quality(database, "dlt")
        logger.info("  排列三: %d 期, 质量 %.0f (%s)", p3q.total_issues, p3q.quality_score, p3q.quality_level)
        logger.info("  大乐透: %d 期, 质量 %.0f (%s)", dltq.total_issues, dltq.quality_score, dltq.quality_level)
    except Exception as e:
        logger.warning("  质量检查失败: %s", e)

    logger.info("=== 更新完成 ===")


if __name__ == "__main__":
    main()
