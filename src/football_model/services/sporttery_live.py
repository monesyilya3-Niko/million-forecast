from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from football_model.data import LocalDatabase, SportteryRepository
from football_model.data.adapters import SportteryAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveRefreshResult:
    matches: int
    odds: int
    total_count: int
    last_update: str


class SportteryLiveService:
    def __init__(self, database: LocalDatabase) -> None:
        self.repository = SportteryRepository(database)
        self.adapter = SportteryAdapter()
        logger.info("SportteryLiveService initialized")

    def refresh(self) -> LiveRefreshResult:
        """Refresh live match data from sporttery.cn."""
        snapshot = self.adapter.fetch()
        matches, odds = self.repository.upsert(snapshot.matches, snapshot.odds)
        return LiveRefreshResult(
            matches=matches,
            odds=odds,
            total_count=snapshot.total_count,
            last_update=snapshot.last_update,
        )

    def today_string(self) -> str:
        return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    def dates(self) -> list[str]:
        return self.repository.dates()

    def matches_for_date(self, business_date: str) -> pd.DataFrame:
        return self.repository.matches_for_date(business_date)
