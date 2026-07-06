from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_model.core import competition_for_league, get_settings  # noqa: E402
from football_model.data import LocalDatabase, SportteryRepository  # noqa: E402
from football_model.services import LiveContextService  # noqa: E402


def sync_once(*, horizon_hours: float = 120, lookback_hours: float = 12) -> None:
    settings = get_settings(ROOT)
    database = LocalDatabase(settings.database_path)
    database.initialize()
    repository = SportteryRepository(database)
    service = LiveContextService(database)
    now = pd.Timestamp.now()
    for business_date in repository.dates():
        for _, match in repository.matches_for_date(business_date).iterrows():
            kickoff = pd.Timestamp(match["kickoff"])
            if now - timedelta(hours=lookback_hours) <= kickoff <= now + timedelta(hours=horizon_hours):
                competition = competition_for_league(str(match["league_name"]))
                if competition:
                    result = service.sync_match(match, competition)
                    print(match["match_id"], result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    while True:
        sync_once(horizon_hours=1.5, lookback_hours=1.0) if args.loop else sync_once()
        if not args.loop:
            return 0
        time.sleep(900)


if __name__ == "__main__":
    raise SystemExit(main())
