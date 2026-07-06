from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_model.core import get_settings  # noqa: E402
from football_model.data import LocalDatabase, MatchRepository, OddsRepository  # noqa: E402
from football_model.data.adapters import FootballDataCsvAdapter  # noqa: E402
from football_model.services import ModelTrainingService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="导入公开英超历史数据并训练本地模型")
    parser.add_argument("--seasons", nargs="+", default=["2324", "2425", "2526"])
    parser.add_argument("--division", default="E0")
    args = parser.parse_args()

    settings = get_settings(ROOT)
    database = LocalDatabase(settings.database_path)
    database.initialize()
    matches = MatchRepository(database)
    odds = OddsRepository(database)
    adapter = FootballDataCsvAdapter(settings.raw_dir)

    total_matches = 0
    total_odds = 0
    competition = ""
    for season in args.seasons:
        csv_path = adapter.download(season, args.division)
        match_frame, odds_frame = adapter.normalize(csv_path, season=season, division=args.division)
        total_matches += matches.import_frame(match_frame, source="football-data.co.uk")
        total_odds += odds.import_frame(odds_frame)
        competition = str(match_frame.iloc[0]["competition"])
        print(f"IMPORTED season={season} matches={len(match_frame)} odds={len(odds_frame)}")

    model, model_id = ModelTrainingService(database, settings.artifacts_dir).train_dixon_coles(competition)
    print(f"TOTAL matches={total_matches} odds={total_odds}")
    print(
        f"TRAINED model_id={model_id} competition={competition} "
        f"teams={len(model.teams)} nll={model.metrics['weighted_nll_per_match']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
