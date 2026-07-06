from __future__ import annotations

import logging
import hashlib
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_model.core import get_settings  # noqa: E402
from football_model.data import LocalDatabase, MatchRepository, OddsRepository  # noqa: E402
from football_model.services import ModelTrainingService  # noqa: E402


def stable_id(competition: str, kickoff: object, home: str, away: str) -> str:
    """Generate a stable match ID from match details."""
    raw = f"{competition}|{kickoff}|{home}|{away}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def normalize_sweden(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize Swedish league data from CSV."""
    logger.info(f"Normalizing Swedish league data from {path}")
    raw = pd.read_csv(path)
    kickoff = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), dayfirst=True)
    competition = "瑞典超级联赛"
    matches = pd.DataFrame(
        {
            "kickoff": kickoff,
            "competition": competition,
            "season": raw["Season"].astype(str),
            "home_team": raw["Home"],
            "away_team": raw["Away"],
            "home_goals": raw["HG"],
            "away_goals": raw["AG"],
            "status": "completed",
            "source": "football-data.co.uk",
        }
    )
    matches["match_id"] = matches.apply(
        lambda row: stable_id(competition, row["kickoff"], row["home_team"], row["away_team"]), axis=1
    )
    odds_wide = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "captured_at": matches["kickoff"],
            "H": pd.to_numeric(raw["AvgCH"], errors="coerce"),
            "D": pd.to_numeric(raw["AvgCD"], errors="coerce"),
            "A": pd.to_numeric(raw["AvgCA"], errors="coerce"),
        }
    )
    odds = odds_wide.melt(
        id_vars=["match_id", "captured_at"],
        value_vars=["H", "D", "A"],
        var_name="selection",
        value_name="odds",
    ).dropna(subset=["odds"])
    odds["market"] = "1x2_closing_average"
    odds["goal_line"] = None
    odds["source"] = "football-data.co.uk"
    return matches, odds


def normalize_world_cup(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for sheet in ["WorldCup2014", "WorldCup2018", "WorldCup2022", "WorldCup2026"]:
        raw = pd.read_excel(path, sheet_name=sheet)
        raw = raw.loc[raw["Finished"] == "90 minutes"].copy()
        frames.append(
            pd.DataFrame(
                {
                    "kickoff": pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str)),
                    "season": sheet.str.replace("WorldCup", "") if hasattr(sheet, "str") else sheet[-4:],
                    "home_team": raw["Home"],
                    "away_team": raw["Away"],
                    "home_goals": raw["HGFT"],
                    "away_goals": raw["AGFT"],
                    "odds_h": pd.to_numeric(raw["H-Avg"], errors="coerce"),
                    "odds_d": pd.to_numeric(raw["D-Avg"], errors="coerce"),
                    "odds_a": pd.to_numeric(raw["A-Avg"], errors="coerce"),
                }
            )
        )
    qualifiers = pd.read_excel(path, sheet_name="WorldCup2026Qualifiers")
    frames.append(
        pd.DataFrame(
            {
                "kickoff": pd.to_datetime(qualifiers["Date"]),
                "season": "2026Q",
                "home_team": qualifiers["Home"],
                "away_team": qualifiers["Away"],
                "home_goals": qualifiers["HG"],
                "away_goals": qualifiers["AG"],
                "odds_h": pd.to_numeric(qualifiers["H_Avg"], errors="coerce"),
                "odds_d": pd.to_numeric(qualifiers["D_Avg"], errors="coerce"),
                "odds_a": pd.to_numeric(qualifiers["A_Avg"], errors="coerce"),
            }
        )
    )
    data = pd.concat(frames, ignore_index=True).dropna(subset=["kickoff", "home_goals", "away_goals"])
    competition = "世界杯国家队"
    matches = data[["kickoff", "season", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    matches["competition"] = competition
    matches["status"] = "completed"
    matches["source"] = "football-data.co.uk"
    matches["match_id"] = matches.apply(
        lambda row: stable_id(competition, row["kickoff"], row["home_team"], row["away_team"]), axis=1
    )
    odds_wide = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "captured_at": matches["kickoff"],
            "H": data["odds_h"],
            "D": data["odds_d"],
            "A": data["odds_a"],
        }
    )
    odds = odds_wide.melt(
        id_vars=["match_id", "captured_at"],
        value_vars=["H", "D", "A"],
        var_name="selection",
        value_name="odds",
    ).dropna(subset=["odds"])
    odds["market"] = "1x2_closing_average"
    odds["goal_line"] = None
    odds["source"] = "football-data.co.uk"
    return matches, odds


def main() -> int:
    settings = get_settings(ROOT)
    database = LocalDatabase(settings.database_path)
    database.initialize()
    match_repository = MatchRepository(database)
    odds_repository = OddsRepository(database)
    raw = settings.raw_dir / "today-leagues"
    sources = [
        (normalize_sweden(raw / "SWE.csv"), None),
        (
            normalize_world_cup(raw / "WorldCup2026.xlsx"),
            set(pd.read_excel(raw / "WorldCup2026.xlsx", sheet_name="WorldCup2026")["Home"])
            | set(pd.read_excel(raw / "WorldCup2026.xlsx", sheet_name="WorldCup2026")["Away"]),
        ),
    ]
    for (matches, odds), team_scope in sources:
        competition = str(matches.iloc[0]["competition"])
        match_repository.import_frame(matches, source="football-data.co.uk")
        odds_repository.import_frame(odds)
        model, model_id = ModelTrainingService(database, settings.artifacts_dir).train_dixon_coles(
            competition, team_scope=team_scope
        )
        print(f"TRAINED competition={competition} matches={len(matches)} teams={len(model.teams)} model_id={model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
