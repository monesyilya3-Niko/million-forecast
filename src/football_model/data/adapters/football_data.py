from __future__ import annotations

import hashlib
import logging
import subprocess
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


LEAGUE_NAMES = {
    "E0": "英格兰超级联赛",
    "E1": "英格兰冠军联赛",
    "D1": "德国甲级联赛",
    "I1": "意大利甲级联赛",
    "SP1": "西班牙甲级联赛",
    "F1": "法国甲级联赛",
}


class FootballDataCsvAdapter:
    """Adapter for Football-Data's public historical CSV files."""

    base_url = "https://www.football-data.co.uk/mmz4281"

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir) / "football-data"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def source_url(self, season: str, division: str) -> str:
        return f"{self.base_url}/{season}/{division}.csv"

    def download(self, season: str, division: str = "E0") -> Path:
        destination = self.raw_dir / f"{division}_{season}.csv"
        if destination.exists() and destination.stat().st_size > 100:
            return destination
        url = self.source_url(season, division)
        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=30,
                headers={"User-Agent": "FootballModelResearch/0.2"},
            )
            response.raise_for_status()
            destination.write_bytes(response.content)
        except httpx.HTTPError:
            subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "--http1.1",
                    "--output",
                    str(destination),
                    url,
                ],
                check=True,
                capture_output=True,
            )
        if not destination.exists() or destination.stat().st_size < 100:
            raise ValueError(f"下载的数据为空：{season}/{division}")
        return destination

    def normalize(self, csv_path: str | Path, *, season: str, division: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        content = Path(csv_path).read_bytes()
        raw = pd.read_csv(BytesIO(content), encoding="cp1252", on_bad_lines="skip")
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Football-Data文件缺少字段：{', '.join(sorted(missing))}")

        raw = raw.dropna(subset=list(required)).copy()
        date_text = raw["Date"].astype(str)
        if "Time" in raw:
            date_text = date_text + " " + raw["Time"].fillna("00:00").astype(str)
        kickoff = pd.to_datetime(date_text, dayfirst=True, errors="coerce")
        competition = LEAGUE_NAMES.get(division, division)

        matches = pd.DataFrame(
            {
                "kickoff": kickoff,
                "competition": competition,
                "season": self._display_season(season),
                "home_team": raw["HomeTeam"].astype(str).str.strip(),
                "away_team": raw["AwayTeam"].astype(str).str.strip(),
                "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
                "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
                "status": "completed",
                "source": "football-data.co.uk",
            }
        ).dropna(subset=["kickoff", "home_goals", "away_goals"])
        matches["match_id"] = matches.apply(self._match_id, axis=1)

        # Import ALL available odds types
        all_odds = []
        for home_col, draw_col, away_col, market in self._all_odds_columns():
            if {home_col, draw_col, away_col}.issubset(raw.columns):
                odds_wide = pd.DataFrame(
                    {
                        "match_id": matches["match_id"].to_numpy(),
                        "captured_at": matches["kickoff"].to_numpy(),
                        "H": pd.to_numeric(raw.loc[matches.index, home_col], errors="coerce").to_numpy(),
                        "D": pd.to_numeric(raw.loc[matches.index, draw_col], errors="coerce").to_numpy(),
                        "A": pd.to_numeric(raw.loc[matches.index, away_col], errors="coerce").to_numpy(),
                    }
                )
                melted = odds_wide.melt(
                    id_vars=["match_id", "captured_at"],
                    value_vars=["H", "D", "A"],
                    var_name="selection",
                    value_name="odds",
                ).dropna(subset=["odds"])
                melted["market"] = market
                melted["source"] = "football-data.co.uk"
                all_odds.append(melted)

        odds = pd.concat(all_odds, ignore_index=True) if all_odds else pd.DataFrame()
        return matches.reset_index(drop=True), odds.reset_index(drop=True)

    @staticmethod
    def _display_season(season: str) -> str:
        start = 2000 + int(season[:2])
        end = 2000 + int(season[2:])
        return f"{start}/{end}"

    @staticmethod
    def _match_id(row: pd.Series) -> str:
        identity = f"{row['competition']}|{row['kickoff']}|{row['home_team']}|{row['away_team']}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _all_odds_columns() -> list[tuple[str, str, str, str]]:
        """All odds column sets to import."""
        return [
            ("AvgCH", "AvgCD", "AvgCA", "1x2_closing_average"),
            ("AvgH", "AvgD", "AvgA", "1x2_opening_average"),
            ("B365CH", "B365CD", "B365CA", "1x2_b365_closing"),
            ("B365H", "B365D", "B365A", "1x2_b365_opening"),
            ("BWCH", "BWCD", "BVCA", "1x2_bw_closing"),
            ("BWH", "BWD", "BWA", "1x2_bw_opening"),
            ("PSCH", "PSCD", "PSCA", "1x2_pinnacle_closing"),
            ("PSH", "PSD", "PSA", "1x2_pinnacle_opening"),
        ]

    @staticmethod
    def _select_odds_columns(raw: pd.DataFrame) -> tuple[str, str, str, str] | None:
        candidates = [
            ("AvgCH", "AvgCD", "AvgCA", "1x2_closing_average"),
            ("AvgH", "AvgD", "AvgA", "1x2_opening_average"),
            ("B365CH", "B365CD", "B365CA", "1x2_b365_closing"),
            ("B365H", "B365D", "B365A", "1x2_b365_opening"),
        ]
        for candidate in candidates:
            if set(candidate[:3]).issubset(raw.columns):
                return candidate
        return None
