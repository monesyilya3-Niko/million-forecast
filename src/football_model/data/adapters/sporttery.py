from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SportterySnapshot:
    matches: pd.DataFrame
    odds: pd.DataFrame
    last_update: str
    total_count: int


class SportteryAdapter:
    endpoint = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchListV1.qry?clientCode=3001"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch(self) -> SportterySnapshot:
        """Fetch live match data with automatic retry on failure."""
        logger.info("Fetching live match data from sporttery.cn")
        payload = self._request_json()
        if str(payload.get("errorCode")) != "0":
            raise RuntimeError(f"竞彩网接口返回错误：{payload.get('errorMessage', '未知错误')}")
        value = payload.get("value") or {}
        last_update = str(value.get("lastUpdateTime") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        captured_at = pd.to_datetime(last_update)
        match_rows: list[dict[str, object]] = []
        odds_rows: list[dict[str, object]] = []

        for date_group in value.get("matchInfoList") or []:
            for match in date_group.get("subMatchList") or []:
                match_id = f"sporttery:{match['matchId']}"
                pools = {pool.get("poolCode"): pool for pool in match.get("poolList") or []}
                available = sorted(code for code in pools if code)
                match_rows.append(
                    {
                        "match_id": match_id,
                        "official_match_id": int(match["matchId"]),
                        "business_date": pd.to_datetime(match["businessDate"]).date(),
                        "match_number": match.get("matchNumStr", ""),
                        "kickoff": pd.to_datetime(f"{match['matchDate']} {match['matchTime']}"),
                        "weekday": match.get("weekday", ""),
                        "league_id": str(match.get("leagueId", "")),
                        "league_name": match.get("leagueAllName", ""),
                        "home_team_id": match.get("homeTeamId"),
                        "home_team": match.get("homeTeamAllName", ""),
                        "away_team_id": match.get("awayTeamId"),
                        "away_team": match.get("awayTeamAllName", ""),
                        "sell_status": str(match.get("sellStatus", "")),
                        "match_status": match.get("matchStatus", ""),
                        "remark": match.get("remark", ""),
                        "had_single": bool((pools.get("HAD") or {}).get("cbtSingle", 0)),
                        "hhad_single": bool((pools.get("HHAD") or {}).get("cbtSingle", 0)),
                        "available_pools": ",".join(available),
                        "last_update": captured_at,
                        "source": "sporttery.cn",
                    }
                )
                for odds_group in match.get("oddsList") or []:
                    market = odds_group.get("poolCode")
                    # Support HAD, HHAD, and TTG (total goals)
                    if market not in {"HAD", "HHAD", "TTG", "CRS", "HAFU"}:
                        continue
                    for selection, key in (("H", "h"), ("D", "d"), ("A", "a")):
                        value_text = odds_group.get(key)
                        if value_text in (None, ""):
                            continue
                        odds_rows.append(
                            {
                                "match_id": match_id,
                                "captured_at": captured_at,
                                "market": market,
                                "selection": selection,
                                "odds": float(value_text),
                                "goal_line": odds_group.get("goalLine") or None,
                                "source": "sporttery.cn",
                            }
                        )

        return SportterySnapshot(
            matches=pd.DataFrame(match_rows),
            odds=pd.DataFrame(odds_rows),
            last_update=last_update,
            total_count=int(value.get("totalCount") or len(match_rows)),
        )

    def _request_json(self) -> dict[str, object]:
        headers = {"Referer": "https://www.sporttery.cn/", "User-Agent": "FootballModelResearch/0.3"}
        try:
            response = httpx.get(self.endpoint, headers=headers, timeout=20, follow_redirects=True)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            result = subprocess.run(
                [
                    "curl.exe",
                    "--silent",
                    "--show-error",
                    "--fail",
                    "--location",
                    "--http1.1",
                    "--header",
                    "Referer: https://www.sporttery.cn/",
                    self.endpoint,
                ],
                check=True,
                capture_output=True,
            )
            return json.loads(result.stdout.decode("utf-8"))
