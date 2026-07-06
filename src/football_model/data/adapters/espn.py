"""ESPN API adapter for fetching match lineups and team rosters.

Uses ESPN's public API (no API key required) to get lineup data
for international and club football matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# ESPN league IDs for supported competitions
ESPN_LEAGUES = {
    "世界杯": "fifa.world",
    "世界杯国家队": "fifa.world",
    "英格兰超级联赛": "eng.1",
    "瑞典超级联赛": "swe.1",
    "西甲": "esp.1",
    "意甲": "ita.1",
    "德甲": "ger.1",
    "法甲": "fra.1",
    "欧冠": "uefa.champions",
}

# Team name mapping (Chinese to ESPN search terms)
TEAM_SEARCH_MAP = {
    "葡萄牙": "Portugal", "西班牙": "Spain", "巴西": "Brazil",
    "阿根廷": "Argentina", "法国": "France", "德国": "Germany",
    "英格兰": "England", "荷兰": "Netherlands", "比利时": "Belgium",
    "日本": "Japan", "韩国": "South Korea", "摩洛哥": "Morocco",
    "瑞士": "Switzerland", "哥伦比亚": "Colombia", "克罗地亚": "Croatia",
    "挪威": "Norway", "墨西哥": "Mexico", "美国": "USA",
    "沙特": "Saudi Arabia", "伊朗": "Iran", "澳大利亚": "Australia",
    "加拿大": "Canada", "塞内加尔": "Senegal", "乌拉圭": "Uruguay",
    "埃及": "Egypt", "加纳": "Ghana", "突尼斯": "Tunisia",
    "厄瓜多尔": "Ecuador", "卡塔尔": "Qatar", "沙特阿拉伯": "Saudi Arabia",
    "喀麦隆": "Cameroon", "科特迪瓦": "Côte d'Ivoire", "巴拉圭": "Paraguay",
    # Swedish teams
    "埃尔夫斯堡": "Elfsborg", "哈马比": "Hammarby", "IFK哥德堡": "Goteborg",
    "AIK索尔纳": "AIK", "哥德堡盖斯": "GAIS", "北雪平": "Norrkoping",
    "赫根": "Hacken", "佐加顿斯": "Djurgarden", "天狼星": "Sirius",
    "米亚尔比": "Mjallby", "卡尔马": "Kalmar", "布鲁马波卡纳": "Brommapojkarna",
    "代格福什": "Degerfors", "瓦纳默": "Varnamo", "韦斯特罗斯": "Vasteras SK",
    "厄尔格里特": "Orgryte",
}


@dataclass
class ESPNPlayer:
    name: str
    position: str
    jersey: str = ""
    starter: bool = False


@dataclass
class ESPNLineup:
    team_name: str
    formation: str
    players: list[ESPNPlayer]
    is_confirmed: bool = True


class ESPNAdapter:
    """Fetch match lineups from ESPN public API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

    def __init__(self, timeout: float = 20) -> None:
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def get_team_roster(self, league: str, team_name: str) -> list[ESPNPlayer] | None:
        """Get team roster from most recent completed match.

        Tries multiple approaches:
        1. Search recent events for the team
        2. Use the team's roster endpoint directly
        """
        espn_league = ESPN_LEAGUES.get(league)
        if not espn_league:
            logger.warning("ESPN: unsupported league '%s'", league)
            return None

        en_team = TEAM_SEARCH_MAP.get(team_name, team_name)

        # Approach 1: Search recent events
        players = self._search_recent_events(espn_league, en_team)
        if players and len(players) >= 11:
            return players

        # Approach 2: Try direct team roster
        players = self._get_direct_roster(espn_league, en_team)
        if players and len(players) >= 11:
            return players

        # Approach 3: Return whatever we have if >= 5 players
        if players and len(players) >= 5:
            return players

        logger.warning("ESPN: could not get full roster for %s (got %d players)", team_name, len(players) if players else 0)
        return players if players else None

    def _search_recent_events(self, league: str, team_name: str) -> list[ESPNPlayer] | None:
        """Search recent events for team lineup."""
        today = datetime.now()
        # Try multiple date ranges
        for days_back in [7, 14, 30]:
            start = (today - timedelta(days=days_back)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            events = self._get_events(league, start, end)
            if not events:
                continue

            for event in events:
                event_name = event.get("name", "")
                if team_name.lower() in event_name.lower():
                    comp = event.get("competitions", [{}])[0]
                    status = comp.get("status", {}).get("type", {}).get("name", "")
                    if "FULL_TIME" in status or "FINAL" in status or "POST" in status:
                        players = self._get_event_roster(event["id"], league, team_name)
                        if players and len(players) >= 11:
                            return players

        return None

    def _get_direct_roster(self, league: str, team_name: str) -> list[ESPNPlayer] | None:
        """Try to get roster directly from team endpoint."""
        try:
            # First find team ID
            url = f"{self.BASE_URL}/{league}/teams"
            resp = self._client.get(url)
            if resp.status_code != 200:
                return None

            teams = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            team_id = None
            for t in teams:
                team_info = t.get("team", {})
                display_name = team_info.get("displayName", "")
                short_name = team_info.get("shortDisplayName", "")
                name = team_info.get("name", "")
                if (team_name.lower() in display_name.lower() or
                    team_name.lower() in short_name.lower() or
                    team_name.lower() in name.lower()):
                    team_id = team_info.get("id")
                    break

            if not team_id:
                return None

            # Get roster
            url = f"{self.BASE_URL}/{league}/teams/{team_id}/roster"
            resp = self._client.get(url)
            if resp.status_code != 200:
                return None

            data = resp.json()
            athletes = data.get("athletes", [])
            players = []

            for group in athletes:
                for item in group.get("items", []):
                    athlete = item.get("athlete", {})
                    pos_info = item.get("position", {})
                    players.append(ESPNPlayer(
                        name=athlete.get("displayName", "Unknown"),
                        position=pos_info.get("abbreviation", "??"),
                        jersey=item.get("jersey", ""),
                        starter=len(players) < 11,  # First 11 are starters
                    ))

            return players if players else None

        except Exception as e:
            logger.warning("ESPN direct roster failed: %s", e)
            return None

    def _get_events(self, league: str, start: str, end: str) -> list[dict]:
        """Get events for a league within date range."""
        try:
            url = f"{self.BASE_URL}/{league}/scoreboard?dates={start}-{end}"
            resp = self._client.get(url)
            if resp.status_code == 200:
                return resp.json().get("events", [])
        except Exception as e:
            logger.warning("ESPN scoreboard failed for %s: %s", league, e)
        return []

    def _get_event_roster(self, event_id: str, league: str, team_name: str) -> list[ESPNPlayer] | None:
        """Get roster for a specific team from event."""
        try:
            url = f"{self.BASE_URL}/{league}/summary?event={event_id}"
            resp = self._client.get(url)
            if resp.status_code != 200:
                return None

            data = resp.json()

            # Try rosters first
            rosters = data.get("rosters", [])
            for roster in rosters:
                roster_team = roster.get("team", {}).get("displayName", "")
                if team_name.lower() in roster_team.lower():
                    players_raw = roster.get("roster", [])
                    if len(players_raw) >= 11:
                        players = []
                        for p in players_raw:
                            athlete = p.get("athlete", {})
                            pos_info = p.get("position", {})
                            players.append(ESPNPlayer(
                                name=athlete.get("displayName", "Unknown"),
                                position=pos_info.get("abbreviation", "??"),
                                jersey=p.get("jersey", ""),
                                starter=p.get("starter", len(players) < 11),
                            ))
                        return players

            # Try boxscore
            boxscore = data.get("boxscore", {})
            teams = boxscore.get("teams", [])
            for team_data in teams:
                team_info = team_data.get("team", {})
                if team_name.lower() in team_info.get("displayName", "").lower():
                    # Boxscore statistics available but less reliable
                    pass

            # Try gameInfo lineup
            game_info = data.get("gameInfo", {})
            if game_info:
                status = game_info.get("status", {})
                if status.get("type", {}).get("name", "") in ("FULL_TIME", "FINAL"):
                    # Try to get lineup from competitors
                    competitors = data.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
                    for comp in competitors:
                        comp_team = comp.get("team", {})
                        if team_name.lower() in comp_team.get("displayName", "").lower():
                            # Found team but roster might be in different structure
                            pass

            return None

        except Exception as e:
            logger.warning("ESPN roster fetch failed: %s", e)
            return None
