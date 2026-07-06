from __future__ import annotations

TEAM_MAPS: dict[str, dict[str, str]] = {
    "瑞典超级联赛": {
        "卡尔马": "Kalmar",
        "厄尔格里特": "Orgryte",
        "IFK哥德堡": "Goteborg",
        "AIK索尔纳": "AIK",
        "索尔纳": "AIK",
        "埃尔夫斯堡": "Elfsborg",
        "哈马比": "Hammarby",
        "赫根": "Hacken",
        "佐加顿斯": "Djurgarden",
        "布鲁马波卡纳": "Brommapojkarna",
        "哥德堡盖斯": "GAIS",
        "北雪平": "Norrkoping",
        "米亚尔比": "Mjallby",
        "天狼星": "Sirius",
        "瓦纳默": "Varnamo",
        "韦斯特罗斯": "Vasteras SK",
        "代格福什": "Degerfors",
    },
    "世界杯国家队": {
        "巴西": "Brazil",
        "挪威": "Norway",
        "墨西哥": "Mexico",
        "英格兰": "England",
        "葡萄牙": "Portugal",
        "西班牙": "Spain",
        "美国": "USA",
        "比利时": "Belgium",
        "阿根廷": "Argentina",
        "埃及": "Egypt",
        "瑞士": "Switzerland",
        "哥伦比亚": "Colombia",
        "法国": "France",
        "摩洛哥": "Morocco",
        "德国": "Germany",
        "荷兰": "Netherlands",
        "克罗地亚": "Croatia",
        "日本": "Japan",
        "韩国": "South Korea",
        "澳大利亚": "Australia",
        "加拿大": "Canada",
        "塞内加尔": "Senegal",
        "加纳": "Ghana",
        "乌拉圭": "Uruguay",
        "厄瓜多尔": "Ecuador",
        "卡塔尔": "Qatar",
        "沙特阿拉伯": "Saudi Arabia",
        "沙特": "Saudi Arabia",
        "伊朗": "Iran",
        "突尼斯": "Tunisia",
    },
}


def competition_for_league(league_name: str) -> str | None:
    if "瑞典" in league_name:
        return "瑞典超级联赛"
    if league_name == "世界杯":
        return "世界杯国家队"
    return None


def map_team_name(competition: str, team_name: str) -> str:
    return TEAM_MAPS.get(competition, {}).get(team_name, team_name)
