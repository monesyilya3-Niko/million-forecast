"""双语支持模块 — 中英文对照."""

from __future__ import annotations

# ── 足球术语 ──
FOOTBALL_TERMS = {
    # 比赛状态
    "scheduled": "未开始",
    "live": "进行中",
    "halftime": "中场休息",
    "finished": "已结束",
    "postponed": "延期",
    "cancelled": "取消",
    "selling": "销售中",
    "paused": "已暂停",
    "waiting": "待开售",

    # 赛果
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
    "主胜": "Home Win",
    "平局": "Draw",
    "客胜": "Away Win",

    # 赔率市场
    "HAD": "胜平负",
    "HHAD": "让球胜平负",
    "TOTAL_GOALS": "总进球数",
    "CORRECT_SCORE": "比分",
    "HALF_FULL": "半全场",

    # 球队数据
    "elo_rating": "Elo评分",
    "goals_for": "场均进球",
    "goals_against": "场均失球",
    "points_per_game": "场均积分",
    "clean_sheets": "零封场次",
    "form_last_5": "近5场状态",
    "form_last_10": "近10场状态",
    "home_strength": "主场强度",
    "away_strength": "客场强度",

    # 模型指标
    "log_loss": "对数损失",
    "brier_score": "布里尔分数",
    "accuracy": "准确率",
    "calibration": "校准度",
    "confidence": "可信度",
    "ev": "期望价值",
    "kelly": "凯利指数",
    "roi": "投资回报率",

    # 比分矩阵
    "score_matrix": "比分概率矩阵",
    "top_scores": "最可能比分",
    "over_25": "大于2.5球",
    "btts": "双方进球",
    "clean_sheet": "零封",

    # 阵容
    "formation": "阵型",
    "starting_xi": "首发阵容",
    "substitutes": "替补席",
    "injuries": "伤停名单",
    "suspensions": "停赛名单",

    # 技战术
    "attack_style": "进攻风格",
    "defense_style": "防守风格",
    "pressing": "压迫强度",
    "counter_attack": "反击能力",
    "wing_strength": "边路强度",
    "central_control": "中场控制",
    "set_piece": "定位球威胁",
    "defensive_weakness": "防守弱点",

    # 风险
    "risk_level": "风险等级",
    "low_risk": "低风险",
    "medium_risk": "中风险",
    "high_risk": "高风险",
    "not_recommended": "不建议",
}

# ── 彩票术语 ──
LOTTERY_TERMS = {
    # 排列三
    "p3": "排列三",
    "digit_1": "百位",
    "digit_2": "十位",
    "digit_3": "个位",
    "sum_value": "和值",
    "span_value": "跨度",
    "odd_count": "奇数个数",
    "even_count": "偶数个数",
    "big_count": "大数个数",
    "small_count": "小数个数",
    "pattern_type": "形态类型",
    "baozi": "豹子",
    "zusan": "组三",
    "zuliu": "组六",
    "road_012": "012路",
    "consecutive": "连号",
    "repeat": "重号",

    # 大乐透
    "dlt": "超级大乐透",
    "front_zone": "前区",
    "back_zone": "后区",
    "front_sum": "前区和值",
    "back_sum": "后区和值",
    "front_span": "前区跨度",
    "back_span": "后区跨度",
    "zone_ratio": "分区比",
    "odd_even_ratio": "奇偶比",
    "big_small_ratio": "大小比",

    # 通用
    "hot_number": "热号",
    "cold_number": "冷号",
    "warm_number": "温号",
    "omission": "遗漏值",
    "frequency": "出现频率",
    "candidate": "候选组合",
    "reference": "参考组合",
}

# ── 位置映射（中英文） ──
POSITION_MAP = {
    "G": ("门将", "Goalkeeper"),
    "GK": ("门将", "Goalkeeper"),
    "D": ("后卫", "Defender"),
    "CB": ("中后卫", "Centre-Back"),
    "LB": ("左后卫", "Left-Back"),
    "RB": ("右后卫", "Right-Back"),
    "LCB": ("左中卫", "Left Centre-Back"),
    "RCB": ("右中卫", "Right Centre-Back"),
    "LWB": ("左翼卫", "Left Wing-Back"),
    "RWB": ("右翼卫", "Right Wing-Back"),
    "M": ("中场", "Midfielder"),
    "CM": ("中场", "Central Midfielder"),
    "CDM": ("后腰", "Defensive Midfielder"),
    "CAM": ("前腰", "Attacking Midfielder"),
    "LM": ("左中场", "Left Midfielder"),
    "RM": ("右中场", "Right Midfielder"),
    "F": ("前锋", "Forward"),
    "ST": ("中锋", "Striker"),
    "CF": ("前锋", "Centre-Forward"),
    "LW": ("左边锋", "Left Winger"),
    "RW": ("右边锋", "Right Winger"),
    "SS": ("影锋", "Second Striker"),
    "SUB": ("替补", "Substitute"),
}


def bilingual(cn: str, en: str) -> str:
    """Format bilingual text."""
    if cn == en:
        return cn
    return f"{cn} / {en}"


def get_position_cn(pos: str) -> str:
    """Get Chinese position name."""
    return POSITION_MAP.get(pos.upper(), (pos, pos))[0]


def get_position_bilingual(pos: str) -> str:
    """Get bilingual position name."""
    cn, en = POSITION_MAP.get(pos.upper(), (pos, pos))
    return bilingual(cn, en)
