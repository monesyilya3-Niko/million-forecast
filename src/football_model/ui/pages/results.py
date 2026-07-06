from __future__ import annotations

import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase
from football_model.ui.components import hero_pro, empty_state


def render_results(database: LocalDatabase) -> None:
    hero_pro("比赛结果", "完场比分与赛前模型概率对照。", "RESULTS & SETTLEMENT", ["赛果验证", "命中率"])
    with database.connection(read_only=True) as connection:
        frame = connection.execute(
            """
            SELECT s.business_date, s.league_name, s.home_team, s.away_team,
                   r.status, r.home_goals, r.away_goals, r.provider, r.updated_at,
                   p.home_probability, p.draw_probability, p.away_probability,
                   p.confidence, p.model_version
            FROM match_results r
            JOIN sporttery_matches s USING (match_id)
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY created_at DESC) AS rn
                FROM predictions
            ) p ON r.match_id = p.match_id AND p.rn = 1
            ORDER BY s.business_date DESC, s.kickoff DESC
            """
        ).df()
    if frame.empty:
        empty_state("暂未同步到完场结果", "配置API_FOOTBALL_KEY后，更新服务会自动写入结果。", "📊")
        return
    frame["比分"] = frame["home_goals"].astype(str) + "–" + frame["away_goals"].astype(str)
    frame["比赛"] = frame["home_team"] + " vs " + frame["away_team"]
    frame["实际赛果"] = frame.apply(
        lambda row: (
            "主胜"
            if row["home_goals"] > row["away_goals"]
            else "平局"
            if row["home_goals"] == row["away_goals"]
            else "客胜"
        ),
        axis=1,
    )
    probability_columns = ["home_probability", "draw_probability", "away_probability"]
    labels = ["主胜", "平局", "客胜"]
    frame["模型首选"] = frame.apply(
        lambda row: (
            labels[int(pd.Series([row[column] for column in probability_columns]).astype(float).idxmax())]
            if all(pd.notna(row[column]) for column in probability_columns)
            else "未预测"
        ),
        axis=1,
    )
    frame["命中"] = frame["模型首选"] == frame["实际赛果"]
    for column in ["home_probability", "draw_probability", "away_probability"]:
        frame[column] = frame[column].map(lambda value: f"{value:.1%}" if pd.notna(value) else "未预测")
    st.dataframe(
        frame[
            [
                "business_date",
                "league_name",
                "比赛",
                "比分",
                "实际赛果",
                "模型首选",
                "命中",
                "status",
                "home_probability",
                "draw_probability",
                "away_probability",
                "confidence",
                "provider",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
