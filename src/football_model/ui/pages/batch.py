from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from football_model.services import AnalysisService
from football_model.ui.components import format_percent_columns, hero_pro

logger = logging.getLogger(__name__)


TEMPLATE = pd.DataFrame(
    [
        {
            "home_team": "主队 A",
            "away_team": "客队 B",
            "home_xg": 1.65,
            "away_xg": 1.10,
            "odds_home": 1.95,
            "odds_draw": 3.35,
            "odds_away": 3.75,
        },
        {
            "home_team": "主队 C",
            "away_team": "客队 D",
            "home_xg": 1.20,
            "away_xg": 1.35,
            "odds_home": 2.55,
            "odds_draw": 3.10,
            "odds_away": 2.65,
        },
    ]
)


def render_batch(service: AnalysisService) -> None:
    hero_pro("批量比赛分析", "上传标准CSV，一次生成多场比赛的概率、比分和市场价值结果。", "BATCH ANALYSIS", ["CSV导入", "批量计算"])
    st.download_button(
        "下载CSV模板",
        TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
        "batch_template.csv",
        "text/csv",
    )
    upload = st.file_uploader("上传比赛CSV", type="csv")
    frame = pd.read_csv(upload) if upload else TEMPLATE
    missing = set(TEMPLATE.columns) - set(frame.columns)
    if missing:
        st.error(f"缺少字段：{', '.join(sorted(missing))}")
        return
    st.caption("未上传文件时显示示例数据。")
    st.dataframe(frame, hide_index=True, width="stretch")
    results = service.analyze_batch(frame)
    formatted = format_percent_columns(results, ["主胜概率", "平局概率", "客胜概率", "最高理论EV"])
    st.subheader("分析结果")
    st.dataframe(formatted, hide_index=True, width="stretch")
    st.download_button(
        "导出分析结果",
        results.to_csv(index=False).encode("utf-8-sig"),
        "analysis_results.csv",
        "text/csv",
    )
