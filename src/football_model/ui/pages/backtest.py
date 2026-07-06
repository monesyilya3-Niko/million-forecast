from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import log_loss

from football_model.ui.components import hero_pro

logger = logging.getLogger(__name__)


def render_backtest() -> None:
    hero_pro("历史回测", "检查概率质量、校准表现和基于历史SP的模拟结果。", "BACKTESTING", ["Log Loss", "Brier", "校准曲线"])
    st.info("CSV字段：prob_home、prob_draw、prob_away、result；result使用H/D/A。可选加入odds_home、odds_draw、odds_away。")
    upload = st.file_uploader("上传历史预测CSV", type="csv", key="backtest")
    if upload is None:
        st.markdown("回测必须使用预测发生时已经可见的数据，禁止使用未来阵容、最终SP或赛后统计。")
        return
    frame = pd.read_csv(upload)
    required = {"prob_home", "prob_draw", "prob_away", "result"}
    if not required.issubset(frame.columns):
        st.error(f"缺少字段：{', '.join(sorted(required - set(frame.columns)))}")
        return
    mapping = {"H": 0, "D": 1, "A": 2}
    valid = frame[frame["result"].isin(mapping)].copy()
    y_true = valid["result"].map(mapping).to_numpy()
    probabilities = valid[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    loss = log_loss(y_true, probabilities, labels=[0, 1, 2])
    one_hot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    accuracy = float((probabilities.argmax(axis=1) == y_true).mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", f"{len(valid):,}")
    c2.metric("Log Loss", f"{loss:.4f}")
    c3.metric("Brier Score", f"{brier:.4f}", f"命中率 {accuracy:.1%}")

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(int)
    bins = pd.cut(confidence, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = (
        pd.DataFrame({"置信度": confidence, "命中": correct, "区间": bins})
        .groupby("区间", observed=True)
        .agg(预测概率=("置信度", "mean"), 实际命中率=("命中", "mean"), 样本数=("命中", "size"))
        .reset_index()
    )
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="理想校准",
        line=dict(color="#94a3b8", dash="dash"),
    ))
    figure.add_trace(go.Scatter(
        x=calibration["预测概率"], y=calibration["实际命中率"],
        mode="lines+markers", name="模型",
        line=dict(color="#2563eb", width=3),
    ))
    figure.update_layout(
        height=400,
        xaxis_title="平均预测概率", yaxis_title="实际命中率",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        font=dict(color="#475569"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(figure, width="stretch")
