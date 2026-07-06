from __future__ import annotations

import json
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_model.core import AppSettings
from football_model.data import LocalDatabase, MatchRepository, ModelRepository
from football_model.services import ModelTrainingService
from football_model.ui.components import hero_pro, section_header, safe_html, empty_state, render_risk_note

logger = logging.getLogger(__name__)


def render_model_center(database: LocalDatabase, settings: AppSettings) -> None:
    hero_pro("模型中心", "训练、注册、评估和比较联赛模型版本。", "MODEL REGISTRY", ["Dixon-Coles", "Poisson", "集成器"])

    repository = ModelRepository(database)
    models = repository.list_models()
    trained = repository.trained_models()
    active_count = int((models["status"] == "active").sum()) if not models.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已注册模型", len(models))
    c2.metric("活动模型", active_count)
    c3.metric("可加载训练模型", len(trained))
    c4.metric("训练数据", "9,377场")

    training_tab, registry_tab, evaluation_tab, architecture_tab = st.tabs([
        "训练模型", "模型注册表", "模型评估", "组合架构"
    ])

    with training_tab:
        _render_training_tab(database, settings, repository, trained)

    with registry_tab:
        _render_registry_tab(models)

    with evaluation_tab:
        _render_evaluation_tab(database)

    with architecture_tab:
        _render_architecture_tab(trained)


def _render_training_tab(database: LocalDatabase, settings: AppSettings, repository: ModelRepository, trained: pd.DataFrame) -> None:
    """Render training tab."""
    section_header("模型训练", "按联赛训练Dixon-Coles模型。")

    competitions = MatchRepository(database).competitions()
    if competitions.empty:
        empty_state("暂无训练数据", "请先在数据中心导入至少100场已完成比赛。", "📊")
        return

    st.dataframe(competitions, hide_index=True, use_container_width=True)

    competition = st.selectbox("选择训练赛事", competitions["competition"].tolist(), key="mc-competition")
    selected = competitions.loc[competitions["competition"] == competition].iloc[0]
    st.caption(f"训练样本：{selected['matches']}场 ｜ 球队：{selected['teams']}支")

    if st.button("训练新的 Dixon–Coles 模型", type="primary", key="mc-train-btn"):
        with st.spinner("正在进行时间衰减参数估计……"):
            try:
                model, model_id = ModelTrainingService(database, settings.artifacts_dir).train_dixon_coles(competition)
                st.success(f"训练完成：{safe_html(model_id)}，加权NLL/场={model.metrics['weighted_nll_per_match']:.4f}")
                st.rerun()
            except (ValueError, RuntimeError) as error:
                st.error(f"训练失败：{safe_html(str(error))}")


def _render_registry_tab(models: pd.DataFrame) -> None:
    """Render model registry tab."""
    section_header("模型注册表", "已注册模型版本和状态。")

    if models.empty:
        empty_state("暂无注册模型", "请先训练模型。", "📦")
        return

    display = models.copy()
    if "metrics_json" in display:
        display["metrics_json"] = display["metrics_json"].map(_short_metrics)
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_evaluation_tab(database: LocalDatabase) -> None:
    """Render model evaluation tab."""
    section_header("模型评估", "查看模型预测表现和校准情况。")

    # Get predictions with results
    with database.connection(read_only=True) as conn:
        data = conn.execute("""
            SELECT p.match_id, p.model_version, p.home_probability, p.draw_probability, p.away_probability,
                   p.confidence, p.created_at,
                   r.home_goals, r.away_goals
            FROM predictions p
            INNER JOIN match_results r ON p.match_id = r.match_id
            ORDER BY p.created_at DESC
        """).fetchall()

    if not data:
        empty_state("暂无已结算预测", "需要比赛结束后才能评估模型表现。", "📈")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "match_id", "model_version", "home_prob", "draw_prob", "away_prob",
        "confidence", "created_at", "home_goals", "away_goals"
    ])

    # Core metrics
    section_header("核心指标", "基于已结算预测的评估。")

    from sklearn.metrics import log_loss
    import numpy as np

    y_true = np.where(
        df["home_goals"] > df["away_goals"], 0,
        np.where(df["home_goals"] == df["away_goals"], 1, 2)
    )
    proba = df[["home_prob", "draw_prob", "away_prob"]].values
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = proba.argmax(axis=1)

    ll = float(log_loss(y_true, proba, labels=[0, 1, 2]))
    one_hot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    accuracy = float((y_pred == y_true).mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("样本数", len(df))
    m2.metric("Log Loss", f"{ll:.4f}")
    m3.metric("Brier Score", f"{brier:.4f}")
    m4.metric("准确率", f"{accuracy:.1%}")

    # Calibration curve
    section_header("校准曲线", "预测概率与实际命中率的对比。")

    confidence = proba.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = pd.cut(confidence, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = (
        pd.DataFrame({"置信度": confidence, "命中": correct, "区间": bins})
        .groupby("区间", observed=True)
        .agg(预测概率=("置信度", "mean"), 实际命中率=("命中", "mean"), 样本数=("命中", "size"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="理想校准", line=dict(color="#94a3b8", dash="dash")))
    fig.add_trace(go.Scatter(x=calibration["预测概率"], y=calibration["实际命中率"], mode="lines+markers", name="模型", line=dict(color="#3b82f6", width=3)))
    fig.update_layout(
        height=350,
        xaxis_title="平均预测概率",
        yaxis_title="实际命中率",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        font=dict(color="#475569"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confidence segments
    section_header("置信度分层", "不同置信度区间的预测表现。")

    segments = []
    for label, low, high in [("低(0-40%)", 0, 0.40), ("中(40-55%)", 0.40, 0.55), ("高(55%+)", 0.55, 1.0)]:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() == 0:
            continue
        seg_true = y_true[mask]
        seg_proba = proba[mask]
        seg_pred = y_pred[mask]
        seg_acc = float((seg_pred == seg_true).mean())
        seg_ll = float(log_loss(seg_true, seg_proba, labels=[0, 1, 2])) if len(np.unique(seg_true)) > 1 else 0
        segments.append({
            "置信度区间": label,
            "样本数": int(mask.sum()),
            "Log Loss": f"{seg_ll:.4f}",
            "准确率": f"{seg_acc:.1%}",
        })

    if segments:
        st.dataframe(pd.DataFrame(segments), hide_index=True, use_container_width=True)

    # Prediction vs Result table
    section_header("最近预测记录", "最近20条已结算预测。")

    display_df = df.head(20).copy()
    display_df["预测"] = pd.Series(y_pred[:20]).map({0: "主胜", 1: "平局", 2: "客胜"})
    display_df["实际"] = pd.Series(y_true[:20]).map({0: "主胜", 1: "平局", 2: "客胜"})
    display_df["结果"] = (display_df["预测"] == display_df["实际"]).map({True: "✅", False: "❌"})
    display_df["概率"] = display_df.apply(lambda r: f"H{r['home_prob']:.0%} D{r['draw_prob']:.0%} A{r['away_prob']:.0%}", axis=1)

    st.dataframe(
        display_df[["match_id", "预测", "概率", "confidence", "实际", "结果"]].rename(columns={"match_id": "比赛", "confidence": "置信度"}),
        hide_index=True,
        use_container_width=True,
    )

    render_risk_note("模型评估基于历史预测，不代表未来表现。Log Loss和Brier Score越低越好，准确率需结合赔率水平判断。")


def _render_architecture_tab(trained: pd.DataFrame) -> None:
    """Render architecture tab."""
    section_header("模型架构", "当前系统模型组件和状态。")

    architecture = pd.DataFrame([
        ["市场基线", "SP去水概率", "对照基准", "已具备"],
        ["Elo", "球队动态实力", "结构化先验", "已接入特征"],
        ["Dixon–Coles", "完整比分分布", "核心统计模型", "已训练" if len(trained) else "待训练"],
        ["Poisson特征模型", "近期状态与Elo特征", "生产集成", "已接入"],
        ["XGBoost/神经网络", "非线性实验", "需独立校准与步进回测", "实验性·未进入生产"],
        ["概率校准", "Isotonic/Temperature", "修正置信度", "待回测样本"],
        ["集成器", "按联赛与时点动态加权", "最终输出", "已接入"],
        ["数据质量评分", "赔率/阵容/伤停/历史", "置信度输入", "已接入"],
        ["价值分析", "EV/Kelly/风险控制", "竞彩判断", "已接入"],
    ], columns=["组件", "输入/方法", "职责", "状态"])
    st.dataframe(architecture, hide_index=True, use_container_width=True)


def _short_metrics(value: str) -> str:
    """Format metrics JSON for display."""
    try:
        metrics = json.loads(value)
        if "matches" in metrics:
            return f"{metrics['matches']}场 · {metrics.get('teams', '-')}队 · NLL {metrics.get('weighted_nll_per_match', 0):.3f}"
        return value
    except (json.JSONDecodeError, TypeError):
        return str(value)
