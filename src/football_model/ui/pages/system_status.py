"""系统状态页面 — 全面监控."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from football_model.core import AppSettings
from football_model.data import LocalDatabase
from football_model.ui.components import hero_pro, safe_html, section_header

logger = logging.getLogger(__name__)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def render_system_status(settings: AppSettings, database: LocalDatabase) -> None:
    hero_pro("系统状态", "数据库健康、数据源状态、模型状态和系统信息。", "SYSTEM MONITOR", ["本地部署", "DuckDB", "Python " + sys.version.split()[0]])

    disk = shutil.disk_usage(settings.project_root.drive + "\\")
    healthy = database.health_check()

    # 数据库统计
    try:
        with database.connection(read_only=True) as conn:
            sporttery_count = conn.execute("SELECT COUNT(*) FROM sporttery_matches").fetchone()[0]
            results_count = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
            predictions_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            models_count = conn.execute("SELECT COUNT(*) FROM model_registry WHERE artifact_path IS NOT NULL").fetchone()[0]
            training_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            p3_count = conn.execute("SELECT COUNT(*) FROM p3_draws").fetchone()[0]
            dlt_count = conn.execute("SELECT COUNT(*) FROM dlt_draws").fetchone()[0]
            odds_count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    except Exception as e:
        st.error(f"数据库查询异常: {e}")
        sporttery_count = results_count = predictions_count = models_count = 0
        training_count = p3_count = dlt_count = odds_count = 0

    section_header("数据总览", "各类数据统计。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据库状态", "正常" if healthy else "异常")
    c2.metric("足球比赛", f"{training_count:,}")
    c3.metric("竞彩比赛", str(sporttery_count))
    c4.metric("赔率快照", f"{odds_count:,}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预测记录", str(predictions_count))
    c2.metric("比赛结果", str(results_count))
    c3.metric("排列三", str(p3_count))
    c4.metric("大乐透", str(dlt_count))

    # 训练数据详情
    section_header("训练数据", "各联赛训练样本。")
    try:
        with database.connection(read_only=True) as conn:
            comps = conn.execute('SELECT competition, COUNT(*) FROM matches GROUP BY competition ORDER BY COUNT(*) DESC').fetchall()
    except Exception:
        comps = []
    if comps:
        comp_df = pd.DataFrame(comps, columns=["联赛", "场次"])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # 模型状态
    section_header("已训练模型", "模型版本和状态。")
    try:
        with database.connection(read_only=True) as conn:
            models = conn.execute("""
                SELECT model_id, model_type, version, status, metrics_json, created_at
                FROM model_registry
                WHERE artifact_path IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 10
            """).fetchall()
    except Exception:
        models = []

    if models:
        for m in models:
            model_id, model_type, version, status, metrics_json, created_at = m
            with st.expander(f"📦 {safe_html(model_id[:40])} | {safe_html(model_type)}"):
                st.caption(f"版本: {safe_html(version)} | 状态: {safe_html(status)} | 创建: {safe_html(str(created_at))}")
                if metrics_json:
                    import json
                    try:
                        metrics = json.loads(metrics_json)
                        mc1, mc2, mc3 = st.columns(3)
                        if "matches" in metrics:
                            mc1.metric("训练样本", f"{metrics['matches']:,}场")
                        if "teams" in metrics:
                            mc2.metric("球队数", f"{metrics['teams']}支")
                        if "holdout_log_loss" in metrics:
                            mc3.metric("Log Loss", f"{metrics['holdout_log_loss']:.4f}")
                    except json.JSONDecodeError:
                        pass

    # 数据同步
    section_header("数据同步", "手动同步数据。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 同步竞彩数据", key="sync-sporttery", use_container_width=True):
            with st.spinner("正在同步..."):
                try:
                    from football_model.services import SportteryLiveService
                    service = SportteryLiveService(database)
                    result = service.refresh()
                    st.success(f"已同步 {result.total_count} 场比赛，{result.odds} 条赔率")
                except Exception as e:
                    st.error(f"同步失败: {e}")

    with col2:
        if st.button("📊 更新比赛结果", key="sync-results", use_container_width=True):
            with st.spinner("正在检查比赛结果..."):
                try:
                    from football_model.services.result_updater import auto_update_results
                    message = auto_update_results(database)
                    st.success(message)
                except Exception as e:
                    st.error(f"更新失败: {e}")

    # 系统信息
    section_header("系统信息", "运行路径和存储。")
    paths = pd.DataFrame([
        ["项目根目录", str(settings.project_root)],
        ["本地数据库", str(settings.database_path)],
        ["模型文件", str(settings.artifacts_dir)],
    ], columns=["用途", "路径"])
    st.dataframe(paths, hide_index=True, use_container_width=True)

    sys_c1, sys_c2, sys_c3, sys_c4 = st.columns(4)
    sys_c1.metric("D盘可用", f"{disk.free / 1024**3:.1f} GB")
    sys_c2.metric("数据目录", f"{_directory_size(settings.data_dir) / 1024**2:.1f} MB")
    sys_c3.metric("Python", sys.version.split()[0])
    sys_c4.metric("模型数量", str(models_count))
