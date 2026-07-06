from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from football_model.core import AppSettings
from football_model.data import LocalDatabase
from football_model.ui.components import hero_pro, render_status_dot, safe_html, section_header

logger = logging.getLogger(__name__)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def render_system_status(settings: AppSettings, database: LocalDatabase) -> None:
    hero_pro("系统状态", "检查本地服务、存储空间、数据源健康和运行环境。", "LOCAL RUNTIME", ["本地部署", "DuckDB", "Python " + sys.version.split()[0]])

    disk = shutil.disk_usage(settings.project_root.drive + "\\")
    healthy = database.health_check()

    # 数据库统计
    with database.connection(read_only=True) as conn:
        sporttery_count = conn.execute("SELECT COUNT(*) FROM sporttery_matches").fetchone()[0]
        results_count = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        predictions_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        models_count = conn.execute("SELECT COUNT(*) FROM model_registry WHERE artifact_path IS NOT NULL").fetchone()[0]
        training_count = conn.execute("SELECT COUNT(*) FROM matches WHERE status='completed'").fetchone()[0]
        live_states_count = conn.execute("SELECT COUNT(*) FROM match_live_states").fetchone()[0]

    section_header("系统总览", "数据库健康、存储和核心指标。")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("数据库", "正常" if healthy else "异常")
    c2.metric("竞彩比赛", sporttery_count)
    c3.metric("已记录结果", results_count)
    c4.metric("预测记录", predictions_count)
    c5.metric("训练数据", training_count)
    c6.metric("实时状态", live_states_count)

    # 数据同步
    section_header("数据同步", "更新比赛结果、实时状态和赔率快照。")

    col1, col2, col3 = st.columns(3)
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

    with col3:
        if st.button("⚡ 同步实时状态", key="sync-live", use_container_width=True):
            with st.spinner("正在同步实时状态..."):
                try:
                    from football_model.services.match_live_state import MatchLiveStateService
                    service = MatchLiveStateService(database)
                    results = service.sync_all_pending()
                    success = sum(1 for r in results if r.success)
                    st.success(f"已同步 {success}/{len(results)} 场比赛实时状态")
                except Exception as e:
                    st.error(f"同步失败: {e}")

    # 数据源健康
    section_header("数据源健康", "各数据提供商的连接状态和配额。")

    health_col1, health_col2 = st.columns(2)

    with health_col1:
        st.markdown("**竞彩网数据源**")
        st.markdown(f"{render_status_dot('live')} 竞彩网API · 赛程+赔率", unsafe_allow_html=True)
        st.caption("自动刷新，每60秒同步一次")

    with health_col2:
        st.markdown("**API-Football**")
        try:
            from football_model.providers.live_state_provider import LiveStateProvider
            provider = LiveStateProvider()
            health = provider.health_check()
            if health.is_available:
                st.markdown(f"{render_status_dot('live')} API-Football · 实时状态+结果", unsafe_allow_html=True)
                if health.api_calls_remaining is not None:
                    st.caption(f"剩余配额: {health.api_calls_remaining} 次/天")
            else:
                st.markdown(f"{render_status_dot('error')} API-Football · 未配置", unsafe_allow_html=True)
                st.caption("设置 API_FOOTBALL_KEY 环境变量启用")
        except Exception:
            st.markdown(f"{render_status_dot('error')} API-Football · 初始化失败", unsafe_allow_html=True)

    # 同步日志
    section_header("最近同步日志", "查看数据同步历史。")

    try:
        from football_model.services.match_live_state import MatchLiveStateService
        live_service = MatchLiveStateService(database)
        logs = live_service.get_sync_logs(limit=20)

        if not logs.empty:
            st.dataframe(logs, hide_index=True, use_container_width=True)
        else:
            st.info("暂无同步日志")
    except Exception:
        st.info("同步日志服务不可用")

    # 模型状态
    section_header("模型状态", "已训练模型和性能指标。")

    with database.connection(read_only=True) as conn:
        models = conn.execute("""
            SELECT model_id, model_type, version, status, metrics_json, created_at
            FROM model_registry
            WHERE artifact_path IS NOT NULL
            ORDER BY created_at DESC
        """).fetchall()

    if models:
        for m in models:
            model_id, model_type, version, status, metrics_json, created_at = m
            with st.expander(f"📦 {safe_html(model_id[:40])}... | {safe_html(model_type)} | {safe_html(status)}"):
                st.caption(f"版本: {safe_html(version)} | 创建: {safe_html(str(created_at))}")
                if metrics_json:
                    import json
                    try:
                        metrics = json.loads(metrics_json)
                        if "holdout_log_loss" in metrics:
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Log Loss", f"{metrics['holdout_log_loss']:.4f}")
                            mc2.metric("Brier Score", f"{metrics.get('holdout_brier', 0):.4f}")
                            mc3.metric("准确率", f"{metrics.get('holdout_accuracy', 0):.1%}")
                        if "matches" in metrics:
                            st.caption(f"训练样本: {metrics['matches']}场 | 球队: {metrics.get('teams', '-')}支")
                    except json.JSONDecodeError:
                        pass
    else:
        st.info("暂无已训练模型")

    # 预测 vs 实际
    section_header("预测表现", "最近预测与实际结果对比。")

    with database.connection(read_only=True) as conn:
        predictions = conn.execute("""
            SELECT p.match_id, p.model_version, p.home_probability, p.draw_probability, p.away_probability,
                   p.confidence, p.created_at,
                   r.home_goals, r.away_goals
            FROM predictions p
            LEFT JOIN match_results r ON p.match_id = r.match_id
            ORDER BY p.created_at DESC
            LIMIT 20
        """).fetchall()

    if predictions:
        rows = []
        correct = 0
        total_with_result = 0

        for p in predictions:
            match_id, model_ver, h_prob, d_prob, a_prob, confidence, created, hg, ag = p

            pred = "主胜" if h_prob > max(d_prob, a_prob) else "平局" if d_prob > max(h_prob, a_prob) else "客胜"

            if hg is not None and ag is not None:
                actual = "主胜" if hg > ag else "平局" if hg == ag else "客胜"
                is_correct = pred == actual
                total_with_result += 1
                if is_correct:
                    correct += 1
                result_str = f"{int(hg)}:{int(ag)} ({actual})"
                status = "✅" if is_correct else "❌"
            else:
                result_str = "未完场"
                status = "⏳"

            rows.append({
                "比赛": safe_html(str(match_id)[:25]),
                "预测": pred,
                "概率": f"H{h_prob:.0%} D{d_prob:.0%} A{a_prob:.0%}",
                "置信度": confidence,
                "实际": result_str,
                "状态": status,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        if total_with_result > 0:
            accuracy = correct / total_with_result
            st.metric("已结算预测准确率", f"{accuracy:.1%}", f"{correct}/{total_with_result}")
    else:
        st.info("暂无预测记录")

    # 系统信息
    section_header("系统信息", "运行路径和存储。")

    paths = pd.DataFrame([
        ["项目根目录", str(settings.project_root)],
        ["本地数据库", str(settings.database_path)],
        ["原始数据", str(settings.raw_dir)],
        ["特征数据", str(settings.features_dir)],
        ["模型文件", str(settings.artifacts_dir)],
        ["分析报告", str(settings.reports_dir)],
    ], columns=["用途", "路径"])
    st.dataframe(paths, hide_index=True, use_container_width=True)

    sys_c1, sys_c2, sys_c3, sys_c4 = st.columns(4)
    sys_c1.metric("D盘可用", f"{disk.free / 1024**3:.1f} GB")
    sys_c2.metric("数据目录", f"{_directory_size(settings.data_dir) / 1024**2:.1f} MB")
    sys_c3.metric("Python", sys.version.split()[0])
    sys_c4.metric("模型数量", models_count)
