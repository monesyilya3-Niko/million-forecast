"""数据中心 — 支持足球和彩票数据导入."""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from football_model.data import LocalDatabase, MatchRepository
from football_model.ui.components import hero_pro, section_header, empty_state, data_quality_panel

logger = logging.getLogger(__name__)

MATCH_TEMPLATE = pd.DataFrame([{
    "kickoff": "2026-08-01 19:30:00",
    "competition": "示例联赛",
    "season": "2026",
    "home_team": "主队 A",
    "away_team": "客队 B",
    "home_goals": "",
    "away_goals": "",
    "status": "scheduled",
}])

P3_TEMPLATE = pd.DataFrame([
    {"issue_no": "2024001", "draw_date": "2024-01-01", "digit_1": 3, "digit_2": 7, "digit_3": 2},
    {"issue_no": "2024002", "draw_date": "2024-01-02", "digit_1": 5, "digit_2": 1, "digit_3": 8},
])

DLT_TEMPLATE = pd.DataFrame([{
    "issue_no": "24001", "draw_date": "2024-01-01",
    "front_1": 5, "front_2": 12, "front_3": 18, "front_4": 25, "front_5": 33,
    "back_1": 3, "back_2": 9,
}])

P3_JSON_TEMPLATE = json.dumps([
    {"issue_no": "2024001", "draw_date": "2024-01-01", "numbers": [3, 7, 2], "source": "manual"},
    {"issue_no": "2024002", "draw_date": "2024-01-02", "numbers": [5, 1, 8], "source": "manual"},
], ensure_ascii=False, indent=2)

DLT_JSON_TEMPLATE = json.dumps([
    {"issue_no": "24001", "draw_date": "2024-01-01", "front": [5, 12, 18, 25, 33], "back": [3, 9], "source": "manual"},
], ensure_ascii=False, indent=2)


def render_data_center(database: LocalDatabase) -> None:
    hero_pro("数据中心", "管理足球赛程、彩票开奖和各类数据源。", "DATA CENTER", ["DuckDB", "CSV/JSON导入", "本地存储"])

    counts = database.table_counts()
    p3_count, dlt_count = _get_lottery_counts(database)

    section_header("数据总览", "各类数据统计。")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("足球比赛", f"{counts.get('matches', 0):,}")
    c2.metric("赔率快照", f"{counts.get('odds_snapshots', 0):,}")
    c3.metric("预测记录", f"{counts.get('predictions', 0):,}")
    c4.metric("实时竞彩", f"{counts.get('sporttery_matches', 0):,}")
    c5.metric("排列三", str(p3_count))
    c6.metric("大乐透", str(dlt_count))

    # 数据质量面板
    from football_model.lottery.validators import calculate_lottery_data_quality
    q1, q2 = st.columns(2)
    with q1:
        p3q = calculate_lottery_data_quality(database, "p3")
        p3_issues = [f"排列三: {p3q.quality_level}，共{p3q.total_issues}期"] + p3q.missing_issues + p3q.duplicate_issues
        data_quality_panel(p3q.quality_score, p3_issues or None)
    with q2:
        dltq = calculate_lottery_data_quality(database, "dlt")
        dlt_issues = [f"大乐透: {dltq.quality_level}，共{dltq.total_issues}期"] + dltq.missing_issues + dltq.duplicate_issues
        data_quality_panel(dltq.quality_score, dlt_issues or None)

    # Tabs
    football_tab, p3_tab, dlt_tab, records_tab = st.tabs([
        "足球数据", "排列三数据", "大乐透数据", "本地记录"
    ])

    with football_tab:
        _render_football_import(database)

    with p3_tab:
        _render_p3_import(database)

    with dlt_tab:
        _render_dlt_import(database)

    with records_tab:
        _render_records(database)


def _get_lottery_counts(database: LocalDatabase) -> tuple[int, int]:
    """Get lottery draw counts."""
    try:
        with database.connection(read_only=True) as conn:
            p3 = conn.execute("SELECT COUNT(*) FROM p3_draws").fetchone()[0]
            dlt = conn.execute("SELECT COUNT(*) FROM dlt_draws").fetchone()[0]
        return p3, dlt
    except Exception:
        return 0, 0


def _render_import_result(result) -> None:
    """Render detailed import result."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功", result.success)
    c2.metric("更新", result.updated)
    c3.metric("跳过", result.skipped)
    c4.metric("错误", result.error_count)
    if result.errors:
        with st.expander(f"错误详情 ({result.error_count} 条)", expanded=False):
            for err in result.errors[:20]:
                st.caption(f"⚠️ {err}")


def _render_football_import(database: LocalDatabase) -> None:
    """足球数据导入."""
    section_header("足球比赛导入", "导入历史赛程和赛果。")

    competitions = MatchRepository(database).competitions()
    if not competitions.empty:
        st.dataframe(competitions, hide_index=True, use_container_width=True)

    repository = MatchRepository(database)

    st.download_button(
        "📥 下载足球数据模板",
        MATCH_TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
        "matches_template.csv",
        "text/csv",
        key="fb-template",
    )

    upload = st.file_uploader("上传足球CSV", type=["csv"], key="fb-upload")
    if upload is not None:
        frame = pd.read_csv(upload)
        st.dataframe(frame.head(10), hide_index=True, use_container_width=True)
        errors = repository.validate(frame)
        if errors:
            for error in errors:
                st.error(error)
        elif st.button("写入数据库", type="primary", key="fb-import"):
            try:
                imported = repository.import_frame(frame, source=upload.name)
                st.success(f"成功写入 {imported} 场比赛")
                st.rerun()
            except (ValueError, TypeError) as error:
                st.error(f"导入失败：{error}")


def _render_p3_import(database: LocalDatabase) -> None:
    """排列三数据导入."""
    from football_model.lottery import LotteryRepository

    section_header("排列三开奖数据", "导入排列三历史开奖数据。")

    repo = LotteryRepository(database)
    draws = repo.get_p3_draws(limit=10)

    if not draws.empty:
        st.caption(f"已有 {len(repo.get_p3_draws(limit=99999))} 期数据")
        st.dataframe(draws.head(10), hide_index=True, use_container_width=True)

    # CSV导入
    st.markdown("**CSV导入**")
    st.download_button(
        "📥 下载CSV模板",
        P3_TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
        "p3_template.csv",
        "text/csv",
        key="p3-template-dl",
    )
    st.caption("CSV字段: issue_no, draw_date, digit_1, digit_2, digit_3")

    upload = st.file_uploader("上传排列三CSV", type=["csv"], key="p3-upload-dc")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)
            if st.button("确认导入CSV", key="p3-import-confirm", type="primary"):
                result = repo.import_p3_from_csv(upload)
                _render_import_result(result)
                if result.total > 0:
                    st.success(f"处理完成: 新增{result.success} 更新{result.updated}")
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")

    # JSON导入
    st.markdown("**JSON导入**")
    st.download_button(
        "📥 下载JSON模板",
        P3_JSON_TEMPLATE.encode("utf-8"),
        "p3_template.json",
        "application/json",
        key="p3-json-template",
    )

    json_upload = st.file_uploader("上传排列三JSON", type=["json"], key="p3-json-upload")
    if json_upload is not None:
        try:
            data = json.loads(json_upload.read())
            st.json(data[:3])
            if st.button("确认导入JSON", key="p3-json-import", type="primary"):
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(data, f)
                    temp_path = f.name
                count = repo.import_p3_from_json(temp_path)
                st.success(f"成功导入 {count.total} 期排列三数据")
                if count.errors:
                    _render_import_result(count)
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")


def _render_dlt_import(database: LocalDatabase) -> None:
    """大乐透数据导入."""
    from football_model.lottery import LotteryRepository

    section_header("大乐透开奖数据", "导入超级大乐透历史开奖数据。")

    repo = LotteryRepository(database)
    draws = repo.get_dlt_draws(limit=10)

    if not draws.empty:
        st.caption(f"已有 {len(repo.get_dlt_draws(limit=99999))} 期数据")
        st.dataframe(draws.head(10), hide_index=True, use_container_width=True)

    # CSV导入
    st.markdown("**CSV导入**")
    st.download_button(
        "📥 下载CSV模板",
        DLT_TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
        "dlt_template.csv",
        "text/csv",
        key="dlt-template-dl",
    )
    st.caption("CSV字段: issue_no, draw_date, front_1~front_5, back_1~back_2")

    upload = st.file_uploader("上传大乐透CSV", type=["csv"], key="dlt-upload-dc")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)
            if st.button("确认导入CSV", key="dlt-import-confirm-dc", type="primary"):
                result = repo.import_dlt_from_csv(upload)
                _render_import_result(result)
                if result.total > 0:
                    st.success(f"处理完成: 新增{result.success} 更新{result.updated}")
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")

    # JSON导入
    st.markdown("**JSON导入**")
    st.download_button(
        "📥 下载JSON模板",
        DLT_JSON_TEMPLATE.encode("utf-8"),
        "dlt_template.json",
        "application/json",
        key="dlt-json-template",
    )

    json_upload = st.file_uploader("上传大乐透JSON", type=["json"], key="dlt-json-upload")
    if json_upload is not None:
        try:
            data = json.loads(json_upload.read())
            st.json(data[:3])
            if st.button("确认导入JSON", key="dlt-json-import", type="primary"):
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(data, f)
                    temp_path = f.name
                count = repo.import_dlt_from_json(temp_path)
                st.success(f"成功导入 {count.total} 期大乐透数据")
                if count.errors:
                    _render_import_result(count)
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")


def _render_records(database: LocalDatabase) -> None:
    """本地记录."""
    section_header("本地记录", "查看已导入的比赛数据。")

    repository = MatchRepository(database)
    recent = repository.recent(limit=200)
    if recent.empty:
        empty_state("暂无比赛记录", "请先使用模板导入数据。", "📭")
    else:
        st.dataframe(recent, hide_index=True, use_container_width=True)
