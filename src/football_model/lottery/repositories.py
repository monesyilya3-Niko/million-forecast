"""Lottery data repositories."""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from football_model.data import LocalDatabase
from .models import P3Draw, DLTDraw

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Detailed import result."""
    success: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.success + self.updated + self.skipped

    @property
    def error_count(self) -> int:
        return len(self.errors)


class LotteryRepository:
    """Repository for lottery draw data."""

    def __init__(self, database: LocalDatabase) -> None:
        self.database = database

    def _p3_exists(self, conn, issue_no: str) -> bool:
        row = conn.execute("SELECT 1 FROM p3_draws WHERE issue_no = ?", [issue_no]).fetchone()
        return row is not None

    def _dlt_exists(self, conn, issue_no: str) -> bool:
        row = conn.execute("SELECT 1 FROM dlt_draws WHERE issue_no = ?", [issue_no]).fetchone()
        return row is not None

    # ── P3 ──────────────────────────────────────────────────────

    def get_p3_draws(self, limit: int = 100) -> pd.DataFrame:
        """Get P3 draws."""
        with self.database.connection(read_only=True) as conn:
            return conn.execute(
                "SELECT * FROM p3_draws ORDER BY issue_no DESC LIMIT ?",
                [limit],
            ).df()

    def get_p3_by_issue(self, issue_no: str) -> P3Draw | None:
        """Get single P3 draw by issue number."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT issue_no, draw_date, digit_1, digit_2, digit_3 FROM p3_draws WHERE issue_no = ?",
                [issue_no],
            ).fetchone()
        if not row:
            return None
        return P3Draw(issue_no=row[0], draw_date=str(row[1]), digit_1=row[2], digit_2=row[3], digit_3=row[4])

    def save_p3_draw(self, draw: P3Draw) -> None:
        """Save a P3 draw."""
        with self.database.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO p3_draws
                (issue_no, draw_date, digit_1, digit_2, digit_3, number_text,
                 sum_value, span_value, odd_count, even_count, big_count, small_count,
                 pattern_type, road_012, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', CURRENT_TIMESTAMP)""",
                [
                    draw.issue_no, draw.draw_date, draw.digit_1, draw.digit_2, draw.digit_3,
                    draw.number_text, draw.sum_value, draw.span_value,
                    draw.odd_count, draw.even_count, draw.big_count, draw.small_count,
                    draw.pattern_type, draw.road_012,
                ],
            )

    def import_p3_from_csv(self, filepath: str | Path) -> ImportResult:
        """Import P3 draws from CSV with detailed result."""
        df = pd.read_csv(filepath)
        required = {"issue_no", "draw_date", "digit_1", "digit_2", "digit_3"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV缺少字段: {missing}")

        result = ImportResult()
        with self.database.connection() as conn:
            for idx, row in df.iterrows():
                try:
                    issue_no = str(row["issue_no"])
                    is_update = self._p3_exists(conn, issue_no)
                    draw = P3Draw(
                        issue_no=issue_no,
                        draw_date=str(row["draw_date"]),
                        digit_1=int(row["digit_1"]),
                        digit_2=int(row["digit_2"]),
                        digit_3=int(row["digit_3"]),
                    )
                    self.save_p3_draw(draw)
                    if is_update:
                        result.updated += 1
                    else:
                        result.success += 1
                except Exception as e:
                    result.errors.append(f"第{idx+1}行: {e}")
        return result

    # ── DLT ─────────────────────────────────────────────────────

    def get_dlt_draws(self, limit: int = 100) -> pd.DataFrame:
        """Get DLT draws."""
        with self.database.connection(read_only=True) as conn:
            return conn.execute(
                "SELECT * FROM dlt_draws ORDER BY issue_no DESC LIMIT ?",
                [limit],
            ).df()

    def get_dlt_by_issue(self, issue_no: str) -> DLTDraw | None:
        """Get single DLT draw by issue number."""
        with self.database.connection(read_only=True) as conn:
            row = conn.execute(
                """SELECT issue_no, draw_date, front_1, front_2, front_3, front_4, front_5,
                   back_1, back_2 FROM dlt_draws WHERE issue_no = ?""",
                [issue_no],
            ).fetchone()
        if not row:
            return None
        return DLTDraw(
            issue_no=row[0], draw_date=str(row[1]),
            front_1=row[2], front_2=row[3], front_3=row[4], front_4=row[5], front_5=row[6],
            back_1=row[7], back_2=row[8],
        )

    def save_dlt_draw(self, draw: DLTDraw) -> None:
        """Save a DLT draw."""
        with self.database.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO dlt_draws
                (issue_no, draw_date, front_1, front_2, front_3, front_4, front_5,
                 back_1, back_2, front_sum, back_sum, front_span, back_span,
                 front_odd_count, front_even_count, zone_1_count, zone_2_count, zone_3_count,
                 source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', CURRENT_TIMESTAMP)""",
                [
                    draw.issue_no, draw.draw_date,
                    draw.front_1, draw.front_2, draw.front_3, draw.front_4, draw.front_5,
                    draw.back_1, draw.back_2,
                    draw.front_sum, draw.back_sum, draw.front_span, draw.back_span,
                    draw.front_odd_count, draw.front_even_count,
                    draw.zone_counts[0], draw.zone_counts[1], draw.zone_counts[2],
                ],
            )

    def import_dlt_from_csv(self, filepath: str | Path) -> ImportResult:
        """Import DLT draws from CSV with detailed result."""
        df = pd.read_csv(filepath)
        required = {"issue_no", "draw_date", "front_1", "front_2", "front_3", "front_4", "front_5", "back_1", "back_2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV缺少字段: {missing}")

        result = ImportResult()
        with self.database.connection() as conn:
            for idx, row in df.iterrows():
                try:
                    issue_no = str(row["issue_no"])
                    is_update = self._dlt_exists(conn, issue_no)
                    draw = DLTDraw(
                        issue_no=issue_no,
                        draw_date=str(row["draw_date"]),
                        front_1=int(row["front_1"]), front_2=int(row["front_2"]),
                        front_3=int(row["front_3"]), front_4=int(row["front_4"]),
                        front_5=int(row["front_5"]),
                        back_1=int(row["back_1"]), back_2=int(row["back_2"]),
                    )
                    self.save_dlt_draw(draw)
                    if is_update:
                        result.updated += 1
                    else:
                        result.success += 1
                except Exception as e:
                    result.errors.append(f"第{idx+1}行: {e}")
        return result

    def import_p3_from_json(self, filepath: str | Path) -> ImportResult:
        """Import P3 draws from JSON file with detailed result."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = ImportResult()
        with self.database.connection() as conn:
            for idx, item in enumerate(data):
                try:
                    numbers = item.get("numbers", [])
                    if len(numbers) != 3:
                        result.skipped += 1
                        result.errors.append(f"第{idx+1}项: 号码数量不是3个")
                        continue
                    issue_no = str(item["issue_no"])
                    is_update = self._p3_exists(conn, issue_no)
                    draw = P3Draw(
                        issue_no=issue_no,
                        draw_date=str(item["draw_date"]),
                        digit_1=int(numbers[0]),
                        digit_2=int(numbers[1]),
                        digit_3=int(numbers[2]),
                    )
                    self.save_p3_draw(draw)
                    if is_update:
                        result.updated += 1
                    else:
                        result.success += 1
                except Exception as e:
                    result.errors.append(f"第{idx+1}项: {e}")
        return result

    def import_dlt_from_json(self, filepath: str | Path) -> ImportResult:
        """Import DLT draws from JSON file with detailed result."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = ImportResult()
        with self.database.connection() as conn:
            for idx, item in enumerate(data):
                try:
                    front = item.get("front", [])
                    back = item.get("back", [])
                    if len(front) != 5 or len(back) != 2:
                        result.skipped += 1
                        result.errors.append(f"第{idx+1}项: 前区应5个后区应2个")
                        continue
                    issue_no = str(item["issue_no"])
                    is_update = self._dlt_exists(conn, issue_no)
                    draw = DLTDraw(
                        issue_no=issue_no,
                        draw_date=str(item["draw_date"]),
                        front_1=int(front[0]), front_2=int(front[1]),
                        front_3=int(front[2]), front_4=int(front[3]),
                        front_5=int(front[4]),
                        back_1=int(back[0]), back_2=int(back[1]),
                    )
                    self.save_dlt_draw(draw)
                    if is_update:
                        result.updated += 1
                    else:
                        result.success += 1
                except Exception as e:
                    result.errors.append(f"第{idx+1}项: {e}")
        return result
