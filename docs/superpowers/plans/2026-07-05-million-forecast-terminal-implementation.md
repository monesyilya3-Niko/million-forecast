# Million Forecast Terminal Repair and UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair runtime, data, lottery, navigation, and empty-state defects, then deliver a consistent financial-intelligence Streamlit interface without removing existing business functions.

**Architecture:** Preserve the current Streamlit → page → service/repository → DuckDB layering. Add test seams around database initialization, imports, navigation, page registration, and UI helpers; migrate presentation through a backward-compatible component library rather than rewriting football models.

**Tech Stack:** Python 3.13, Streamlit 1.58+, DuckDB 1.5+, Pandas 3.0+, NumPy 2.5+, Plotly 6.8+, pytest 9.1+, Ruff 0.15+

## Global Constraints

- Keep Streamlit; do not add React, Vue, external frontend frameworks, CDN resources, external fonts, or JavaScript.
- Keep all application data local under `D:\football-model`; bind only to `127.0.0.1:8502`.
- Preserve existing football, P3, and DLT business behavior unless a failing regression test proves it defective.
- Empty databases, unavailable providers, and absent optional data must render a stable empty state.
- Never fabricate odds, results, lineups, injuries, prizes, ROI, or provider status.
- Use neutral copy such as “候选组合” and “分析参考”; never claim guaranteed results, winnings, or returns.
- Back up `data/football.duckdb` before applying a production database migration.
- Every behavioral change follows RED → GREEN → full regression verification.

---

## File Map

- Create `tests/test_lottery_models.py`: P3/DLT derived metrics and validation behavior.
- Create `tests/test_lottery_imports.py`: atomic CSV/JSON imports and duplicate handling.
- Create `tests/test_lottery_services.py`: seeded generation, constraints, and finite attempts.
- Create `tests/test_navigation.py`: page history and filter restoration with fake session state.
- Create `tests/test_ui_components.py`: escaping, formatting, themes, and component contracts.
- Create `tests/test_page_registry.py`: page registration and unique navigation/widget identifiers.
- Create `src/football_model/ui/page_registry.py`: pure page metadata and registry construction, separated from Streamlit startup side effects.
- Create `tests/test_empty_pages.py`: render every page against a temporary empty DuckDB database.
- Create `tests/test_compliance_copy.py`: prohibited-copy and risk-disclaimer checks.
- Modify `src/football_model/lottery/models.py`: canonical lottery metrics.
- Modify `src/football_model/lottery/validators.py`: complete number/date/issue validation and quality reports.
- Modify `src/football_model/lottery/repositories.py`: validated transactional import results.
- Modify `src/football_model/lottery/services.py`: reproducible, bounded, non-duplicating generation.
- Modify `src/football_model/data/database.py`: idempotent schema/migrations and safe empty queries.
- Modify `src/football_model/ui/navigation.py`: deterministic stack navigation and filter snapshots.
- Modify `src/football_model/ui/components.py`: complete compatible design system and Plotly theme.
- Modify `src/football_model/ui/pages/*.py`: empty-state guards, stable keys, unified components, and risk copy.
- Modify `app.py`: declarative registry, fault-isolated rendering, and terminal sidebar.
- Modify `.streamlit/config.toml`: local dark theme and server settings only.
- Modify `README.md`: verified startup, import, and validation instructions.
- Create `reports/audit-2026-07-05.md`: final severity, remediation, evidence, and remaining limitations.

### Task 1: Freeze the Runtime and Audit Baseline

**Files:**
- Create: `tests/test_page_registry.py`
- Create: `reports/audit-2026-07-05.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: current `app.py` page imports and existing pytest suite.
- Produces: `EXPECTED_RENDERERS: set[str]` audit contract and a severity ledger used through all later tasks.

- [ ] **Step 1: Write the failing page-contract test**

```python
from pathlib import Path
import ast

EXPECTED_RENDERERS = {
    "render_live_matches", "render_recommendations", "render_results",
    "render_match_analysis", "render_match_detail", "render_single_match",
    "render_parlay", "render_batch", "render_model_center", "render_backtest",
    "render_data_center", "render_system_status", "render_p3_analysis",
    "render_p3_history", "render_p3_generator", "render_dlt_analysis",
    "render_dlt_history", "render_dlt_generator", "render_lottery_backtest",
}

def test_every_expected_page_renderer_exists():
    found = set()
    for path in Path("src/football_model/ui/pages").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.update(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
    assert EXPECTED_RENDERERS <= found
```

- [ ] **Step 2: Run the contract and record the baseline**

Run: `.venv\Scripts\python.exe -m pytest tests/test_page_registry.py -v`

Expected: PASS for existing renderers; if it passes immediately, add the missing registry assertion in Task 7 before changing application code.

- [ ] **Step 3: Create the audit ledger with verified baseline facts**

```markdown
# Audit — 2026-07-05

| ID | Severity | Area | Evidence | Planned task | Fixed |
|---|---|---|---|---|---|
| AUD-001 | High | Lottery tests | No lottery references under tests/ | 2–4 | No |
| AUD-002 | High | Page runtime | Existing tests do not render all pages on empty DB | 8 | No |
| AUD-003 | Medium | Widgets | Several interactive widgets omit explicit keys | 5, 7, 8 | No |
| AUD-004 | High | Imports | Lottery CSV imports save row-by-row without full-file validation | 3 | No |
```

- [ ] **Step 4: Run the unchanged full baseline**

Run: `.venv\Scripts\python.exe -m ruff check app.py src tests && .venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`

Expected: `All checks passed!` and `23 passed` after adding the contract test.

- [ ] **Step 5: Commit the audit harness**

```powershell
git add tests/test_page_registry.py reports/audit-2026-07-05.md .gitignore
git commit -m "test: establish application audit baseline"
```

### Task 2: Make Lottery Domain Rules Explicit

**Files:**
- Create: `tests/test_lottery_models.py`
- Modify: `src/football_model/lottery/models.py`
- Modify: `src/football_model/lottery/validators.py`

**Interfaces:**
- Produces: `validate_p3_draw(...) -> ValidationResult`, `validate_dlt_draw(...) -> ValidationResult`, sorted `DLTDraw.front_numbers` and `DLTDraw.back_numbers`.

- [ ] **Step 1: Write failing P3 and DLT validation tests**

```python
import pytest
from football_model.lottery.models import DLTDraw, P3Draw
from football_model.lottery.validators import validate_dlt_draw, validate_p3_draw

@pytest.mark.parametrize("digits", [(-1, 2, 3), (1, 10, 3)])
def test_p3_rejects_digits_outside_zero_to_nine(digits):
    result = validate_p3_draw(*digits, issue_no="2026001", draw_date="2026-01-01")
    assert not result.is_valid

def test_p3_metrics_are_consistent():
    draw = P3Draw("2026001", "2026-01-01", 1, 1, 3)
    assert (draw.sum_value, draw.span_value, draw.pattern_type) == (5, 2, "组三")
    assert (draw.odd_count, draw.even_count, draw.big_count, draw.small_count) == (3, 0, 0, 3)

def test_dlt_rejects_duplicate_and_unsorted_numbers():
    duplicate = validate_dlt_draw([1, 2, 2, 4, 5], [1, 1], "2026001", "2026-01-01")
    unsorted = validate_dlt_draw([5, 4, 3, 2, 1], [2, 1], "2026001", "2026-01-01")
    assert not duplicate.is_valid
    assert not unsorted.is_valid
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_models.py -v`

Expected: FAIL because unsorted DLT input is currently accepted.

- [ ] **Step 3: Implement canonical validators without changing public signatures**

```python
def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True

def validate_dlt_draw(front, back, issue_no, draw_date):
    errors: list[str] = []
    if len(front) != 5 or len(set(front)) != 5 or any(not 1 <= n <= 35 for n in front):
        errors.append("前区必须是 5 个不重复的 1–35 号码")
    if front != sorted(front):
        errors.append("前区号码必须升序")
    if len(back) != 2 or len(set(back)) != 2 or any(not 1 <= n <= 12 for n in back):
        errors.append("后区必须是 2 个不重复的 1–12 号码")
    if back != sorted(back):
        errors.append("后区号码必须升序")
    if not str(issue_no).strip():
        errors.append("期号不能为空")
    if not _is_iso_date(draw_date):
        errors.append("开奖日期必须为 YYYY-MM-DD")
    return ValidationResult(not errors, errors, [])
```

- [ ] **Step 4: Verify GREEN and regression**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_models.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_lottery_models.py src/football_model/lottery/models.py src/football_model/lottery/validators.py
git commit -m "fix: enforce lottery domain rules"
```

### Task 3: Make Imports Validated and Atomic

**Files:**
- Create: `tests/test_lottery_imports.py`
- Modify: `src/football_model/lottery/repositories.py`
- Modify: `src/football_model/data/database.py`
- Modify: `src/football_model/ui/pages/data_center.py`

**Interfaces:**
- Produces: `ImportResult(inserted: int, replaced: int, rejected: int, errors: tuple[str, ...])`; `import_p3_frame(frame)` and `import_dlt_frame(frame)`.

- [ ] **Step 1: Write a failing all-or-nothing import test**

```python
import pandas as pd
from football_model.data import LocalDatabase
from football_model.lottery.repositories import LotteryRepository

def test_invalid_row_prevents_partial_p3_import(tmp_path):
    db = LocalDatabase(tmp_path / "test.duckdb")
    db.initialize()
    repo = LotteryRepository(db)
    frame = pd.DataFrame([
        {"issue_no": "1", "draw_date": "2026-01-01", "digit_1": 1, "digit_2": 2, "digit_3": 3},
        {"issue_no": "2", "draw_date": "bad", "digit_1": 1, "digit_2": 2, "digit_3": 30},
    ])
    result = repo.import_p3_frame(frame)
    assert result.inserted == 0
    assert result.rejected == 1
    assert repo.get_p3_draws().empty
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_imports.py -v`

Expected: FAIL with missing `import_p3_frame`.

- [ ] **Step 3: Implement parse/validate/write phases**

```python
@dataclass(frozen=True)
class ImportResult:
    inserted: int
    replaced: int
    rejected: int
    errors: tuple[str, ...]

def import_p3_frame(self, frame: pd.DataFrame) -> ImportResult:
    required = {"issue_no", "draw_date", "digit_1", "digit_2", "digit_3"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return ImportResult(0, 0, len(frame), (f"缺少字段: {', '.join(missing)}",))
    draws, errors = [], []
    for row_no, row in enumerate(frame.to_dict("records"), start=2):
        try:
            draw = P3Draw(str(row["issue_no"]).strip(), str(row["draw_date"]), int(row["digit_1"]), int(row["digit_2"]), int(row["digit_3"]))
            check = validate_p3_draw(draw.digit_1, draw.digit_2, draw.digit_3, draw.issue_no, draw.draw_date)
        except (TypeError, ValueError) as error:
            errors.append(f"第 {row_no} 行: {error}")
            continue
        if not check.is_valid:
            errors.extend(f"第 {row_no} 行: {message}" for message in check.errors)
        draws.append(draw)
    if errors:
        return ImportResult(0, 0, len(errors), tuple(errors))
    with self.database.connection() as conn:
        conn.begin()
        try:
            for draw in draws:
                self._save_p3(conn, draw)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return ImportResult(len(draws), 0, 0, ())
```

- [ ] **Step 4: Route CSV and JSON uploads through the frame importer**

Use `pd.read_csv(upload)` or `pd.DataFrame(json.load(upload))`, show `result.errors`, and only show success when `result.inserted > 0`.

- [ ] **Step 5: Verify GREEN and database idempotency**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_imports.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS; repeated `LocalDatabase.initialize()` does not change table counts.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_lottery_imports.py src/football_model/lottery/repositories.py src/football_model/data/database.py src/football_model/ui/pages/data_center.py
git commit -m "fix: validate lottery imports transactionally"
```

### Task 4: Bound and Reproduce Lottery Generation and Backtests

**Files:**
- Create: `tests/test_lottery_services.py`
- Modify: `src/football_model/lottery/services.py`
- Modify: `src/football_model/ui/pages/p3_generator.py`
- Modify: `src/football_model/ui/pages/dlt_generator.py`
- Modify: `src/football_model/ui/pages/lottery_backtest.py`

**Interfaces:**
- Produces: `generate_reference_combinations(..., seed: int | None = None) -> list[dict]` for both services.

- [ ] **Step 1: Write failing determinism and uniqueness tests**

```python
def test_p3_generation_is_seeded_and_unique(p3_service, empty_draws):
    first = p3_service.generate_reference_combinations(empty_draws, count=20, seed=42)
    second = p3_service.generate_reference_combinations(empty_draws, count=20, seed=42)
    assert first == second
    assert len({item["号码"] for item in first}) == len(first)

def test_dlt_generation_is_seeded_and_unique(dlt_service, empty_draws):
    result = dlt_service.generate_reference_combinations(empty_draws, count=20, seed=42)
    keys = {(item["前区"], item["后区"]) for item in result}
    assert len(keys) == len(result)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_services.py -v`

Expected: FAIL because `seed` is not accepted and P3 duplicates are possible.

- [ ] **Step 3: Use one seeded generator and a seen set**

```python
rng = np.random.default_rng(seed)
seen: set[tuple[int, ...]] = set()
max_attempts = max(100, count * 200)
# Reject a tuple already in `seen`; add it only after every filter passes.
```

- [ ] **Step 4: Enforce time-causal backtests**

For evaluation index `i`, create training data using `draws.iloc[i + 1:]` when frames are newest-first; assert the current row is absent. If prize columns are absent, return `roi=None` and render “缺少真实奖金数据，未计算 ROI”.

- [ ] **Step 5: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lottery_services.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_lottery_services.py src/football_model/lottery/services.py src/football_model/ui/pages/p3_generator.py src/football_model/ui/pages/dlt_generator.py src/football_model/ui/pages/lottery_backtest.py
git commit -m "fix: make lottery research reproducible and causal"
```

### Task 5: Make Navigation Deterministic

**Files:**
- Create: `tests/test_navigation.py`
- Modify: `src/football_model/ui/navigation.py`
- Modify: `src/football_model/ui/pages/live_matches.py`
- Modify: `src/football_model/ui/pages/recommendations.py`
- Modify: `src/football_model/ui/pages/match_detail.py`
- Modify: `src/football_model/ui/pages/p3_history.py`
- Modify: `src/football_model/ui/pages/dlt_history.py`

**Interfaces:**
- Produces: `navigate_to(page_name, context=None, set_return=True)`, `go_back(default_page)`, `remember_filter`, `restore_filter` with a true LIFO history.

- [ ] **Step 1: Write failing history and filter tests with a fake Streamlit state**

```python
def test_back_uses_page_history_before_default(monkeypatch):
    state = {"nav_page": "比赛详情", "page_history": ["今日竞彩", "比赛分析"], "page_filters": {}}
    monkeypatch.setattr(navigation.st, "session_state", state)
    monkeypatch.setattr(navigation.st, "rerun", lambda: None)
    navigation.go_back()
    assert state["nav_page"] == "比赛分析"
    assert state["page_history"] == ["今日竞彩"]
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_navigation.py -v`

Expected: FAIL because current `go_back` ignores `page_history`.

- [ ] **Step 3: Implement one source of truth**

```python
def go_back(default_page: str = DEFAULT_PAGE) -> None:
    _init_state()
    history = list(st.session_state[PAGE_HISTORY])
    target = history.pop() if history else default_page
    st.session_state[PAGE_HISTORY] = history
    st.session_state[NAV_PAGE] = target
    st.session_state.pop(RETURN_PAGE, None)
    st.rerun()
```

- [ ] **Step 4: Restore filters as widget defaults before constructing widgets**

Use `restore_filter("live", "business_date", default)` for the initial value and call `remember_filter` immediately after each selection. Add explicit page-prefixed keys to every touched widget.

- [ ] **Step 5: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_navigation.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_navigation.py src/football_model/ui/navigation.py src/football_model/ui/pages/live_matches.py src/football_model/ui/pages/recommendations.py src/football_model/ui/pages/match_detail.py src/football_model/ui/pages/p3_history.py src/football_model/ui/pages/dlt_history.py
git commit -m "fix: preserve navigation and filter state"
```

### Task 6: Complete the Backward-Compatible Design System

**Files:**
- Create: `tests/test_ui_components.py`
- Modify: `src/football_model/ui/components.py`
- Modify: `.streamlit/config.toml`

**Interfaces:**
- Produces all component signatures listed in the approved design; preserves `hero`, `probability_chart`, `score_heatmap`, and `format_percent_columns`.

- [ ] **Step 1: Write failing pure-helper tests**

```python
from football_model.ui.components import format_number, format_percent, safe_html

def test_safe_html_escapes_markup():
    assert safe_html('<script>alert("x")</script>') == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"

def test_formatters_handle_none_nan_and_infinity():
    assert format_number(None) == "-"
    assert format_number(float("nan")) == "-"
    assert format_percent(float("inf")) == "-"
```

- [ ] **Step 2: Verify RED for missing component contracts**

Add imports for `data_quality_panel`, `provider_status_panel`, `risk_panel`, `odds_box`, `probability_triplet`, `lottery_number_row`, `candidate_combo_card`, and `backtest_summary_card`.

Run: `.venv\Scripts\python.exe -m pytest tests/test_ui_components.py -v`

Expected: collection FAIL until missing functions exist.

- [ ] **Step 3: Implement components with escaped dynamic content**

```python
def odds_box(label: str, value: object, trend: float | None = None) -> str:
    trend_class = "up" if trend and trend > 0 else "down" if trend and trend < 0 else "flat"
    trend_text = "" if trend is None else f"{trend:+.2f}"
    return (
        f'<div class="odds-box {trend_class}">'
        f'<div class="odds-label">{safe_html(label)}</div>'
        f'<div class="odds-value">{format_number(value)}</div>'
        f'<div class="odds-trend">{safe_html(trend_text)}</div></div>'
    )
```

Every component returns escaped HTML or renders through `st.markdown`; no component injects scripts, remote URLs, or fonts.

- [ ] **Step 4: Apply the selected A visual tokens**

Use `#030712`, `#07111f`, `#0f172a`, low-alpha slate borders, cyan/blue brand accents, and semantic green/yellow/red only. Set Streamlit dark theme and `server.address = "127.0.0.1"`, `server.port = 8502`.

- [ ] **Step 5: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ui_components.py -v && .venv\Scripts\python.exe -m ruff check src/football_model/ui/components.py`

Expected: PASS and no Ruff findings.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_ui_components.py src/football_model/ui/components.py .streamlit/config.toml
git commit -m "feat: add financial terminal design system"
```

### Task 7: Fault-Isolate the App Shell and Sidebar

**Files:**
- Modify: `app.py`
- Create: `src/football_model/ui/page_registry.py`
- Modify: `tests/test_page_registry.py`

**Interfaces:**
- Produces: immutable `PageSpec` records, `build_page_specs(renderers) -> tuple[PageSpec, ...]`, and `render_current_page(registry, current_page)` fault boundary.

- [ ] **Step 1: Add a failing registry uniqueness test**

```python
def test_page_names_and_navigation_keys_are_unique():
    specs = build_page_specs(fake_renderers())
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.key for spec in specs}) == len(specs)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_page_registry.py -v`

Expected: FAIL because `PageSpec` and `build_page_specs` do not exist.

- [ ] **Step 3: Introduce declarative page records**

```python
@dataclass(frozen=True)
class PageSpec:
    key: str
    section: str
    name: str
    icon: str
    render: Callable[[], None]

def build_page_specs(renderers: Mapping[str, Callable[[], None]]) -> tuple[PageSpec, ...]:
    definitions = (
        ("live-today", "实时中心", "今日竞彩", "●", "live"),
        ("football-analysis", "足球分析", "比赛分析", "◇", "analysis"),
        ("p3-analysis", "数字彩票", "排列三分析", "③", "p3"),
        ("dlt-analysis", "数字彩票", "大乐透分析", "⑦", "dlt"),
    )
    return tuple(PageSpec(key, section, name, icon, renderers[renderer]) for key, section, name, icon, renderer in definitions)
```

Build keys from stable ASCII identifiers such as `live-today`, `football-analysis`, `p3-history`, and `system-status`; do not derive keys from translated labels.

- [ ] **Step 4: Add a page-level fault boundary**

Catch only expected application exceptions (`OSError`, `RuntimeError`, `ValueError`, DuckDB errors), log with `logger.exception`, and render a standard error/empty state. Do not use a blanket silent `except Exception` around the whole app.

- [ ] **Step 5: Build the sidebar from status data**

Render product identity, `Football · P3 · DLT`, `127.0.0.1:8502`, DB health, football provider, lottery provider, and last sync. Use “未配置/无缓存” when evidence is absent.

- [ ] **Step 6: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_page_registry.py -v && .venv\Scripts\python.exe -m ruff check app.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app.py src/football_model/ui/page_registry.py tests/test_page_registry.py
git commit -m "refactor: isolate pages behind terminal app shell"
```

### Task 8: Make Every Page Safe on Empty Data and Migrate the UI

**Files:**
- Create: `tests/test_empty_pages.py`
- Modify: `src/football_model/ui/pages/*.py`

**Interfaces:**
- Consumes: Task 6 component APIs and Task 7 page registry.
- Produces: every registered renderer completes against an initialized empty database without `KeyError`, `AttributeError`, `ImportError`, or duplicate widget IDs.

- [ ] **Step 1: Write the empty-database smoke harness**

```python
def test_registered_pages_render_with_empty_database(empty_app_dependencies, monkeypatch):
    calls = []
    monkeypatch.setattr(st, "markdown", lambda *a, **k: calls.append("markdown"))
    monkeypatch.setattr(st, "dataframe", lambda *a, **k: calls.append("dataframe"))
    for spec in build_page_specs(empty_app_dependencies):
        spec.render()
    assert calls
```

Provide lightweight Streamlit container doubles that implement context management and return valid default widget values; do not mock service business calculations.

- [ ] **Step 2: Verify RED and list each failing renderer in the audit report**

Run: `.venv\Scripts\python.exe -m pytest tests/test_empty_pages.py -v --maxfail=20`

Expected: FAIL with the first concrete empty-state or widget issue.

- [ ] **Step 3: Fix pages in dependency order**

For each renderer, add:

```python
hero_pro(title, subtitle, eyebrow="MILLION FORECAST TERMINAL", meta=[source_label])
if frame.empty:
    empty_state("暂无可分析数据", "请先前往数据中心导入或同步数据。")
    render_risk_note(domain_disclaimer)
    return
required = {"expected", "columns"}
missing = required - set(frame.columns)
if missing:
    empty_state("数据结构不完整", f"缺少字段：{', '.join(sorted(missing))}")
    return
```

Run the single smoke test after each repaired page, then proceed to the next page. Replace raw metrics/cards with Task 6 components while preserving all controls and service calls.

- [ ] **Step 4: Add explicit stable keys to every interactive widget**

Use page prefixes: `single-*`, `parlay-*`, `batch-*`, `backtest-*`, `model-*`, `p3-*`, `dlt-*`, and `data-*`. Assert AST-level duplicate literal keys are absent.

- [ ] **Step 5: Verify all empty pages and regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/test_empty_pages.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit page groups separately**

```powershell
git add tests/test_empty_pages.py src/football_model/ui/pages/live_matches.py src/football_model/ui/pages/match_analysis.py src/football_model/ui/pages/match_detail.py src/football_model/ui/pages/recommendations.py
git commit -m "feat: upgrade football intelligence pages"
git add src/football_model/ui/pages/p3_*.py src/football_model/ui/pages/dlt_*.py src/football_model/ui/pages/lottery_backtest.py
git commit -m "feat: upgrade lottery research pages"
git add src/football_model/ui/pages
git commit -m "feat: harden remaining terminal pages"
```

### Task 9: Enforce Compliance Copy and Unified Risk Panels

**Files:**
- Create: `tests/test_compliance_copy.py`
- Modify: `src/football_model/ui/components.py`
- Modify: `src/football_model/ui/pages/*.py`

**Interfaces:**
- Produces: `FOOTBALL_RISK_TEXT`, `LOTTERY_RISK_TEXT`, and no prohibited promise copy in application-facing source.

- [ ] **Step 1: Write the failing source scan**

```python
from pathlib import Path
import re

PROHIBITED = re.compile(r"必中|稳赚|包中|保赢|稳中|预测中奖|杀号必中|绝杀|包赔|无风险")

def test_application_copy_contains_no_guaranteed_outcome_claims():
    offenders = []
    for path in [Path("app.py"), *Path("src/football_model/ui").rglob("*.py")]:
        if PROHIBITED.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == []
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_compliance_copy.py -v`

Expected: FAIL if any prohibited copy remains; otherwise add disclaimer-presence assertions that fail before page migration is complete.

- [ ] **Step 3: Centralize exact disclaimers**

Define the approved football and lottery disclaimers once in `components.py`; render the correct disclaimer at the bottom of every corresponding page through `render_risk_note` or `lottery_risk_disclaimer`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_compliance_copy.py -v && .venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_compliance_copy.py src/football_model/ui/components.py src/football_model/ui/pages
git commit -m "fix: standardize risk and compliance copy"
```

### Task 10: Full Runtime, Browser, and Delivery Verification

**Files:**
- Modify: `README.md`
- Modify: `reports/audit-2026-07-05.md`

**Interfaces:**
- Produces: reproducible startup/import/verification documentation and final evidence ledger.

- [ ] **Step 1: Back up the production database before migration verification**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Copy-Item -LiteralPath data\football.duckdb -Destination "data\football.duckdb.$stamp.bak"
```

Expected: timestamped backup exists beside the database.

- [ ] **Step 2: Run fresh compile, lint, and full tests**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.venv\Scripts\python.exe -c "from pathlib import Path; files=[Path('app.py'),*Path('src').rglob('*.py')]; [compile(p.read_text(encoding='utf-8'),str(p),'exec') for p in files]; print(f'compiled={len(files)}')"
.venv\Scripts\python.exe -m ruff check app.py src tests
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: all source files compile, Ruff reports `All checks passed!`, pytest reports zero failures.

- [ ] **Step 3: Verify an independent empty database**

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from tempfile import TemporaryDirectory; from football_model.data import LocalDatabase; t=TemporaryDirectory(); db=LocalDatabase(Path(t.name)/'empty.duckdb'); db.initialize(); assert db.health_check(); print(db.table_counts())"
```

Expected: exit 0 and all required tables listed with zero rows.

- [ ] **Step 4: Start Streamlit and probe health**

Run Streamlit on `127.0.0.1:8502`, wait until `http://127.0.0.1:8502/_stcore/health` returns `ok`, then inspect the process log for tracebacks.

Expected: health endpoint responds successfully and startup log contains no traceback.

- [ ] **Step 5: Perform browser acceptance**

Open every navigation item. Verify today filters survive detail/back navigation; P3/DLT empty pages render; CSV/JSON templates download; invalid input is rejected; legal input reaches confirmation; charts use the unified dark theme; no page shows a guaranteed-result claim.

- [ ] **Step 6: Update audit and README with evidence**

For each audit row, set `Fixed` to `Yes` only when a named test or browser action proves it. Document startup, P3/DLT import schemas, football/lottery checks, navigation verification, backtest limitations, real-provider dependencies, and exact test counts.

- [ ] **Step 7: Inspect the final diff and commit**

```powershell
git status --short
git diff --check
git diff --stat HEAD~10..HEAD
git add README.md reports/audit-2026-07-05.md
git commit -m "docs: publish terminal audit and verification guide"
```

Expected: no whitespace errors, no database or backup files staged, and documentation matches fresh command output.
