from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    app_name: str = "绿茵智析"
    version: str = "0.7.2"
    database_name: str = "football.duckdb"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def database_path(self) -> Path:
        return self.data_dir / self.database_name

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.normalized_dir,
            self.features_dir,
            self.predictions_dir,
            self.artifacts_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings(project_root: str | Path | None = None) -> AppSettings:
    configured_root = project_root or os.environ.get("FOOTBALL_MODEL_HOME")
    root = Path(configured_root).resolve() if configured_root else Path(__file__).resolve().parents[3]
    settings = AppSettings(project_root=root)
    settings.ensure_directories()
    return settings
