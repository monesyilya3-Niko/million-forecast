"""Train all models for all leagues."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

from football_model.data import LocalDatabase, MatchRepository  # noqa: E402
from football_model.core import get_settings  # noqa: E402
from football_model.models.dixon_coles import DixonColesModel  # noqa: E402
from football_model.models.poisson import PoissonModel  # noqa: E402
from football_model.models.elo import EloModel  # noqa: E402
from football_model.models.xgboost_model import XGBoostModel  # noqa: E402
from football_model.models.neural_net import NeuralNetModel  # noqa: E402
from football_model.models.random_forest import RandomForestModel  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings(Path(__file__).resolve().parent.parent)
db = LocalDatabase(settings.database_path)
repo = MatchRepository(db)

COMPETITIONS = [
    "英格兰超级联赛",
    "西班牙甲级联赛",
    "意大利甲级联赛",
    "德国甲级联赛",
    "法国甲级联赛",
    "瑞典超级联赛",
    "世界杯国家队",
]

MAXITER = 100_000
ALL_RESULTS = []


def train_dc(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    model = DixonColesModel.fit(frame, competition=comp, maxiter=MAXITER)
    m = model.metrics
    model.save(settings.artifacts_dir / "dixon_coles" / f"dc_{comp}_v2.json")
    return {"type": "Dixon-Coles", "comp": comp, "metrics": m, "time": time.time() - t0}


def train_poisson(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    frame["cutoff_at"] = frame["kickoff"]
    model = PoissonModel.fit(frame, frame, competition=comp)
    m = model.metrics
    model.save(settings.artifacts_dir / "poisson" / f"poisson_{comp}_v2.json")
    return {"type": "Poisson", "comp": comp, "metrics": m, "time": time.time() - t0}


def train_elo(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    model = EloModel.fit(frame, competition=comp, k_factor=20)
    m = model.metrics
    model.save(settings.artifacts_dir / "elo" / f"elo_{comp}_v1.json")
    return {"type": "Elo", "comp": comp, "metrics": m, "time": time.time() - t0}


def train_xgboost(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    frame["cutoff_at"] = frame["kickoff"]
    model = XGBoostModel.fit(frame, frame, competition=comp, n_estimators=300, max_depth=6)
    m = model.metrics
    return {"type": "XGBoost", "comp": comp, "metrics": m, "time": time.time() - t0}


def train_nn(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    frame["cutoff_at"] = frame["kickoff"]
    model = NeuralNetModel.fit(frame, frame, competition=comp)
    m = model.metrics
    return {"type": "NeuralNet", "comp": comp, "metrics": m, "time": time.time() - t0}


def train_rf(comp: str) -> dict:
    t0 = time.time()
    frame = repo.training_frame(comp)
    frame["cutoff_at"] = frame["kickoff"]
    model = RandomForestModel.fit(frame, frame, competition=comp, n_estimators=300, max_depth=8)
    m = model.metrics
    return {"type": "RandomForest", "comp": comp, "metrics": m, "time": time.time() - t0}


MODELS = {
    "Dixon-Coles": train_dc,
    "Poisson": train_poisson,
    "Elo": train_elo,
    "XGBoost": train_xgboost,
    "NeuralNet": train_nn,
    "RandomForest": train_rf,
}


if __name__ == "__main__":
    for comp in COMPETITIONS:
        print(f"\n{'='*70}")
        print(f"联赛: {comp}")
        print(f"{'='*70}")
        for name, fn in MODELS.items():
            try:
                result = fn(comp)
                m = result["metrics"]
                ll = m.get("holdout_log_loss", "N/A")
                acc = m.get("holdout_accuracy", "N/A")
                print(f"  {name:<15} LL={ll}  Acc={acc}  time={result['time']:.1f}s")
                ALL_RESULTS.append(result)
            except Exception as e:
                print(f"  {name:<15} FAILED: {e}")
                ALL_RESULTS.append({"type": name, "comp": comp, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")
    print(f"{'联赛':<16} {'模型':<15} {'Holdout LL':>12} {'准确率':>8} {'耗时':>6}")
    print("-" * 65)
    for r in ALL_RESULTS:
        if "error" in r:
            print(f"{r['comp']:<16} {r['type']:<15} {'FAILED':>12}")
            continue
        m = r["metrics"]
        ll = m.get("holdout_log_loss", "N/A")
        acc = m.get("holdout_accuracy", "N/A")
        ll_s = f"{ll:.4f}" if isinstance(ll, float) else str(ll)
        acc_s = f"{acc:.1%}" if isinstance(acc, float) else str(acc)
        print(f"{r['comp']:<16} {r['type']:<15} {ll_s:>12} {acc_s:>8} {r['time']:>5.0f}s")

    with open(settings.data_dir / "training_results_all.json", "w", encoding="utf-8") as f:
        json.dump(ALL_RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到 {settings.data_dir / 'training_results_all.json'}")
