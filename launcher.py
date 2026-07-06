from __future__ import annotations

import logging
import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
URL = "http://127.0.0.1:8502"
HEALTH_URL = f"{URL}/_stcore/health"
LOG_PATH = ROOT / "reports" / "server.log"
UPDATER_LOG_PATH = ROOT / "reports" / "live-updater.log"
UPDATER_PID_PATH = ROOT / "reports" / "live-updater.pid"


def is_healthy() -> bool:
    """Check if the Streamlit server is healthy."""
    logger.debug(f"Checking health at {HEALTH_URL}")
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            return response.status == 200 and response.read().decode("utf-8").strip() == "ok"
    except (OSError, urllib.error.URLError):
        return False


def start_server() -> subprocess.Popen[bytes]:
    if not PYTHON.exists():
        raise FileNotFoundError(f"找不到项目Python环境：{PYTHON}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_CACHE_DIR": str(ROOT / ".cache" / "pip"),
            "TEMP": str(ROOT / ".cache" / "tmp"),
            "TMP": str(ROOT / ".cache" / "tmp"),
            "FOOTBALL_MODEL_HOME": str(ROOT),
        }
    )
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    log_file = LOG_PATH.open("ab")
    return subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=8502",
        ],
        cwd=ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )


def start_live_updater() -> None:
    api_key = os.environ.get("API_FOOTBALL_KEY") or os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        return
    if UPDATER_PID_PATH.exists():
        try:
            existing_pid = int(UPDATER_PID_PATH.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
            return
        except (OSError, ValueError):
            pass
    environment = os.environ.copy()
    environment["FOOTBALL_MODEL_HOME"] = str(ROOT)
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    log_file = UPDATER_LOG_PATH.open("ab")
    process = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / "sync_live_context.py"), "--loop"],
        cwd=ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    UPDATER_PID_PATH.write_text(str(process.pid), encoding="utf-8")


def wait_until_healthy(timeout_seconds: int = 30) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_healthy():
            return True
        time.sleep(0.25)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="启动绿茵智析本地可视化程序")
    parser.add_argument("--check-only", action="store_true", help="只检查服务，不打开浏览器")
    args = parser.parse_args()

    if args.check_only:
        print("HEALTHY" if is_healthy() else "NOT_RUNNING")
        return 0 if is_healthy() else 1

    if not is_healthy():
        print("正在启动本地分析服务……")
        start_server()
        if not wait_until_healthy():
            print(f"服务启动失败，请查看日志：{LOG_PATH}")
            return 1

    start_live_updater()

    print(f"服务已就绪，正在打开：{URL}")
    if not webbrowser.open(URL, new=2):
        print(f"浏览器未自动打开，请手动访问：{URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
