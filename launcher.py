from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import load_config
from app.database import ReviewDatabase


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".app-pids.json"
WORKER_STATUS_FILE = ROOT / ".worker-status.json"
LOG_DIR = ROOT / "logs"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_worker_status(status: str, *, pid: int = 0, restart_count: int = 0,
                         last_exit_code: int | None = None, message: str = "") -> None:
    _write_json(WORKER_STATUS_FILE, {
        "status": status, "pid": pid, "restart_count": restart_count,
        "last_exit_code": last_exit_code, "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def _spawn(command: list[str], log_name: str) -> tuple[subprocess.Popen[bytes], object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = (LOG_DIR / log_name).open("ab")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags,
    )
    return process, log_handle


def main() -> int:
    # Initialize and migrate SQLite once before the web process and worker start.
    # This avoids a clean installation racing two schema initializers.
    ReviewDatabase(load_config().storage.database_v2)
    server_host = os.environ.get("QAQC_HOST", "127.0.0.1").strip() or "127.0.0.1"
    streamlit_command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app.py"),
               "--browser.gatherUsageStats=false", f"--server.address={server_host}"]
    worker_command = [sys.executable, str(ROOT / "worker.py")]
    worker, worker_log = _spawn(worker_command, "worker.log")
    web, web_log = _spawn(streamlit_command, "web.log")
    restart_count = 0
    restart_window = time.monotonic()
    worker_failed = False
    last_exit_code: int | None = None
    worker_message = ""
    _write_json(PID_FILE, {"streamlit": web.pid, "worker": worker.pid})
    _write_worker_status("online", pid=worker.pid)
    try:
        while web.poll() is None:
            exit_code = worker.poll() if worker is not None else None
            if exit_code is not None and not worker_failed:
                last_exit_code = int(exit_code)
                worker_log.close()
                if time.monotonic() - restart_window > 60:
                    restart_count = 0
                    restart_window = time.monotonic()
                restart_count += 1
                if restart_count > 5:
                    worker_failed = True
                    worker_message = "worker 连续退出，请查看 logs/worker.log"
                    _write_worker_status(
                        "failed", restart_count=restart_count, last_exit_code=last_exit_code,
                        message=worker_message,
                    )
                else:
                    _write_worker_status(
                        "restarting", restart_count=restart_count, last_exit_code=last_exit_code,
                        message="worker 已退出，启动器正在自动恢复",
                    )
                    time.sleep(min(restart_count, 3))
                    worker, worker_log = _spawn(worker_command, "worker.log")
                    worker_message = "worker 已自动恢复"
                    _write_json(PID_FILE, {"streamlit": web.pid, "worker": worker.pid})
                    _write_worker_status("online", pid=worker.pid, restart_count=restart_count,
                                         last_exit_code=last_exit_code, message=worker_message)
            elif worker is not None and worker.poll() is None:
                _write_worker_status("online", pid=worker.pid, restart_count=restart_count,
                                     last_exit_code=last_exit_code, message=worker_message)
            time.sleep(1)
        return int(web.returncode or 0)
    finally:
        for process in (web, worker):
            if process is not None and process.poll() is None:
                process.terminate()
        for handle in (web_log, worker_log):
            try:
                handle.close()
            except Exception:
                pass
        _write_worker_status("stopped", restart_count=restart_count, message="应用已停止")
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
