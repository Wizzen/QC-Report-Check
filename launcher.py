from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".app-pids.json"


def main() -> int:
    streamlit_command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app.py"),
               "--browser.gatherUsageStats=false", "--server.address=127.0.0.1"]
    worker_command = [sys.executable, str(ROOT / "worker.py")]
    worker = subprocess.Popen(worker_command, cwd=ROOT)
    web = subprocess.Popen(streamlit_command, cwd=ROOT)
    PID_FILE.write_text(json.dumps({"streamlit": web.pid, "worker": worker.pid}), encoding="utf-8")
    try:
        return web.wait()
    finally:
        for process in (web, worker):
            if process.poll() is None:
                process.terminate()
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
