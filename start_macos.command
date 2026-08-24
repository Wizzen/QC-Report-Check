#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

pause_on_error() {
  printf '\nStartup failed. Review the messages above.\n'
  read -r -p 'Press Return to close this window...'
}
trap pause_on_error ERR

if ! command -v python3 >/dev/null 2>&1; then
  printf 'Python 3 was not found. Install Python 3.11 or newer from python.org first.\n'
  exit 1
fi

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  printf 'Python 3.11 or newer is required.\n'
  exit 1
}

MAC_VENV="$PROJECT_DIR/.venv-macos"
if [ ! -x "$MAC_VENV/bin/python" ]; then
  printf 'Creating the macOS virtual environment...\n'
  python3 -m venv "$MAC_VENV"
fi

REQ_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"
HASH_FILE="$MAC_VENV/.requirements-sha256"
INSTALLED_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"
if [ "$REQ_HASH" != "$INSTALLED_HASH" ]; then
  printf 'Installing project dependencies...\n'
  "$MAC_VENV/bin/python" -m pip install --upgrade pip
  "$MAC_VENV/bin/python" -m pip install -r requirements.txt
  printf '%s' "$REQ_HASH" > "$HASH_FILE"
fi

export PYTHONUTF8=1
(sleep 4 && open 'http://127.0.0.1:8501') &
printf 'Starting Supplier Quality AI at http://127.0.0.1:8501\n'
printf 'Keep this Terminal window open. Press Control-C to stop the app.\n\n'
exec "$MAC_VENV/bin/python" launcher.py
