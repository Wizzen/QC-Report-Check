#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

pause_on_error() {
  printf '\nStartup failed. Review the messages above.\n'
  read -r -p 'Press Return to close this window...'
}
trap pause_on_error ERR

export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

find_python311() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  for candidate in \
    "$HOME/.local/bin/python3.13" \
    "$HOME/.local/bin/python3.12" \
    "$HOME/.local/bin/python3.11" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3"; do
    if [ -x "$candidate" ] \
      && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(find_python311 || true)"
if [ -z "$PYTHON" ]; then
  printf 'Python 3.11 or newer was not found. Install it from python.org or ensure python3.12 is on PATH.\n'
  exit 1
fi
printf 'Using Python: %s (%s)\n' "$PYTHON" "$("$PYTHON" --version 2>&1)"

MAC_VENV="$PROJECT_DIR/.venv-macos"
if [ ! -x "$MAC_VENV/bin/python" ]; then
  printf 'Creating the macOS virtual environment...\n'
  "$PYTHON" -m venv "$MAC_VENV"
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
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
(sleep 4 && open 'http://127.0.0.1:8501') &
printf 'Starting Supplier Quality AI at http://127.0.0.1:8501\n'
printf 'Keep this Terminal window open. Press Control-C to stop the app.\n\n'
exec "$MAC_VENV/bin/python" launcher.py
