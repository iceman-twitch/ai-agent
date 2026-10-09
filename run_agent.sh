#!/usr/bin/env bash
set -eu

APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    echo "Virtual environment not found. Run ./setup_venv.sh first." >&2
    exit 1
fi

"$APP_DIR/.venv/bin/python" "$APP_DIR/ensure_api_key.py"

exec "$APP_DIR/.venv/bin/python" "$APP_DIR/codeagent.py" "$@"
