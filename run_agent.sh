#!/usr/bin/env bash
set -eu

cd -- "$(dirname -- "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual environment not found. Run ./setup_venv.sh first." >&2
    exit 1
fi

.venv/bin/python ensure_api_key.py

exec .venv/bin/python codeagent.py "$@"
