#!/usr/bin/env bash
set -eu

cd -- "$(dirname -- "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

echo "Virtual environment is ready."
