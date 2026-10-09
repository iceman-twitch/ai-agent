"""Prompt for an Anthropic API key and save it to the project .env file."""

from __future__ import annotations

import re
import sys
from getpass import getpass
from pathlib import Path

from dotenv import dotenv_values


ENV_FILE = Path(__file__).with_name(".env")
KEY_ASSIGNMENT = re.compile(r"^\s*ANTHROPIC_API_KEY\s*=.*$", re.MULTILINE)


def is_placeholder(value: str | None) -> bool:
    if not value or not value.strip():
        return True
    normalized = value.strip().lower()
    placeholders = ("your-key", "your_api_key", "replace-me", "placeholder")
    return any(marker in normalized for marker in placeholders)


def main() -> int:
    if not is_placeholder(dotenv_values(ENV_FILE).get("ANTHROPIC_API_KEY")):
        return 0

    print("ANTHROPIC_API_KEY is missing from .env.")
    while True:
        try:
            api_key = getpass("Enter your Anthropic API key (input hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAPI key setup cancelled.", file=sys.stderr)
            return 1

        if not api_key or is_placeholder(api_key):
            print("Please enter a non-empty API key, not a placeholder.")
            continue
        break

    content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    assignment = f"ANTHROPIC_API_KEY={api_key}"
    if KEY_ASSIGNMENT.search(content):
        content = KEY_ASSIGNMENT.sub(assignment, content)
    else:
        if content and not content.endswith(("\n", "\r")):
            content += "\n"
        content += assignment + "\n"

    ENV_FILE.write_text(content, encoding="utf-8")
    print("Saved your API key to .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
