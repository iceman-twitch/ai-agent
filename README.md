# CodeAgent

An AI coding assistant agent for the command line, powered by the Anthropic
Claude API (`claude-opus-4-8`). It behaves as a friendly, expert software
engineer that helps you **write, debug, refactor, explain, and test** code in any
major language — with syntax-highlighted, Markdown-formatted answers and
in-memory conversation history for the session.

## Features

- **Interactive REPL** or **single-query** mode (`-q`).
- **Streaming** responses, live-rendered as Markdown with syntax highlighting (`rich`).
- **Conversation memory** within a session (resend full history each turn — the API is stateless).
- **Adaptive thinking** — Claude decides how much to reason per request (toggle with `--no-thinking`).
- **Effort control** — `--effort {low,medium,high,max}` trades quality for token cost.
- **Switch models mid-session** with `/model`.
- **Lower token usage** — prompt caching (on by default), an optional history window,
  and live usage reporting so you can see the savings (see below).
- **Graceful error handling** for auth, rate-limit, network, and server errors.
- API key from the `ANTHROPIC_API_KEY` env var **or** a `.env` file (`python-dotenv`).
- Pipe code in via stdin: it's appended to your query as a code block.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your key
# or: export ANTHROPIC_API_KEY=sk-ant-...
```

Get an API key at <https://console.anthropic.com/>.

## Usage

```bash
# Interactive chat
python codeagent.py

# One-shot question
python codeagent.py -q "Write a Python function to check if a string is a palindrome"

# Debug a file by piping it in
cat buggy.py | python codeagent.py -q "Find and fix the bug in this code"

# Use a different model
python codeagent.py -m claude-sonnet-4-6

# Cheaper / faster: no thinking, low effort
python codeagent.py --no-thinking --effort low
```

### Options

| Flag                 | Effect                                                        |
| -------------------- | ------------------------------------------------------------ |
| `-q, --query`        | Run one query and exit (interactive otherwise)               |
| `-m, --model`        | Model to use (default `claude-opus-4-8`)                     |
| `--effort`           | `low` / `medium` / `high` / `max` — lower = fewer tokens     |
| `--no-thinking`      | Disable adaptive thinking (faster, cheaper)                  |
| `--no-cache`         | Disable prompt caching                                       |
| `--max-history N`    | Keep only the last N messages per request (0 = unlimited)    |
| `--no-usage`         | Hide the per-reply token-usage line                          |

### Interactive commands

| Command            | Action                                          |
| ------------------ | ----------------------------------------------- |
| `exit` / `quit`    | End the session                                 |
| `/reset`           | Clear conversation history (fresh start)        |
| `/model <name>`    | Switch model mid-session                        |
| `/effort <level>`  | Set effort: `low` / `medium` / `high` / `max`   |
| `/thinking on\|off` | Toggle adaptive thinking                        |
| `/cache on\|off`    | Toggle prompt caching                           |
| `/window <n>`      | Keep only the last n messages (0 = unlimited)   |
| `/tokens`          | Show session token usage                        |
| `/help`            | Show help                                       |

## Reducing token usage

Three levers, all built in:

1. **Prompt caching** (on by default). Each turn the stable prefix of the
   conversation is cached, so resent history is billed at roughly **0.1×** on
   subsequent turns instead of full price. Disable with `--no-cache` or `/cache off`.
2. **History window** (`--max-history N` / `/window N`). Sends only the last N
   messages, capping tokens per request in long sessions. Default is unlimited.
3. **Less reasoning.** `--no-thinking` and `--effort low` both reduce the tokens
   the model generates.

After every reply, a dim line shows that turn's token usage (input, cache
read/write, output). Run `/tokens` for the session totals — `cache read` tokens
are the cheap ones, so a high cache-read count means caching is working.

## Requirements

- Python 3.10+
- `anthropic`, `python-dotenv`, `rich` (see `requirements.txt`)
- Pydantic is constrained below 2.14 because newer versions use a `ForwardRef`
  argument unavailable in Python 3.10.0.

## Notes

The agent's persona lives in the `SYSTEM_PROMPT` constant in
[codeagent.py](codeagent.py) — edit it to customize behavior.
