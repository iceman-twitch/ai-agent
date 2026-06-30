#!/usr/bin/env python3
"""
CodeAgent — an AI coding assistant agent powered by the Anthropic Claude API.

A command-line tool that behaves as a highly skilled, friendly software engineer.
It helps you write, debug, refactor, explain, and test code in any major
programming language. Conversation history is kept in memory for the duration of
the session, so the agent remembers earlier messages.

Usage:
    python codeagent.py                      # interactive REPL
    python codeagent.py -q "Reverse a list"  # single-query mode
    echo "code..." | python codeagent.py -q "Find the bug"   # piped stdin

    python codeagent.py --no-thinking --effort low   # cheaper / faster
    python codeagent.py --max-history 20             # cap tokens per request

Environment:
    ANTHROPIC_API_KEY   Your Anthropic API key (or place it in a .env file).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable

# --- Third-party dependencies (see requirements.txt) --------------------------
try:
    import anthropic
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "Missing a required dependency: {0}\n"
        "Install everything with:\n\n"
        "    pip install -r requirements.txt\n\n".format(exc.name)
    )
    raise SystemExit(1)


# --- Configuration ------------------------------------------------------------

# Default to the latest, most capable Claude model. Override with --model.
DEFAULT_MODEL = "claude-opus-4-8"

# Streaming responses can be long; give the model generous room. Streaming
# avoids the SDK's HTTP-timeout guard that triggers on large non-streaming calls.
MAX_TOKENS = 16000

# Valid effort levels (output_config.effort). Lower effort => fewer tokens.
EFFORT_LEVELS = ("low", "medium", "high", "max")
DEFAULT_EFFORT = "high"

# The agent's persona. The system prompt is what makes Claude behave as a
# friendly, expert software engineer. Replace the text below with your own exact
# wording if you have a specific prompt you want to use.
SYSTEM_PROMPT = """\
You are CodeAgent, a highly skilled, friendly, and patient senior software \
engineer. Your sole purpose is to help the user write, debug, improve, explain, \
and test program code in any major programming language (Python, JavaScript, \
TypeScript, Java, C/C++, C#, Go, Rust, Ruby, PHP, SQL, HTML/CSS, shell, and more).

How you work:
1. When given an error or faulty code, first clearly identify the problem, then \
explain the root cause in simple terms, and finally provide the corrected code.
2. If a request is ambiguous, ask targeted clarifying questions (language, \
framework, constraints, expected input/output) before generating code.
3. Always specify the language in markdown code fences (e.g. ```python).
4. Write clean, readable, well-named, and appropriately commented code. Consider \
edge cases, error handling, and input validation.
5. Return complete, runnable code rather than fragments with "..." — unless the \
user explicitly asks for a snippet.
6. Keep explanations concise but thorough: what the problem was, why it happened, \
and how the fix works. When relevant, mention time/space complexity and trade-offs.
7. Follow language-specific conventions (PEP 8 for Python, idiomatic style for \
each language). When suggesting libraries, briefly mention alternatives and trade-offs.
8. Never produce malicious, unethical, or destructive code (malware, backdoors, \
credential theft, etc.). Decline such requests politely.
9. Maintain a supportive, encouraging tone — the user may be a beginner.

Stay on topic as a coding assistant. Use GitHub-flavored Markdown in your replies."""

# Appended to the system prompt when thinking is disabled, to stop Opus from
# leaking exploratory reasoning into the visible answer (which wastes tokens).
NO_THINKING_SUFFIX = (
    "\n\nRespond with your final answer only — do not include step-by-step "
    "exploratory reasoning, rejected drafts, or meta-commentary about your process."
)

# Commands that end the session.
EXIT_COMMANDS = {"exit", "quit", ":q", ":quit"}

console = Console()


# --- Token usage tracking -----------------------------------------------------

@dataclass
class Usage:
    """Running totals of token usage across the session."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, u) -> "Usage":
        """Accumulate one response's usage object."""
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.requests += 1
        return self


def format_usage_line(u) -> str:
    """Compact one-line summary of a single response's token usage."""
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cread = getattr(u, "cache_read_input_tokens", 0) or 0
    cwrite = getattr(u, "cache_creation_input_tokens", 0) or 0
    cache_note = ""
    if cread or cwrite:
        cache_note = f", cache read {cread:,} / write {cwrite:,}"
    return f"tokens — in {inp:,}{cache_note} · out {out:,}"


# --- Core agent ---------------------------------------------------------------

@dataclass
class CodeAgent:
    """Wraps the Anthropic client and an in-memory conversation history."""

    client: anthropic.Anthropic
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    thinking_enabled: bool = True
    use_cache: bool = True          # prompt caching: bills resent history at ~0.1x
    max_history_messages: int = 0   # 0 = unlimited; otherwise a sliding window
    show_usage: bool = True

    history: list[dict] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    last_usage: object | None = None

    def reset(self) -> None:
        """Forget the conversation so far, starting a fresh context."""
        self.history.clear()

    def system_prompt(self) -> str:
        if self.thinking_enabled:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + NO_THINKING_SUFFIX

    def ask(self, user_message: str) -> str:
        """
        Send `user_message` to Claude (with history) and stream the reply.

        Returns the assistant's complete text response. Raises anthropic.* errors
        on API failures; callers are expected to handle them.
        """
        self.history.append({"role": "user", "content": user_message})

        assistant_text = self._stream_response()

        # Persist the assistant turn so future messages have context.
        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def _windowed_history(self) -> list[dict]:
        """
        Return the messages to send. When a window is set, keep only the most
        recent messages to cap tokens per request, while keeping the sequence
        starting on a 'user' turn (required by the API).
        """
        msgs = self.history
        if self.max_history_messages and len(msgs) > self.max_history_messages:
            msgs = msgs[-self.max_history_messages:]
            while msgs and msgs[0]["role"] != "user":
                msgs = msgs[1:]
        return msgs

    def _request_kwargs(self) -> dict:
        """Assemble keyword arguments shared by the streaming request."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": self.system_prompt(),
            "messages": self._windowed_history(),
        }
        # Adaptive thinking lets Claude decide how much to reason per request.
        # Disabling it (and lowering effort) is the simplest way to cut tokens.
        kwargs["thinking"] = (
            {"type": "adaptive"} if self.thinking_enabled else {"type": "disabled"}
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.use_cache:
            # Prompt caching: auto-caches the longest stable prefix so resent
            # history is billed at ~0.1x on subsequent turns.
            kwargs["cache_control"] = {"type": "ephemeral"}
        return kwargs

    def _stream_response(self) -> str:
        """Stream tokens from the API, live-rendering Markdown as they arrive."""
        rendered = ""

        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            with self.client.messages.stream(**self._request_kwargs()) as stream:
                for text in stream.text_stream:
                    rendered += text
                    live.update(Markdown(rendered))

                # get_final_message() gives us the complete, validated response.
                final = stream.get_final_message()

        # Record token usage for reporting.
        self.last_usage = final.usage
        self.total_usage.add(final.usage)
        if self.show_usage:
            console.print(f"[dim]{format_usage_line(final.usage)}[/dim]")

        text_blocks = [b.text for b in final.content if b.type == "text"]
        return "".join(text_blocks) if text_blocks else rendered


# --- Error handling helper ----------------------------------------------------

def describe_api_error(exc: Exception) -> str:
    """Turn an Anthropic SDK exception into a clear, actionable message."""
    if isinstance(exc, anthropic.AuthenticationError):
        return ("Authentication failed. Check that ANTHROPIC_API_KEY is set "
                "correctly (in your environment or a .env file).")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "Your API key lacks permission for this model or endpoint."
    if isinstance(exc, anthropic.NotFoundError):
        return "Model or endpoint not found — check the model name (/model or --model)."
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = "a little while"
        try:
            retry_after = f"{exc.response.headers.get('retry-after', '60')}s"
        except Exception:
            pass
        return f"Rate limited by the API. Please retry after {retry_after}."
    if isinstance(exc, anthropic.BadRequestError):
        # Common cause: 'effort' on a model that doesn't support it.
        return f"The request was rejected: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Network error — could not reach the Anthropic API. Check your connection."
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return f"Anthropic server error ({exc.status_code}). Try again shortly."
        return f"API error ({exc.status_code}): {exc.message}"
    return f"Unexpected error: {exc}"


def build_client() -> anthropic.Anthropic:
    """Load the API key (env or .env) and construct the Anthropic client."""
    # load_dotenv() reads a local .env file if present; it never overrides
    # variables already set in the real environment.
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                Text(
                    "ANTHROPIC_API_KEY is not set.\n\n"
                    "Set it in your shell:\n"
                    "    export ANTHROPIC_API_KEY=sk-ant-...\n\n"
                    "or create a .env file containing:\n"
                    "    ANTHROPIC_API_KEY=sk-ant-...",
                    style="yellow",
                ),
                title="Missing API key",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # The SDK reads ANTHROPIC_API_KEY from the environment automatically.
    return anthropic.Anthropic()


# --- Interactive mode ---------------------------------------------------------

HELP_TEXT = """\
[bold]CodeAgent commands[/bold]
  [cyan]exit[/cyan], [cyan]quit[/cyan]        End the session
  [cyan]/reset[/cyan]             Clear the conversation history (fresh context)
  [cyan]/model[/cyan] <name>      Switch model (e.g. claude-sonnet-4-6)
  [cyan]/effort[/cyan] <level>    Set reasoning effort: low | medium | high | max
  [cyan]/thinking[/cyan] on|off   Toggle adaptive thinking
  [cyan]/cache[/cyan] on|off      Toggle prompt caching (cheaper resent history)
  [cyan]/window[/cyan] <n>        Keep only the last n messages (0 = unlimited)
  [cyan]/tokens[/cyan]            Show token usage this session
  [cyan]/help[/cyan]              Show this help

Type anything else to chat. Paste code and ask the agent to debug, refactor,
explain, or test it. The agent remembers earlier messages in this session.
"""


def status_line(agent: CodeAgent) -> str:
    thinking = "on" if agent.thinking_enabled else "off"
    cache = "on" if agent.use_cache else "off"
    window = agent.max_history_messages or "unlimited"
    return (f"model: {agent.model} · effort: {agent.effort} · thinking: {thinking}"
            f" · cache: {cache} · window: {window}")


def print_banner(agent: CodeAgent) -> None:
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]CodeAgent[/bold cyan] — your AI pair programmer\n"
                f"[dim]{status_line(agent)}[/dim]\n\n"
                "Type your coding question, or [cyan]/help[/cyan] for commands. "
                "[cyan]exit[/cyan] to quit."
            ),
            border_style="cyan",
        )
    )


def show_tokens(agent: CodeAgent) -> None:
    t = agent.total_usage
    if t.requests == 0:
        console.print("[dim]No requests yet this session.[/dim]")
        return
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Session token usage[/bold] ({t.requests} request(s))\n"
                f"  input:        {t.input_tokens:,}\n"
                f"  output:       {t.output_tokens:,}\n"
                f"  cache read:   {t.cache_read_tokens:,}  [dim](billed ~0.1x)[/dim]\n"
                f"  cache write:  {t.cache_write_tokens:,}  [dim](billed ~1.25x)[/dim]"
            ),
            border_style="dim",
        )
    )


def handle_command(agent: CodeAgent, text: str) -> bool:
    """
    Handle a slash/keyword command. Returns True if the input was a command
    (and was handled), False if it should be treated as a normal chat message.
    """
    lowered = text.lower()

    if lowered in EXIT_COMMANDS:
        raise SystemExit(0)
    if lowered in {"/reset", "reset"}:
        agent.reset()
        console.print("[dim]Conversation cleared.[/dim]")
        return True
    if lowered in {"/help", "help", "?"}:
        console.print(Panel(Text.from_markup(HELP_TEXT), border_style="dim"))
        return True
    if lowered in {"/tokens", "tokens"}:
        show_tokens(agent)
        return True
    if lowered in {"/status", "status"}:
        console.print(f"[dim]{status_line(agent)}[/dim]")
        return True

    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/model":
        if not arg:
            console.print(f"[dim]Current model: {agent.model}[/dim]")
        else:
            agent.model = arg
            console.print(f"[dim]Model set to {arg}.[/dim]")
        return True
    if cmd == "/effort":
        if arg.lower() in EFFORT_LEVELS:
            agent.effort = arg.lower()
            console.print(f"[dim]Effort set to {agent.effort}.[/dim]")
        else:
            console.print(f"[yellow]Effort must be one of: {', '.join(EFFORT_LEVELS)}[/yellow]")
        return True
    if cmd == "/thinking":
        if arg.lower() in {"on", "off"}:
            agent.thinking_enabled = arg.lower() == "on"
            console.print(f"[dim]Thinking {'on' if agent.thinking_enabled else 'off'}.[/dim]")
        else:
            console.print("[yellow]Usage: /thinking on|off[/yellow]")
        return True
    if cmd == "/cache":
        if arg.lower() in {"on", "off"}:
            agent.use_cache = arg.lower() == "on"
            console.print(f"[dim]Prompt caching {'on' if agent.use_cache else 'off'}.[/dim]")
        else:
            console.print("[yellow]Usage: /cache on|off[/yellow]")
        return True
    if cmd == "/window":
        if arg.isdigit():
            agent.max_history_messages = int(arg)
            window = agent.max_history_messages or "unlimited"
            console.print(f"[dim]History window: {window} messages.[/dim]")
        else:
            console.print("[yellow]Usage: /window <n>  (0 = unlimited)[/yellow]")
        return True

    return False


def interactive_loop(agent: CodeAgent) -> None:
    """Run the read-eval-print loop until the user exits."""
    print_banner(agent)

    while True:
        try:
            user_input = console.input("\n[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            return

        if not user_input:
            continue

        try:
            if handle_command(agent, user_input):
                continue
        except SystemExit:
            console.print("[dim]Goodbye![/dim]")
            return

        console.print("\n[bold magenta]CodeAgent ›[/bold magenta]")
        try:
            agent.ask(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as exc:  # surface a friendly message, keep the REPL alive
            console.print(f"[red]{describe_api_error(exc)}[/red]")


# --- Single-query mode --------------------------------------------------------

def read_stdin_if_piped() -> str:
    """Return piped stdin text, or an empty string if stdin is a TTY."""
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def single_query(agent: CodeAgent, query: str) -> int:
    """Answer one query and exit. Returns a process exit code."""
    piped = read_stdin_if_piped()
    if piped.strip():
        query = f"{query}\n\n```\n{piped.rstrip()}\n```"

    try:
        agent.ask(query)
    except Exception as exc:
        console.print(f"[red]{describe_api_error(exc)}[/red]")
        return 1
    return 0


# --- Entry point --------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="codeagent",
        description="CodeAgent — an AI coding assistant powered by Claude.",
    )
    parser.add_argument(
        "-q", "--query",
        help="Run a single query and exit (otherwise start interactive mode). "
             "Piped stdin is appended as a code block.",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=DEFAULT_EFFORT,
        help=f"Reasoning effort / token spend (default: {DEFAULT_EFFORT}). "
             "Lower = fewer tokens.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable adaptive thinking (faster, fewer tokens).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching (caching makes resent history ~10x cheaper).",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=0,
        metavar="N",
        help="Keep only the last N messages per request to cap token usage "
             "(0 = unlimited, the default).",
    )
    parser.add_argument(
        "--no-usage",
        action="store_true",
        help="Don't print the per-reply token-usage line.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    client = build_client()
    agent = CodeAgent(
        client=client,
        model=args.model,
        effort=args.effort,
        thinking_enabled=not args.no_thinking,
        use_cache=not args.no_cache,
        max_history_messages=args.max_history,
        show_usage=not args.no_usage,
    )

    if args.query:
        return single_query(agent, args.query)

    interactive_loop(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
