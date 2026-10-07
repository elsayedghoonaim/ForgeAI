"""Interactive chat loop shared by ``forgeai run`` (no prompt) and ``forgeai chat``.

Both are thin clients over the daemon's ``/api/chat``: the conversation history lives here
and is resent on every turn. Ctrl-C while a response is streaming cancels just that
response (closing the HTTP stream makes the daemon abort the generation) and returns to the
prompt; Ctrl-C or EOF at the prompt ends the session.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from rich.console import Console

from forgeai.cli.runtime import DaemonClient

console = Console()

EXIT_COMMANDS = frozenset({"/bye", "/exit", "/quit"})


def chat_turn(
    client: DaemonClient,
    body: dict[str, Any],
    *,
    stream: bool,
    wait_indicator: Callable[[], AbstractContextManager[Any]] | None = None,
) -> tuple[str, bool]:
    """Run one chat request. Returns (assistant_text, cancelled_by_ctrl_c)."""
    if not stream:
        try:
            res = client.request("POST", "/api/chat", json_data=body, long=True)
        except KeyboardInterrupt:
            print("\n[cancelled]", flush=True)
            return "", True
        text = str((res.get("message") or {}).get("content", ""))
        print(text, flush=True)
        return text, False

    gen = client.stream("POST", "/api/chat", json_data=body)
    parts: list[str] = []
    try:
        indicator = wait_indicator() if wait_indicator is not None else nullcontext()
        with indicator:
            chunk = next(gen, None)
        while chunk is not None:
            text = str((chunk.get("message") or {}).get("content", "")) if isinstance(chunk, dict) else ""
            if text:
                print(text, end="", flush=True)
                parts.append(text)
            chunk = next(gen, None)
        print(flush=True)
    except KeyboardInterrupt:
        close = getattr(gen, "close", None)
        if close is not None:
            close()  # drops the connection; the daemon aborts the in-flight generation
        print("\n[cancelled]", flush=True)
        return "".join(parts), True
    return "".join(parts), False


def run_repl(
    client: DaemonClient,
    model: str,
    *,
    system_prompt: str | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: str | None = None,
    stream: bool = True,
    prompt_label: str = ">>> ",
    wait_indicator: Callable[[], AbstractContextManager[Any]] | None = None,
) -> None:
    """Read-eval-print loop keeping the whole conversation in ``history``."""
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        command = user_input.lower()
        if command in EXIT_COMMANDS:
            break
        if command == "/clear":
            history = [m for m in history if m["role"] == "system"]
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue

        history.append({"role": "user", "content": user_input})
        body: dict[str, Any] = {"model": model, "messages": list(history), "stream": stream}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        if options:
            body["options"] = options

        try:
            text, cancelled = chat_turn(client, body, stream=stream, wait_indicator=wait_indicator)
        except BaseException:
            history.pop()
            raise
        if cancelled:
            history.pop()  # drop the unanswered turn so roles keep alternating
            continue
        history.append({"role": "assistant", "content": text or "(empty response)"})
