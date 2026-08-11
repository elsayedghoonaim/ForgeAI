"""forgeai run - one-shot or interactive inference via daemon."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error
from forgeai.core.telemetry import track_event

console = Console()


def run(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    prompt: str | None = typer.Argument(None, help="Input prompt"),
    prompt_option: str | None = typer.Option(None, "--prompt", "-p", help="Input prompt (compatibility alias)"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens as they are generated"),
    keep_alive: str | None = typer.Option(None, "--keep-alive", help="Engine keep_alive TTL"),
    temperature: float | None = typer.Option(None, "--temperature", "-t", help="Sampling temperature"),
    top_p: float | None = typer.Option(None, "--top-p", help="Top-p sampling"),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k sampling"),
    num_predict: int | None = typer.Option(None, "--num-predict", help="Maximum tokens to generate"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum tokens to generate (compatibility alias)"),
    stop: list[str] | None = typer.Option(None, "--stop", help="Stop sequences"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    """Execute inference through the running ForgeAI daemon."""

    if prompt is not None and prompt_option is not None:
        if prompt != prompt_option:
            handle_cli_error("Conflicting prompt positional argument and --prompt/-p option provided.")
        effective_prompt = prompt
    elif prompt is not None:
        effective_prompt = prompt
    elif prompt_option is not None:
        effective_prompt = prompt_option
    else:
        effective_prompt = None

    if num_predict is not None and max_tokens is not None:
        if num_predict != max_tokens:
            handle_cli_error("Conflicting --num-predict and --max-tokens options provided.")
        effective_num_predict = num_predict
    elif num_predict is not None:
        effective_num_predict = num_predict
    elif max_tokens is not None:
        effective_num_predict = max_tokens
    else:
        effective_num_predict = None

    track_event("command.run", {"model": model, "stream": stream, "has_prompt": effective_prompt is not None})

    options_payload: dict[str, Any] = {}
    if temperature is not None:
        options_payload["temperature"] = temperature
    if top_p is not None:
        options_payload["top_p"] = top_p
    if top_k is not None:
        options_payload["top_k"] = top_k
    if effective_num_predict is not None:
        options_payload["num_predict"] = effective_num_predict
    if stop:
        options_payload["stop"] = stop

    try:
        client = DaemonClient(host=host, port=port)
        if effective_prompt is not None:
            body: dict[str, Any] = {
                "model": model,
                "prompt": effective_prompt,
                "stream": stream,
            }
            if keep_alive is not None:
                body["keep_alive"] = keep_alive
            if options_payload:
                body["options"] = options_payload

            if stream:
                for chunk in client.stream("POST", "/api/generate", json_data=body):
                    if isinstance(chunk, dict):
                        response_text = chunk.get("response", "")
                        if response_text:
                            print(response_text, end="", flush=True)
                print(flush=True)
            else:
                res = client.request("POST", "/api/generate", json_data=body)
                response_text = res.get("response", "")
                print(response_text, flush=True)
        else:
            while True:
                try:
                    user_input = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue

                if user_input == "/bye":
                    break

                body = {
                    "model": model,
                    "prompt": user_input,
                    "stream": stream,
                }
                if keep_alive is not None:
                    body["keep_alive"] = keep_alive
                if options_payload:
                    body["options"] = options_payload

                if stream:
                    for chunk in client.stream("POST", "/api/generate", json_data=body):
                        if isinstance(chunk, dict):
                            response_text = chunk.get("response", "")
                            if response_text:
                                print(response_text, end="", flush=True)
                    print(flush=True)
                else:
                    res = client.request("POST", "/api/generate", json_data=body)
                    response_text = res.get("response", "")
                    print(response_text, flush=True)
    except DaemonClientError as err:
        handle_cli_error(err)
