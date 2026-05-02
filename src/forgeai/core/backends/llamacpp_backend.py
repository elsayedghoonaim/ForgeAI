"""llama.cpp implementation of the BaseBackend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console

from forgeai.core.backends.base import BaseBackend, GenerationResult
from forgeai.core.config import BackendType, DevToolSettings

console = Console()


class LlamaCppBackend(BaseBackend):
    """llama.cpp implementation of the inference backend."""

    def __init__(
        self,
        settings: DevToolSettings,
        *,
        streaming: bool = False,
        quiet_startup: bool = False,
    ) -> None:
        super().__init__(settings)
        self._streaming_enabled = streaming
        self._quiet_startup = quiet_startup
        self._engine: Any = None

    @property
    def supports_streaming(self) -> bool:
        return True  # llama-cpp-python always supports streaming

    def initialize(self) -> None:
        """Initialize the llama.cpp engine."""
        if not (self.settings.model_name or self.settings.model_path):
            raise ValueError("A model_name or model_path is required to start the engine.")

        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install 'forgeai[llamacpp]'"
            ) from err

        kwargs = self.settings.to_llamacpp_kwargs()
        if not self._quiet_startup:
            console.print(f"[dim]Initializing llama.cpp engine with: {kwargs}[/dim]")

        self._engine = Llama(**kwargs)
        self._is_running = True

        if not self._quiet_startup:
            console.print(
                f"[green]OK[/green] Engine initialized: "
                f"[bold]{self.settings.model_name or self.settings.model_path}[/bold] "
                f"(backend={BackendType.LLAMA_CPP.value})"
            )

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a model prompt."""
        # We rely on llama.cpp's internal chat templating where possible via create_chat_completion,
        # but if a raw prompt is requested, we can use the engine's built-in formatting.
        if hasattr(self._engine, "metadata") and "tokenizer.chat_template" in self._engine.metadata:
            # llama-cpp-python handles chat templates natively in create_chat_completion
            pass

        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        if not self._is_running or not self._engine:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        # If it's a raw string prompt, we use __call__
        output = self._engine(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        return self._request_output_to_result(output)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream output deltas from the runtime."""
        if not self._is_running or not self._engine:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        # llama-cpp-python streaming is a sync generator, so we wrap it for the async interface
        generator = self._engine(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
        )

        for chunk in generator:
            text = chunk["choices"][0]["text"]
            if text:
                yield text
                await asyncio.sleep(0)  # Yield to event loop

    @staticmethod
    def _request_output_to_result(output: Any) -> GenerationResult:
        """Convert a llama-cpp-python output dict into GenerationResult."""
        choices = output.get("choices", [])
        if not choices:
            return GenerationResult(text="")

        choice = choices[0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "stop")

        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    def shutdown(self) -> None:
        """Gracefully shut down the engine."""
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._is_running = False
        console.print("[yellow]llama.cpp Engine shut down.[/yellow]")
