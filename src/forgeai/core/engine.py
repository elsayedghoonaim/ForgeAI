"""Managed lifecycle wrapper for the inference runtime."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console

from forgeai.core.backends.base import BaseBackend, EngineStatus, GenerationResult
from forgeai.core.config import DevToolSettings

console = Console()


class DevToolEngine:
    """
    Wrapper for the inference backend (vLLM or llama.cpp) with lifecycle management.
    """

    def __init__(
        self,
        settings: DevToolSettings,
        *,
        streaming: bool = False,
        quiet_startup: bool = False,
    ) -> None:
        self.settings = settings
        self._streaming_enabled = streaming
        self._quiet_startup = quiet_startup
        self._backend: BaseBackend | None = None
        self._requests_served = 0
        self._start_time: float | None = None
        self._last_result: GenerationResult | None = None

    @property
    def is_running(self) -> bool:
        return self._backend is not None and self._backend.is_running

    @property
    def supports_streaming(self) -> bool:
        """Whether this engine build supports incremental token streaming."""
        return self._backend is not None and self._backend.supports_streaming

    @property
    def last_result(self) -> GenerationResult | None:
        """Most recent generation result, including streamed requests."""
        return self._last_result

    def initialize(self) -> None:
        """Initialize the configured backend."""
        from forgeai.core.backends.factory import create_backend

        self._backend = create_backend(
            self.settings,
            streaming=self._streaming_enabled,
            quiet_startup=self._quiet_startup
        )
        self._backend.initialize()
        self._start_time = time.time()

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a model prompt."""
        if not self._backend:
            raise RuntimeError("Engine is not initialized.")
        return self._backend.build_prompt(messages)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        if not self.is_running or not self._backend:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        start = time.time()
        result = self._backend.generate(prompt, max_tokens, temperature, top_p, stop)
        result.elapsed_seconds = time.time() - start
        self._requests_served += 1
        self._last_result = result
        return result

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream output deltas from the async runtime."""
        if not self.is_running or not self._backend:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        if not self.supports_streaming:
            raise NotImplementedError(
                "Streaming is not supported by the active backend."
            )

        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"
        chunks: list[str] = []
        self._last_result = None
        start = time.time()

        async for chunk in self._backend.generate_stream(
            prompt, max_tokens, temperature, top_p, stop
        ):
            chunks.append(chunk)
            # We don't have accurate token counts per chunk from all backends
            # in a unified way here, so we approximate or leave them at 0
            # Backend might update these inside GenerationResult later
            completion_tokens += 1
            yield chunk

        elapsed = time.time() - start
        self._requests_served += 1
        self._last_result = GenerationResult(
            text="".join(chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            elapsed_seconds=elapsed,
        )

    def get_status(self) -> EngineStatus:
        """Get current engine status."""
        return EngineStatus(
            model_name=self.settings.model_name,
            backend=self.settings.backend.value if self.settings.backend else "auto",
            is_running=self.is_running,
            requests_served=self._requests_served,
            start_time=self._start_time,
            pid=os.getpid(),
        )

    def shutdown(self) -> None:
        """Gracefully shut down the engine."""
        if self._backend:
            self._backend.shutdown()
            self._backend = None
        console.print("[yellow]Engine shut down.[/yellow]")

    def __enter__(self) -> DevToolEngine:
        self.initialize()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
