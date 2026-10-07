"""Abstract base class for all inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from forgeai.core.config import DevToolSettings


@dataclass
class GenerationResult:
    """Result of a single generation request."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    elapsed_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds > 0:
            return self.completion_tokens / self.elapsed_seconds
        return 0.0


@dataclass
class StreamStats:
    """Per-request streaming statistics, filled in by the backend as the stream runs.

    One instance is created per streaming call, so concurrent requests never share
    state. ``request_id`` lets callers abort the in-flight engine request.
    """

    request_id: str = field(default_factory=lambda: f"stream-{uuid4().hex}")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds > 0:
            return self.completion_tokens / self.elapsed_seconds
        return 0.0


async def count_chunks(source: AsyncIterator[str], stats: StreamStats) -> AsyncIterator[str]:
    """Yield chunks from ``source`` counting them as completion tokens (fallback stats)."""
    try:
        async for chunk in source:
            stats.completion_tokens += 1
            yield chunk
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()


def iter_stream(
    engine: Any, prompt: str, stats: StreamStats, **params: Any
) -> AsyncIterator[str]:
    """Stream from any engine-like object, recording per-request stats in ``stats``.

    Uses ``stream_with_stats`` when the engine provides it; otherwise falls back to
    ``generate_stream`` with chunk counts as the token estimate.
    """
    fn = getattr(engine, "stream_with_stats", None)
    if fn is not None:
        return fn(prompt, stats=stats, **params)  # type: ignore[no-any-return]
    return count_chunks(engine.generate_stream(prompt=prompt, **params), stats)


async def abort_request(engine: Any, request_id: str) -> None:
    """Best-effort abort of an in-flight engine request; never raises."""
    fn = getattr(engine, "abort", None)
    if fn is None:
        return
    try:
        res = fn(request_id)
        if hasattr(res, "__await__"):
            await res
    except Exception:
        pass


@dataclass
class EngineStatus:
    """Current status of the engine."""

    model_name: str = ""
    backend: str = ""
    is_running: bool = False
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    requests_served: int = 0
    start_time: float | None = None
    pid: int = 0


class BaseBackend(ABC):
    """
    The universal contract that every inference engine must implement.
    Allows forgeai to be completely engine-agnostic at the CLI/API layer.
    """

    def __init__(self, settings: DevToolSettings) -> None:
        self.settings = settings
        self._is_running = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the engine with the provided settings."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Run a single asynchronous generation."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        """Run an asynchronous generation that yields string deltas."""
        pass


    async def stream_with_stats(
        self,
        prompt: str,
        *,
        stats: StreamStats,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream deltas while recording per-request stats. Default counts chunks."""
        async for chunk in count_chunks(
            self.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                top_k=top_k,
            ),
            stats,
        ):
            yield chunk

    async def abort(self, request_id: str) -> None:
        """Abort an in-flight request. No-op for backends that cannot abort."""
        return None

    @property
    def supports_concurrency(self) -> bool:
        """True if multiple generate calls may safely overlap on this backend."""
        return False

    @abstractmethod
    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format a list of chat messages into a single prompt string."""
        pass

    async def embed(self, input_texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for input texts."""
        raise NotImplementedError("Embedding is not supported by this backend.")

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully terminate the engine and free resources."""
        pass

    @property
    def is_running(self) -> bool:
        """Return True if the engine is currently loaded in memory."""
        return self._is_running

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return True if this backend natively supports streaming."""
        pass

