"""Abstract base class for all inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from dataclasses import dataclass

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
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Run a single synchronous generation."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Run an asynchronous generation that yields string deltas."""
        pass

    @abstractmethod
    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format a list of chat messages into a single prompt string."""
        pass

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
