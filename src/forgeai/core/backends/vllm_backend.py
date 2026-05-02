"""vLLM implementation of the BaseBackend."""

from __future__ import annotations

import asyncio
import importlib
import math
import os
import tempfile
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.console import Console

from forgeai.core.config import BackendType, DevToolSettings
from forgeai.core.security import check_vllm_version
from forgeai.utils.gpu import GPU_MEMORY_STARTUP_RESERVE_MB
from forgeai.core.backends.base import BaseBackend, GenerationResult

console = Console()

QUIET_STARTUP_WARNING_FILTERS: tuple[tuple[str, type[Warning]], ...] = (
    (r".*unauthenticated requests to the HF Hub.*", UserWarning),
    (r".*cuda\.cudart module is deprecated.*", FutureWarning),
    (r".*cuda\.nvrtc module is deprecated.*", FutureWarning),
)
QUIET_STARTUP_PYTHONWARNING_PREFIXES: tuple[tuple[str, str], ...] = (
    ("You are sending unauthenticated requests to the HF Hub", "UserWarning"),
    ("The cuda.cudart module is deprecated", "FutureWarning"),
    ("The cuda.nvrtc module is deprecated", "FutureWarning"),
)


class VLLMBackend(BaseBackend):
    """vLLM implementation of the inference backend."""

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
        self._tokenizer: Any = None

    @property
    def supports_streaming(self) -> bool:
        return self._streaming_enabled

    def initialize(self) -> None:
        """Initialize the vLLM engine."""
        if not (self.settings.model_name or self.settings.model_path):
            raise ValueError("A model_name or model_path is required to start the engine.")

        with self._startup_context():
            if self.settings.enforce_version_check:
                check_vllm_version(announce_success=not self._quiet_startup)

            self._preflight_vllm_memory()
            if self._streaming_enabled:
                self._init_vllm_async()
            else:
                self._init_vllm()
        self._is_running = True
        if not self._quiet_startup:
            console.print(
                f"[green]OK[/green] Engine initialized: "
                f"[bold]{self.settings.model_name or self.settings.model_path}[/bold] "
                f"(backend={BackendType.VLLM.value})"
            )

    def _init_vllm(self) -> None:
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM
        except ImportError as err:
            raise RuntimeError(
                "vLLM is not installed. Install it with: pip install 'forgeai[vllm]'"
            ) from err

        kwargs = self.settings.to_vllm_kwargs()
        if not self._quiet_startup:
            console.print(f"[dim]Initializing vLLM engine with: {kwargs}[/dim]")
        self._engine = LLM(**kwargs)
        self._tokenizer = self._engine.get_tokenizer()

    def _init_vllm_async(self) -> None:
        """Initialize the async vLLM engine used for streaming requests."""
        try:
            from vllm import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError as err:
            raise RuntimeError(
                "Streaming requires a vLLM build with AsyncLLM support. "
                "Install it with: pip install 'forgeai[vllm]'"
            ) from err

        engine_args_kwargs = self._build_async_engine_args_kwargs()
        if not self._quiet_startup:
            console.print(f"[dim]Initializing vLLM async engine with: {engine_args_kwargs}[/dim]")
        self._engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args_kwargs))
        self._tokenizer = self._engine.get_tokenizer()

    def _build_async_engine_args_kwargs(self) -> dict[str, object]:
        """Translate settings into AsyncEngineArgs-compatible kwargs."""
        kwargs = self.settings.to_vllm_kwargs()
        engine_args_kwargs: dict[str, object] = {
            "model": kwargs["model"],
            "tensor_parallel_size": kwargs["tensor_parallel_size"],
            "gpu_memory_utilization": kwargs["gpu_memory_utilization"],
            "max_num_seqs": kwargs["max_num_seqs"],
            "trust_remote_code": kwargs["trust_remote_code"],
            "disable_log_stats": True,
        }
        if "enforce_eager" in kwargs:
            engine_args_kwargs["enforce_eager"] = kwargs["enforce_eager"]
        if "max_num_batched_tokens" in kwargs:
            engine_args_kwargs["max_num_batched_tokens"] = kwargs["max_num_batched_tokens"]
        if "max_model_len" in kwargs:
            engine_args_kwargs["max_model_len"] = kwargs["max_model_len"]
        if "quantization" in kwargs:
            engine_args_kwargs["quantization"] = kwargs["quantization"]
        return engine_args_kwargs

    @contextmanager
    def _startup_context(self) -> Iterator[None]:
        """Apply chat-friendly startup suppression without hiding real errors."""
        if not self._quiet_startup:
            yield
            return

        original_vllm_logging_level = os.environ.get("VLLM_LOGGING_LEVEL")
        original_hf_hub_verbosity = os.environ.get("HF_HUB_VERBOSITY")
        original_pythonwarnings = os.environ.get("PYTHONWARNINGS")
        original_pythonpath = os.environ.get("PYTHONPATH")
        startup_site = tempfile.TemporaryDirectory(prefix="forgeai-startup-site-")
        hf_logging = None
        hf_original_level = None
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["HF_HUB_VERBOSITY"] = "error"
        os.environ["PYTHONWARNINGS"] = _merge_pythonwarnings(original_pythonwarnings)
        os.environ["PYTHONPATH"] = _merge_pythonpath(
            original_pythonpath,
            startup_site.name,
        )
        _write_startup_sitecustomize(Path(startup_site.name) / "sitecustomize.py")

        with warnings.catch_warnings():
            for message, category in QUIET_STARTUP_WARNING_FILTERS:
                warnings.filterwarnings("ignore", message=message, category=category)
            try:
                try:
                    hf_logging = importlib.import_module("huggingface_hub.utils.logging")
                    hf_original_level = hf_logging.get_verbosity()
                    hf_logging.set_verbosity_error()
                except Exception:
                    hf_logging = None
                yield
            finally:
                if hf_logging is not None and hf_original_level is not None:
                    hf_logging.set_verbosity(hf_original_level)

                if original_vllm_logging_level is None:
                    os.environ.pop("VLLM_LOGGING_LEVEL", None)
                else:
                    os.environ["VLLM_LOGGING_LEVEL"] = original_vllm_logging_level

                if original_hf_hub_verbosity is None:
                    os.environ.pop("HF_HUB_VERBOSITY", None)
                else:
                    os.environ["HF_HUB_VERBOSITY"] = original_hf_hub_verbosity

                if original_pythonwarnings is None:
                    os.environ.pop("PYTHONWARNINGS", None)
                else:
                    os.environ["PYTHONWARNINGS"] = original_pythonwarnings

                if original_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = original_pythonpath
                startup_site.cleanup()

    def _preflight_vllm_memory(self) -> None:
        """Fail early when the requested GPU utilization cannot fit in free VRAM."""
        try:
            from forgeai.utils.gpu import detect_gpus
        except Exception:
            return

        topology = detect_gpus()
        if topology.gpu_count == 0:
            return

        requested_gpus = self.settings.tensor_parallel_size
        if topology.gpu_count < requested_gpus:
            raise RuntimeError(
                f"Tensor parallel size {requested_gpus} requires {requested_gpus} GPUs, "
                f"but only {topology.gpu_count} GPU(s) were detected."
            )

        requested_util = self.settings.gpu_memory_utilization
        startup_reserve_mb = GPU_MEMORY_STARTUP_RESERVE_MB

        for gpu in topology.gpus[:requested_gpus]:
            desired_free_mb = gpu.total_memory_mb * requested_util
            usable_free_mb = max(0.0, gpu.free_memory_mb - startup_reserve_mb)
            if usable_free_mb >= desired_free_mb:
                continue

            safe_fraction = max(
                0.10,
                min(
                    0.95,
                    math.floor(max(0.0, usable_free_mb / gpu.total_memory_mb) * 100) / 100,
                ),
            )
            free_gb = gpu.free_memory_mb / 1024
            total_gb = gpu.total_memory_mb / 1024
            desired_gb = desired_free_mb / 1024

            raise RuntimeError(
                "Insufficient free GPU memory to start vLLM.\n"
                f"GPU {gpu.index} ({gpu.name}) has {free_gb:.2f} GiB free out of {total_gb:.2f} GiB.\n"
                f"Requested --gpu-util {requested_util:.2f} needs about {desired_gb:.2f} GiB free at startup.\n"
                f"Try lowering --gpu-util to about {safe_fraction:.2f}, closing other GPU processes, "
                "or using a smaller model."
            )

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a model prompt."""
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered_prompt, str):
                    return rendered_prompt
            except Exception:
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
        if not self._is_running:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        if self._streaming_enabled:
            return asyncio.run(
                self._generate_vllm_async(prompt, max_tokens or 512, temperature, top_p, stop)
            )
        else:
            return self._generate_vllm(prompt, max_tokens or 512, temperature, top_p, stop)

    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> GenerationResult:
        """Generate using the vLLM backend."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs = self._engine.generate([prompt], params)
        return self._request_output_to_result(outputs[0])

    async def _generate_vllm_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> GenerationResult:
        """Generate a full completion using the async vLLM engine."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        final_output: Any = None
        async for output in self._engine.generate(
            prompt,
            params,
            request_id=f"gen-{uuid4().hex}",
        ):
            final_output = output

        if final_output is None:
            return GenerationResult(text="")

        return self._request_output_to_result(final_output)

    @staticmethod
    def _request_output_to_result(output: Any) -> GenerationResult:
        """Convert a vLLM RequestOutput-like object into GenerationResult."""
        prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
        outputs = getattr(output, "outputs", []) or []
        if not outputs:
            prompt_tokens = len(prompt_token_ids)
            return GenerationResult(
                text="",
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            )

        completion = outputs[0]
        completion_tokens = len(getattr(completion, "token_ids", []) or [])
        prompt_tokens = len(prompt_token_ids)
        return GenerationResult(
            text=getattr(completion, "text", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=getattr(completion, "finish_reason", None) or "stop",
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream output deltas from the async vLLM runtime."""
        if not self._is_running:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        if not self._streaming_enabled:
            raise NotImplementedError(
                "Streaming is not enabled in this build. Use non-streaming requests."
            )

        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        params = SamplingParams(
            max_tokens=max_tokens or 512,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in self._engine.generate(
            prompt,
            params,
            request_id=f"stream-{uuid4().hex}",
        ):
            outputs = getattr(output, "outputs", []) or []
            if not outputs:
                continue

            completion = outputs[0]
            chunk = getattr(completion, "text", "")
            if chunk:
                yield chunk

    def shutdown(self) -> None:
        """Gracefully shut down the engine."""
        if self._engine is not None:
            self._engine = None
            self._tokenizer = None
        self._is_running = False
        console.print("[yellow]vLLM Engine shut down.[/yellow]")


def _merge_pythonwarnings(existing: str | None) -> str:
    """Merge startup warning filters into the PYTHONWARNINGS env var."""
    filters = [
        f"ignore:{message}:{category_name}"
        for message, category_name in QUIET_STARTUP_PYTHONWARNING_PREFIXES
    ]
    if existing:
        return ",".join([existing, *filters])
    return ",".join(filters)


def _merge_pythonpath(existing: str | None, startup_site_dir: str) -> str:
    """Prepend a temporary startup helper directory to PYTHONPATH."""
    if existing:
        return os.pathsep.join([startup_site_dir, existing])
    return startup_site_dir


def _write_startup_sitecustomize(path: Path) -> None:
    """Write a temporary sitecustomize module for spawned startup workers."""
    lines = [
        "import warnings",
        "",
    ]
    for message, category in QUIET_STARTUP_WARNING_FILTERS:
        lines.append(
            f"warnings.filterwarnings('ignore', message={message!r}, category={category.__name__})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
