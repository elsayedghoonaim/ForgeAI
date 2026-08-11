"""Shared CLI runtime tuning helpers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Generator, NoReturn

import httpx
import typer
from rich.console import Console

from forgeai.utils.gpu import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    GPUTopology,
    detect_gpus,
    get_target_gpus,
    recommend_gpu_memory_utilization,
)

console = Console()


class DaemonClientError(Exception):
    """Stable CLI-facing exception for daemon client interactions."""

    pass


def _parse_and_validate_port(port_val: Any, source_name: str) -> int:
    """Validate and convert port value to an integer between 1 and 65535."""
    try:
        p = int(port_val)
    except (ValueError, TypeError) as err:
        raise DaemonClientError(
            f"Invalid port number {port_val!r} from {source_name}. Must be an integer between 1 and 65535."
        ) from err
    if not (1 <= p <= 65535):
        raise DaemonClientError(
            f"Port number {p} from {source_name} out of valid range (1-65535)."
        )
    return p


def resolve_base_url(
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
) -> str:
    """
    Resolve base URL for local daemon.
    Defaults to http://127.0.0.1:11434, overridable by CLI parameters,
    FORGEAI_HOST / FORGEAI_PORT or forgeai_host / forgeai_port.
    """
    if base_url:
        url = base_url.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        return url

    # 1. Check explicit CLI port option
    validated_cli_port: int | None = None
    if port is not None:
        validated_cli_port = _parse_and_validate_port(port, "CLI option")

    # 2. Determine raw host
    raw_host = host
    if raw_host is None:
        raw_host = os.getenv("FORGEAI_HOST") or os.getenv("forgeai_host")

    # 3. Parse scheme and embedded port from host string if present
    embedded_port: int | None = None
    if raw_host:
        h = raw_host.strip().rstrip("/")
        scheme = "http"
        if h.startswith("http://"):
            scheme = "http"
            h = h[7:]
        elif h.startswith("https://"):
            scheme = "https"
            h = h[8:]

        if ":" in h and not h.endswith("]"):
            hostname, host_port_str = h.rsplit(":", 1)
            if host_port_str.isdigit():
                embedded_port = _parse_and_validate_port(host_port_str, "host parameter")
                h = hostname
    else:
        scheme = "http"
        h = "127.0.0.1"

    # 4. Determine final port priority: CLI port > host-embedded port > environment port > default 11434
    if validated_cli_port is not None:
        final_port = validated_cli_port
    elif embedded_port is not None:
        final_port = embedded_port
    else:
        env_port = os.getenv("FORGEAI_PORT") or os.getenv("forgeai_port")
        if env_port is not None:
            final_port = _parse_and_validate_port(env_port, "environment variable")
        else:
            final_port = 11434

    return f"{scheme}://{h}:{final_port}"


class DaemonClient:
    """
    Client for interacting with the local Ollama-compatible daemon.
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = resolve_base_url(host=host, port=port, base_url=base_url)
        self.timeout = httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=5.0)

    def request(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        params: Any = None,
    ) -> dict[str, Any]:
        """Make an ordinary JSON HTTP request and return parsed JSON."""
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, json=json_data, params=params)
                if response.status_code >= 400:
                    try:
                        err_json = response.json()
                        if isinstance(err_json, dict) and "error" in err_json:
                            raise DaemonClientError(str(err_json["error"]))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    raise DaemonClientError(f"HTTP {response.status_code}: {response.text.strip()}")

                try:
                    return response.json()
                except Exception as err:
                    raise DaemonClientError(f"Malformed JSON response from daemon: {err}") from err
        except DaemonClientError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.RequestError) as err:
            raise DaemonClientError(
                f"Could not connect to daemon at {self.base_url}. Is the server running? Start it with: forgeai serve"
            ) from err
        except Exception as err:
            raise DaemonClientError(f"HTTP request failed: {err}") from err

    def stream(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        params: Any = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream NDJSON response line-by-line without buffering the whole response."""
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(method, url, json=json_data, params=params) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        try:
                            err_json = json.loads(body)
                            if isinstance(err_json, dict) and "error" in err_json:
                                raise DaemonClientError(str(err_json["error"]))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                        raise DaemonClientError(f"HTTP {response.status_code}: {body.strip()}")

                    for line in response.iter_lines():
                        if not line or not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except Exception as err:
                            raise DaemonClientError(f"Malformed NDJSON chunk: {err}") from err

                        if isinstance(item, dict) and "error" in item:
                            raise DaemonClientError(str(item["error"]))

                        yield item
        except DaemonClientError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.RequestError) as err:
            raise DaemonClientError(
                f"Could not connect to daemon at {self.base_url}. Is the server running? Start it with: forgeai serve"
            ) from err
        except Exception as err:
            raise DaemonClientError(f"HTTP stream failed: {err}") from err


def handle_cli_error(err: Any) -> NoReturn:
    """Print concise error message to stderr and exit with code 1."""
    err_console = Console(stderr=True)
    err_console.print(f"Error: {err}")
    raise typer.Exit(code=1)


CHAT_MAX_NUM_SEQS_ENV = "forgeai_MAX_NUM_SEQS"
CHAT_MAX_MODEL_LEN_ENV = "forgeai_MAX_MODEL_LEN"
CHAT_ENFORCE_EAGER_ENV = "forgeai_ENFORCE_EAGER"
RUN_MAX_NUM_SEQS_ENV = "forgeai_RUN_MAX_NUM_SEQS"
RUN_MAX_MODEL_LEN_ENV = "forgeai_RUN_MAX_MODEL_LEN"
RUN_ENFORCE_EAGER_ENV = "forgeai_RUN_ENFORCE_EAGER"
RUN_MAX_NUM_BATCHED_TOKENS_ENV = "forgeai_RUN_MAX_NUM_BATCHED_TOKENS"
CHAT_CONTEXT_TOKENS_PER_MB = 4.0
RUN_CONTEXT_TOKENS_PER_MB = 2.0
SEQ_MEMORY_DIVISOR_MB = 256.0


@dataclass
class RuntimeTuning:
    """Resolved runtime settings for CLI commands."""

    profile: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False
    topology: GPUTopology | None = None
    auto_tensor_parallel: bool = False
    auto_gpu_utilization: bool = False
    auto_max_num_seqs: bool = False
    auto_max_model_len: bool = False
    auto_max_num_batched_tokens: bool = False
    auto_enforce_eager: bool = False


def resolve_runtime_tuning(
    *,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    auto_optimize: bool = False,
    chat_mode: bool = False,
    run_mode: bool = False,
    model_name: str | None = None,
) -> RuntimeTuning:
    """Resolve effective TP and GPU utilization for a CLI invocation."""

    tp_size = tensor_parallel_size or 1
    effective_gpu_util = gpu_memory_utilization
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager = False
    topology: GPUTopology | None = None
    auto_tp = False
    auto_gpu_util = False
    auto_max_num_seqs = False
    auto_max_model_len = False
    auto_max_num_batched_tokens = False
    auto_enforce_eager = False

    try:
        topology = detect_gpus()
        if auto_optimize and tensor_parallel_size is None and topology.gpu_count > 0:
            tp_size = topology.recommended_tp_size or 1
            auto_tp = True

        if effective_gpu_util is None:
            if chat_mode:
                effective_gpu_util = recommend_chat_gpu_memory_utilization(
                    topology,
                    tensor_parallel_size=tp_size,
                )
            elif run_mode:
                effective_gpu_util = recommend_run_gpu_memory_utilization(
                    topology,
                    tensor_parallel_size=tp_size,
                )
            else:
                effective_gpu_util = recommend_gpu_memory_utilization(
                    topology,
                    tensor_parallel_size=tp_size,
                    requested_max=DEFAULT_GPU_MEMORY_UTILIZATION,
                )
            auto_gpu_util = True

        if chat_mode and topology.gpu_count > 0:
            if os.getenv(CHAT_MAX_NUM_SEQS_ENV) is None:
                max_num_seqs = recommend_chat_max_num_seqs(
                    topology,
                    tensor_parallel_size=tp_size,
                    model_name=model_name,
                )
                auto_max_num_seqs = True

            if os.getenv(CHAT_MAX_MODEL_LEN_ENV) is None:
                max_model_len = recommend_chat_max_model_len(
                    topology,
                    tensor_parallel_size=tp_size,
                    max_num_seqs=max_num_seqs,
                    model_name=model_name,
                )
                auto_max_model_len = True

            if os.getenv(CHAT_ENFORCE_EAGER_ENV) is None:
                enforce_eager = recommend_chat_enforce_eager(
                    topology,
                    tensor_parallel_size=tp_size,
                )
                auto_enforce_eager = enforce_eager
            else:
                enforce_eager = _parse_bool_env(os.getenv(CHAT_ENFORCE_EAGER_ENV))
        elif run_mode and topology.gpu_count > 0:
            if os.getenv(RUN_MAX_NUM_SEQS_ENV) is None:
                max_num_seqs = recommend_run_max_num_seqs(
                    topology,
                    tensor_parallel_size=tp_size,
                )
                auto_max_num_seqs = True
            else:
                max_num_seqs = _parse_int_env(os.getenv(RUN_MAX_NUM_SEQS_ENV))

            if os.getenv(RUN_MAX_MODEL_LEN_ENV) is None:
                max_model_len = recommend_run_max_model_len(
                    topology,
                    tensor_parallel_size=tp_size,
                    model_name=model_name,
                )
                auto_max_model_len = True
            else:
                max_model_len = _parse_int_env(os.getenv(RUN_MAX_MODEL_LEN_ENV))

            if os.getenv(RUN_MAX_NUM_BATCHED_TOKENS_ENV) is None:
                max_num_batched_tokens = recommend_run_max_num_batched_tokens(
                    topology,
                    tensor_parallel_size=tp_size,
                    max_model_len=max_model_len,
                )
                auto_max_num_batched_tokens = True
            else:
                max_num_batched_tokens = _parse_int_env(os.getenv(RUN_MAX_NUM_BATCHED_TOKENS_ENV))

            if os.getenv(RUN_ENFORCE_EAGER_ENV) is None:
                enforce_eager = recommend_chat_enforce_eager(
                    topology,
                    tensor_parallel_size=tp_size,
                )
                auto_enforce_eager = enforce_eager
            else:
                enforce_eager = _parse_bool_env(os.getenv(RUN_ENFORCE_EAGER_ENV))
    except Exception:
        pass

    if effective_gpu_util is None:
        effective_gpu_util = DEFAULT_GPU_MEMORY_UTILIZATION

    return RuntimeTuning(
        profile="chat" if chat_mode else "run" if run_mode else "default",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=effective_gpu_util,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
        topology=topology,
        auto_tensor_parallel=auto_tp,
        auto_gpu_utilization=auto_gpu_util,
        auto_max_num_seqs=auto_max_num_seqs,
        auto_max_model_len=auto_max_model_len,
        auto_max_num_batched_tokens=auto_max_num_batched_tokens,
        auto_enforce_eager=auto_enforce_eager,
    )


def _estimate_model_weight_mb(model_name: str | None, tensor_parallel_size: int) -> float:
    """Estimate model weight VRAM footprint in MB."""
    model_weight_mb = 0.0
    if model_name:
        try:
            from forgeai.utils.memory_estimator import estimate_from_preset
            preset_est = estimate_from_preset(model_name, available_vram_mb=24000, tensor_parallel_size=tensor_parallel_size)
            if preset_est:
                model_weight_mb = preset_est.model_params_mb + preset_est.activation_mb + preset_est.overhead_mb
            else:
                import re
                match = re.search(r"(\d+)b", model_name.lower())
                if match:
                    b_params = float(match.group(1))
                    params_mb = (b_params * 1e9 * 2.0) / (1024 * 1024) / tensor_parallel_size
                    model_weight_mb = params_mb + (params_mb * 0.1) + 500.0
        except Exception:
            pass
    return model_weight_mb


def recommend_chat_max_num_seqs(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
    model_name: str | None = None,
) -> int:
    """Recommend a low-latency chat concurrency target from free VRAM."""

    target_gpus = get_target_gpus(topology, tensor_parallel_size=tensor_parallel_size)
    if not target_gpus:
        return 1

    free_mb = min(gpu.free_memory_mb for gpu in target_gpus)
    model_weight_mb = _estimate_model_weight_mb(model_name, tensor_parallel_size)
    adjusted_free_mb = max(512.0, free_mb - model_weight_mb)
    raw_target = max(1.0, math.sqrt(max(1.0, adjusted_free_mb / SEQ_MEMORY_DIVISOR_MB)))
    return _round_down_power_of_two(raw_target, minimum=1, maximum=32)


def recommend_chat_gpu_memory_utilization(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
) -> float:
    """Recommend a more conservative GPU util target for interactive chat."""

    return recommend_gpu_memory_utilization(
        topology,
        tensor_parallel_size=tensor_parallel_size,
        requested_max=DEFAULT_GPU_MEMORY_UTILIZATION,
    )


def recommend_run_gpu_memory_utilization(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
) -> float:
    """Recommend a safe GPU util target for one-shot runs on constrained GPUs."""

    return recommend_gpu_memory_utilization(
        topology,
        tensor_parallel_size=tensor_parallel_size,
        requested_max=DEFAULT_GPU_MEMORY_UTILIZATION,
    )


def recommend_chat_max_model_len(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
    max_num_seqs: int | None = None,
    model_name: str | None = None,
) -> int:
    """Recommend a chat context limit that avoids huge startup overhead."""

    target_gpus = get_target_gpus(topology, tensor_parallel_size=tensor_parallel_size)
    if not target_gpus:
        return 4096

    concurrency = max_num_seqs or recommend_chat_max_num_seqs(
        topology,
        tensor_parallel_size=tensor_parallel_size,
        model_name=model_name,
    )
    free_mb = min(gpu.free_memory_mb for gpu in target_gpus)
    model_weight_mb = _estimate_model_weight_mb(model_name, tensor_parallel_size)
    adjusted_free_mb = max(512.0, free_mb - model_weight_mb)
    per_sequence_mb = adjusted_free_mb / max(1, concurrency)
    estimated_tokens = per_sequence_mb * CHAT_CONTEXT_TOKENS_PER_MB
    return _round_down_power_of_two(
        estimated_tokens,
        minimum=1024,
        maximum=131072,
    )


def recommend_run_max_num_seqs(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
) -> int:
    """Recommend a one-shot concurrency target for `run`."""

    return 1


def recommend_run_max_model_len(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
    model_name: str | None = None,
) -> int:
    """Recommend a one-shot context limit to reduce startup overhead."""

    target_gpus = get_target_gpus(topology, tensor_parallel_size=tensor_parallel_size)
    if not target_gpus:
        return 4096

    free_mb = min(gpu.free_memory_mb for gpu in target_gpus)
    model_weight_mb = _estimate_model_weight_mb(model_name, tensor_parallel_size)
    adjusted_free_mb = max(512.0, free_mb - model_weight_mb)
    estimated_tokens = adjusted_free_mb * RUN_CONTEXT_TOKENS_PER_MB
    return _round_down_power_of_two(
        estimated_tokens,
        minimum=1024,
        maximum=131072,
    )


def recommend_run_max_num_batched_tokens(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
) -> int:
    """Recommend a smaller batch-token cap for one-shot latency."""

    context_limit = max_model_len or recommend_run_max_model_len(
        topology,
        tensor_parallel_size=tensor_parallel_size,
    )
    return _round_down_power_of_two(
        context_limit / 8,
        minimum=512,
        maximum=min(8192, context_limit),
    )


def recommend_chat_enforce_eager(
    topology: GPUTopology,
    *,
    tensor_parallel_size: int = 1,
) -> bool:
    """Prefer eager mode for small interactive-chat GPUs to cut startup latency."""

    target_gpus = get_target_gpus(topology, tensor_parallel_size=tensor_parallel_size)
    if not target_gpus:
        return False

    total_mb = min(gpu.total_memory_mb for gpu in target_gpus)
    free_mb = min(gpu.free_memory_mb for gpu in target_gpus)
    free_ratio = free_mb / total_mb if total_mb > 0 else 0.0
    return free_ratio < 0.9


def _parse_bool_env(value: str | None) -> bool:
    """Parse a relaxed boolean environment variable."""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(value: str | None) -> int | None:
    """Parse an integer environment variable when provided."""

    if value is None:
        return None
    return int(value.strip())


def _round_down_power_of_two(
    value: float,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Round a positive float down to a bounded power of two."""

    bounded = max(float(minimum), min(float(maximum), value))
    exponent = max(0, int(math.floor(math.log2(bounded))))
    rounded = 1 << exponent
    return max(minimum, min(maximum, rounded))


def print_runtime_tuning(tuning: RuntimeTuning) -> None:
    """Emit concise tuning details for CLI commands."""

    if tuning.auto_tensor_parallel:
        console.print(f"[dim]Auto-TP: {tuning.tensor_parallel_size} GPU(s)[/dim]")

    if tuning.auto_gpu_utilization:
        detail = ""
        if tuning.topology is not None and tuning.topology.gpu_count > 0:
            target_gpus = get_target_gpus(
                tuning.topology,
                tensor_parallel_size=tuning.tensor_parallel_size,
            )
            if target_gpus:
                limiting_gpu = min(target_gpus, key=lambda gpu: gpu.free_memory_mb)
                detail = (
                    f" based on current free VRAM "
                    f"({limiting_gpu.free_memory_mb / 1024:.2f} GiB free on GPU {limiting_gpu.index})"
                )

        console.print(
            f"[dim]Auto GPU util: {tuning.gpu_memory_utilization:.2f}{detail}[/dim]"
        )

    if tuning.auto_max_num_seqs:
        seqs_detail = ""
        if tuning.profile == "chat" and tuning.topology is not None and tuning.topology.gpu_count > 0:
            target_gpus = get_target_gpus(
                tuning.topology,
                tensor_parallel_size=tuning.tensor_parallel_size,
            )
            if target_gpus:
                limiting_gpu = min(target_gpus, key=lambda gpu: gpu.free_memory_mb)
                seqs_detail = (
                    f" for interactive chat "
                    f"({limiting_gpu.free_memory_mb / 1024:.2f} GiB free on GPU {limiting_gpu.index})"
                )
        label = "Auto chat max_num_seqs" if tuning.profile == "chat" else "Auto run max_num_seqs"
        console.print(f"[dim]{label}: {tuning.max_num_seqs}{seqs_detail}[/dim]")

    if tuning.auto_max_model_len:
        context_detail = ""
        if tuning.topology is not None and tuning.topology.gpu_count > 0:
            target_gpus = get_target_gpus(
                tuning.topology,
                tensor_parallel_size=tuning.tensor_parallel_size,
            )
            if target_gpus:
                limiting_gpu = min(target_gpus, key=lambda gpu: gpu.total_memory_mb)
                context_detail = (
                    f" to reduce startup overhead "
                    f"on {limiting_gpu.total_memory_mb / 1024:.2f} GiB GPUs"
                )
        label = "Auto chat max_model_len" if tuning.profile == "chat" else "Auto run max_model_len"
        console.print(f"[dim]{label}: {tuning.max_model_len}{context_detail}[/dim]")

    if tuning.auto_max_num_batched_tokens:
        console.print(
            f"[dim]Auto run max_num_batched_tokens: {tuning.max_num_batched_tokens} "
            "to reduce startup compile overhead[/dim]"
        )

    if tuning.auto_enforce_eager:
        eager_detail = ""
        if tuning.topology is not None and tuning.topology.gpu_count > 0:
            target_gpus = get_target_gpus(
                tuning.topology,
                tensor_parallel_size=tuning.tensor_parallel_size,
            )
            if target_gpus:
                limiting_gpu = min(target_gpus, key=lambda gpu: gpu.total_memory_mb)
                eager_detail = (
                    f" to reduce startup compile overhead "
                    f"on {limiting_gpu.total_memory_mb / 1024:.2f} GiB GPUs"
                )
        label = "Auto chat eager mode" if tuning.profile == "chat" else "Auto run eager mode"
        console.print(f"[dim]{label}: enabled{eager_detail}[/dim]")
