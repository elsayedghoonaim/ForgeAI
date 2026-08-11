"""Engine lifecycle supervisor and managed runtime wrapper."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
import math
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console

from forgeai.core.backends.base import BaseBackend, GenerationResult
from forgeai.core.backends.base import EngineStatus as LegacyEngineStatus
from forgeai.core.config import DevToolSettings, QuantizationType

console = Console()


class EngineManagerError(RuntimeError):
    """Base exception for EngineManager supervisor errors."""

    code: str = "ERR_ENGINE_MANAGER"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class EngineQueueFullError(EngineManagerError):
    """Raised when daemon request queue depth is exceeded."""

    code: str = "ERR_QUEUE_FULL"


class AdmissionVRAMError(EngineManagerError):
    """Raised when model memory estimate exceeds available VRAM capacity."""

    code: str = "ERR_ADMISSION_VRAM"


class EngineOOMError(EngineManagerError):
    """Raised when engine experiences CUDA Out-Of-Memory during initialization or decode."""

    code: str = "ERR_ENGINE_FAILED_OOM"


class TurboQuantHWCCError(EngineManagerError):
    """Raised when TurboQuant is requested on CUDA Compute Capability < 7.5."""

    code: str = "ERR_TURBOQUANT_HW_CC"


class ROCmDeferredError(EngineManagerError):
    """Raised when TurboQuant is requested on AMD ROCm platform."""

    code: str = "ERR_ROCM_DEFERRED"


class NoCompatibleGPUError(EngineManagerError):
    """Raised when no compatible CUDA/ROCm GPU is detected."""

    code: str = "ERR_NO_COMPATIBLE_GPU"


class EngineDrainingError(EngineManagerError):
    """Raised when attempting to acquire a key that is currently draining or shutting down."""

    code: str = "ERR_ENGINE_DRAINING"


class EngineShuttingDownError(EngineManagerError):
    """Raised when supervisor is shutting down and new acquisitions are refused."""

    code: str = "ERR_ENGINE_SHUTTING_DOWN"


class EngineLoadCancelledError(EngineManagerError):
    """Raised when cold load attempt is cancelled by the leader task."""

    code: str = "ERR_ENGINE_LOAD_CANCELLED"


class EngineState(str, Enum):
    """State machine lifecycle states for a vLLM engine instance."""

    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    UNLOADING = "UNLOADING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class EngineKey:
    """Immutable identity key for a vLLM engine instance."""

    repo_id: str
    snapshot_hash: str = ""
    revision: str = "main"
    tokenizer: str = ""
    tokenizer_revision: str = ""
    chat_template_digest: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 4096
    max_num_seqs: int = 4
    dtype: str = "auto"
    weight_quantization: str = "none"
    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = False
    trust_remote_code: bool = False
    device_runtime_id: str = "cuda:0"

    def __post_init__(self) -> None:
        val = self.weight_quantization.lower().strip()
        allowed = {e.value for e in QuantizationType}
        if val not in allowed:
            raise ValueError(
                f"Invalid weight_quantization '{self.weight_quantization}'. "
                f"Must be one of {sorted(allowed)}."
            )

    def to_settings(self) -> DevToolSettings:
        """Convert engine key to DevToolSettings instance."""
        quant_enum = QuantizationType(self.weight_quantization.lower().strip())
        return DevToolSettings(
            model_name=self.repo_id,
            model_path=self.repo_id if os.path.exists(self.repo_id) else None,
            tokenizer=self.tokenizer if self.tokenizer else None,
            revision=self.revision if self.revision else None,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            dtype=self.dtype,
            quantization=quant_enum,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            max_num_seqs=self.max_num_seqs,
            kv_cache_dtype=self.kv_cache_dtype,
        )




@dataclass(frozen=True)
class KeepAlivePolicy:
    """Parsed keep_alive TTL policy specification."""

    raw_value: Any
    ttl_seconds: float
    is_indefinite: bool = False
    is_immediate_unload: bool = False


DURATION_REGEX = re.compile(r"^([-+]?\d*(?:\.\d+)?)\s*([a-zA-Z\u00b5\u03bc]+)$")
DURATION_UNITS: dict[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}


def parse_keep_alive(val: Any) -> KeepAlivePolicy:
    """
    Parse keep_alive parameter with Ollama semantics.

    - None / omitted => 5m (300s)
    - numbers => seconds
    - duration strings => ns/us/ms/s/m/h/d
    - 0 => unload after final release (is_immediate_unload=True)
    - negative => indefinite (is_indefinite=True)
    - Rejects booleans, NaN/inf, malformed strings, positive overflow with ValueError.
    """
    if isinstance(val, bool):
        raise ValueError("Boolean values are not valid keep_alive specifications.")

    if val is None or val == "":
        return KeepAlivePolicy(raw_value=val, ttl_seconds=300.0)

    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            raise ValueError("NaN and Infinity are not valid keep_alive values.")
        if val > 3153600000:
            raise ValueError(f"keep_alive value {val} exceeds maximum allowed threshold.")
        if val < 0:
            return KeepAlivePolicy(raw_value=val, ttl_seconds=-1.0, is_indefinite=True)
        if val == 0:
            return KeepAlivePolicy(raw_value=val, ttl_seconds=0.0, is_immediate_unload=True)
        return KeepAlivePolicy(raw_value=val, ttl_seconds=float(val))

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return KeepAlivePolicy(raw_value=val, ttl_seconds=300.0)

        try:
            num = float(s)
            return parse_keep_alive(num)
        except ValueError:
            pass

        match = DURATION_REGEX.match(s)
        if not match:
            raise ValueError(
                f"Malformed keep_alive duration string: '{val}'. "
                "Expected format like '5m', '10s', '1h', '500ms', '0', or '-1'."
            )

        num_str, unit_str = match.groups()
        if not num_str or num_str in ("+", "-"):
            raise ValueError(f"Malformed keep_alive number in string: '{val}'.")

        try:
            num_val = float(num_str)
        except ValueError as err:
            raise ValueError(f"Invalid keep_alive numeric component in '{val}'.") from err

        if math.isnan(num_val) or math.isinf(num_val):
            raise ValueError("NaN and Infinity are not valid keep_alive values.")

        unit_lower = unit_str.lower()
        multiplier = DURATION_UNITS.get(unit_lower)
        if multiplier is None:
            raise ValueError(
                f"Unknown keep_alive duration unit '{unit_str}' in '{val}'."
            )

        ttl = num_val * multiplier
        if ttl > 3153600000:
            raise ValueError(f"keep_alive duration '{val}' exceeds maximum threshold.")
        if ttl < 0:
            return KeepAlivePolicy(raw_value=val, ttl_seconds=-1.0, is_indefinite=True)
        if ttl == 0:
            return KeepAlivePolicy(raw_value=val, ttl_seconds=0.0, is_immediate_unload=True)
        return KeepAlivePolicy(raw_value=val, ttl_seconds=ttl)

    raise ValueError(f"Unsupported keep_alive type: {type(val).__name__}.")


class LoadAttempt:
    """Attempt-scoped result object for a cold model load."""

    def __init__(self, key: EngineKey) -> None:
        self.key = key
        self.event: asyncio.Event = asyncio.Event()
        self.result: EngineLease | None = None
        self.exception: Exception | None = None

    def set_result(self, lease: EngineLease) -> None:
        if self.event.is_set():
            return
        self.result = lease
        self.event.set()

    def set_exception(self, exc: Exception) -> None:
        if self.event.is_set():
            return
        self.exception = exc
        self.event.set()

    async def wait_result(self) -> EngineLease:
        await self.event.wait()
        if self.exception is not None:
            raise self.exception
        assert self.result is not None
        return self.result


@dataclass
class EngineLease:
    """Active lease handle representing reference to a warm engine instance."""

    key: EngineKey
    engine: Any
    state: EngineState
    ref_count: int = 0
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    keep_alive_policy: KeepAlivePolicy | None = None
    attempt: LoadAttempt | None = None
    is_admitted: bool = False
    drained_event: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleaned_up: bool = False

    def remaining_keep_alive_seconds(self, now: float | None = None) -> float | None:
        """Return remaining keep_alive TTL in seconds, or None if active/indefinite."""
        if self.state != EngineState.READY:
            return None
        if self.ref_count > 0:
            return None
        policy = self.keep_alive_policy
        if policy is None:
            return 300.0
        if policy.is_indefinite:
            return float("inf")
        if policy.is_immediate_unload:
            return 0.0
        current = time.time() if now is None else now
        elapsed = current - self.last_accessed_at
        return max(0.0, policy.ttl_seconds - elapsed)


@dataclass(frozen=True)
class EngineStatus:
    """Immutable status snapshot of a managed engine."""

    key: EngineKey
    state: EngineState
    ref_count: int
    created_at: float
    last_accessed_at: float
    remaining_keep_alive_seconds: float | None


def is_oom_exception(exc: BaseException) -> bool:
    """Recognize PyTorch/CUDA Out-Of-Memory failure shapes without importing torch eagerly."""
    type_name = type(exc).__name__
    if "OutOfMemory" in type_name or "OOM" in type_name:
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or ("vram" in msg and "out of" in msg)
    )


async def _call_offloop_safe(func: Callable[..., Any], *args: Any) -> None:
    """Call a synchronous or asynchronous function off the main event loop thread."""
    if inspect.iscoroutinefunction(func):
        res = await func(*args)
    else:
        res = await asyncio.to_thread(func, *args)
        if inspect.isawaitable(res):
            await res


async def safe_async_cleanup(
    engine: Any, cleanup_hook: Callable[[Any], Any] | None = None
) -> None:
    """Safely shut down engine instance off-loop without calling ray.shutdown."""
    if engine is not None:
        # 1. Invoke engine shutdown or close if present off-loop
        if hasattr(engine, "shutdown") and callable(engine.shutdown):
            with contextlib.suppress(Exception):
                await _call_offloop_safe(engine.shutdown)
        elif hasattr(engine, "close") and callable(engine.close):
            with contextlib.suppress(Exception):
                await _call_offloop_safe(engine.close)

        # 2. Invoke cleanup_hook if provided off-loop
        if cleanup_hook is not None:
            with contextlib.suppress(Exception):
                await _call_offloop_safe(cleanup_hook, engine)

    # 3. Off-loop gc.collect
    with contextlib.suppress(Exception):
        await asyncio.to_thread(gc.collect)

    # 4. Off-loop torch.cuda.empty_cache
    if "torch" in sys.modules:
        with contextlib.suppress(Exception):
            torch_mod = sys.modules["torch"]
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                await asyncio.to_thread(torch_mod.cuda.empty_cache)


async def perform_lease_cleanup(
    lease: EngineLease, cleanup_hook: Callable[[Any], Any] | None = None
) -> None:
    """Guarantee cleanup is performed exactly once per lease."""
    async with lease.cleanup_lock:
        if lease.cleaned_up:
            return
        lease.cleaned_up = True
        await safe_async_cleanup(lease.engine, cleanup_hook)


async def _orphan_cleanup_worker(
    factory_task: asyncio.Task[Any],
    cleanup_hook: Callable[[Any], Any] | None,
) -> None:
    """Safely await orphaned background factory task and clean up any constructed engine instance."""
    try:
        engine = await factory_task
        if engine is not None:
            await safe_async_cleanup(engine, cleanup_hook)
    except Exception:
        pass


class EngineManager:
    """
    Supervisor managing warm vLLM engine instance lifecycle, reference counting,
    keep_alive timers, admission controls, deduplicated concurrent loads, and OOM recovery.
    """

    def __init__(
        self,
        settings: DevToolSettings | None = None,
        *,
        engine_factory: Callable[[EngineKey], Any] | None = None,
        clock: Callable[[], float] = time.time,
        admission_estimator: Callable[[EngineKey], Any] | None = None,
        cleanup_hook: Callable[[Any], Any] | None = None,
        max_loaded_models: int | None = None,
        load_concurrency: int | None = None,
        request_queue_depth: int | None = None,
        default_keep_alive: Any = None,
    ) -> None:
        self.settings = settings or DevToolSettings()
        self.engine_factory = engine_factory
        self.clock = clock
        self.admission_estimator = admission_estimator
        self.cleanup_hook = cleanup_hook
        self.max_loaded_models = (
            max_loaded_models
            if max_loaded_models is not None
            else self.settings.max_loaded_models
        )
        self.load_concurrency = (
            load_concurrency
            if load_concurrency is not None
            else self.settings.load_concurrency
        )
        self.request_queue_depth = (
            request_queue_depth
            if request_queue_depth is not None
            else self.settings.request_queue_depth
        )
        self.default_keep_alive = (
            default_keep_alive
            if default_keep_alive is not None
            else self.settings.default_keep_alive
        )

        if self.load_concurrency > self.max_loaded_models:
            raise ValueError(
                f"load_concurrency ({self.load_concurrency}) cannot exceed max_loaded_models ({self.max_loaded_models})."
            )

        self._shutting_down: bool = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._entries: dict[EngineKey, EngineLease] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._admission_lock: asyncio.Lock = asyncio.Lock()
        self._load_semaphore: asyncio.Semaphore = asyncio.Semaphore(self.load_concurrency)
        self._timer_tasks: dict[EngineKey, asyncio.Task[None]] = {}
        self._in_progress_loads: set[asyncio.Task[Any]] = set()
        self._queued_loads_count: int = 0

    @property
    def is_shutting_down(self) -> bool:
        """Return True if supervisor is shutting down."""
        return self._shutting_down


    async def _guarded_instantiate(self, key: EngineKey) -> Any:
        """Instantiate engine while acquiring and holding load_concurrency semaphore."""
        async with self._load_semaphore:
            return await self._instantiate_engine(key)

    async def acquire(
        self, key: EngineKey, keep_alive: Any = None
    ) -> EngineLease:
        """Acquire a reference-counted engine lease for the specified EngineKey."""
        policy_val = keep_alive if keep_alive is not None else self.default_keep_alive
        policy = parse_keep_alive(policy_val)

        attempt: LoadAttempt | None = None
        is_leader = False

        async with self._lock:
            if self._shutting_down:
                raise EngineShuttingDownError(
                    "EngineManager is shutting down.", code="ERR_ENGINE_SHUTTING_DOWN"
                )

            entry = self._entries.get(key)
            if entry is not None:
                if entry.state == EngineState.READY:
                    timer = self._timer_tasks.pop(key, None)
                    if timer and not timer.done():
                        timer.cancel()
                    entry.ref_count += 1
                    entry.last_accessed_at = self.clock()
                    entry.keep_alive_policy = policy
                    return EngineLease(
                        key=entry.key,
                        engine=entry.engine,
                        state=entry.state,
                        ref_count=entry.ref_count,
                        created_at=entry.created_at,
                        last_accessed_at=entry.last_accessed_at,
                        keep_alive_policy=entry.keep_alive_policy,
                    )
                elif entry.state in (EngineState.DRAINING, EngineState.UNLOADING):
                    raise EngineDrainingError(
                        f"Engine for key {key.repo_id} is draining and cannot accept new requests.",
                        code="ERR_ENGINE_DRAINING",
                    )
                elif entry.state == EngineState.FAILED:
                    raise RuntimeError(f"Engine for key {key.repo_id} is in FAILED state.")
                elif entry.state == EngineState.LOADING:
                    attempt = entry.attempt

            if attempt is None:
                if self._queued_loads_count >= self.request_queue_depth:
                    raise EngineQueueFullError(
                        f"Daemon request queue is full ({self.request_queue_depth} requests queued). Please retry later.",
                        code="ERR_QUEUE_FULL",
                    )
                self._queued_loads_count += 1
                attempt = LoadAttempt(key)
                new_entry = EngineLease(
                    key=key,
                    engine=None,
                    state=EngineState.LOADING,
                    created_at=self.clock(),
                    last_accessed_at=self.clock(),
                    keep_alive_policy=policy,
                    attempt=attempt,
                    is_admitted=False,
                )
                self._entries[key] = new_entry
                is_leader = True

        if not is_leader:
            assert attempt is not None
            try:
                await asyncio.shield(attempt.wait_result())
            except asyncio.CancelledError:
                raise
            except Exception:
                raise

            async with self._lock:
                active_entry = self._entries.get(key)
                if active_entry is not None and active_entry.state == EngineState.READY:
                    active_entry.ref_count += 1
                    active_entry.last_accessed_at = self.clock()
                    active_entry.keep_alive_policy = policy
                    return EngineLease(
                        key=active_entry.key,
                        engine=active_entry.engine,
                        state=active_entry.state,
                        ref_count=active_entry.ref_count,
                        created_at=active_entry.created_at,
                        last_accessed_at=active_entry.last_accessed_at,
                        keep_alive_policy=active_entry.keep_alive_policy,
                    )
                elif active_entry is not None and active_entry.state in (
                    EngineState.DRAINING,
                    EngineState.UNLOADING,
                ):
                    raise EngineDrainingError(
                        f"Engine for key {key.repo_id} is draining and cannot accept new requests.",
                        code="ERR_ENGINE_DRAINING",
                    )
                raise EngineDrainingError(
                    f"Engine for key {key.repo_id} failed or was drained.",
                    code="ERR_ENGINE_DRAINING",
                )

        # Leader execution path
        assert attempt is not None
        current_task = asyncio.current_task()
        if current_task is not None:
            self._in_progress_loads.add(current_task)

        factory_task: asyncio.Task[Any] | None = None
        constructed_engine: Any = None

        try:
            async with self._admission_lock:
                await self._enforce_capacity_for_leader(key)

                fits = await self._call_admission_estimator(key)
                if not fits:
                    evicted = await self._evict_one_idle_lru()
                    if evicted:
                        fits = await self._call_admission_estimator(key)
                    if not fits:
                        raise AdmissionVRAMError(
                            "Requested model fails VRAM admission check.",
                            code="ERR_ADMISSION_VRAM",
                        )

                async with self._lock:
                    if self._shutting_down:
                        raise EngineShuttingDownError(
                            "EngineManager is shutting down.", code="ERR_ENGINE_SHUTTING_DOWN"
                        )
                    leader_entry = self._entries.get(key)
                    if leader_entry is not None:
                        leader_entry.is_admitted = True

            factory_task = asyncio.create_task(self._guarded_instantiate(key))
            self._in_progress_loads.add(factory_task)

            engine = await asyncio.shield(factory_task)
            constructed_engine = engine

            temp_lease_to_cleanup: EngineLease | None = None
            result_lease: EngineLease | None = None
            error_to_raise: Exception | None = None

            async with self._lock:
                self._queued_loads_count = max(0, self._queued_loads_count - 1)
                e = self._entries.get(key)
                if self._shutting_down or e is None or e.state != EngineState.LOADING:
                    if e is not None:
                        self._entries.pop(key, None)
                    temp_lease_to_cleanup = EngineLease(
                        key=key, engine=engine, state=EngineState.UNLOADING
                    )
                    if self._shutting_down:
                        error_to_raise = EngineShuttingDownError(
                            "EngineManager is shutting down.", code="ERR_ENGINE_SHUTTING_DOWN"
                        )
                    else:
                        error_to_raise = EngineDrainingError(
                            f"Engine for key {key.repo_id} was drained during load.",
                            code="ERR_ENGINE_DRAINING",
                        )
                else:
                    now = self.clock()
                    e.engine = engine
                    e.state = EngineState.READY
                    e.ref_count = 1
                    e.created_at = now
                    e.last_accessed_at = now
                    e.keep_alive_policy = policy
                    result_lease = EngineLease(
                        key=e.key,
                        engine=e.engine,
                        state=e.state,
                        ref_count=e.ref_count,
                        created_at=e.created_at,
                        last_accessed_at=e.last_accessed_at,
                        keep_alive_policy=e.keep_alive_policy,
                    )

            if temp_lease_to_cleanup is not None:
                await perform_lease_cleanup(temp_lease_to_cleanup, self.cleanup_hook)
                assert error_to_raise is not None
                attempt.set_exception(error_to_raise)
                raise error_to_raise

            assert result_lease is not None
            attempt.set_result(result_lease)
            return result_lease

        except asyncio.CancelledError as cancel_exc:
            temp_lease_cancel: EngineLease | None = None
            async with self._lock:
                self._queued_loads_count = max(0, self._queued_loads_count - 1)
                e = self._entries.get(key)
                if e is not None and e.attempt is attempt:
                    self._entries.pop(key, None)
                    if e.engine is not None:
                        temp_lease_cancel = e

            if temp_lease_cancel is not None:
                await perform_lease_cleanup(temp_lease_cancel, self.cleanup_hook)

            if constructed_engine is not None:
                await safe_async_cleanup(constructed_engine, self.cleanup_hook)
            elif factory_task is not None:
                cleanup_task = asyncio.create_task(
                    _orphan_cleanup_worker(factory_task, self.cleanup_hook)
                )
                self._in_progress_loads.add(cleanup_task)
                cleanup_task.add_done_callback(lambda t: self._in_progress_loads.discard(t))

            follower_err = EngineLoadCancelledError(
                f"Cold load for key {key.repo_id} was cancelled by leader.",
                code="ERR_ENGINE_LOAD_CANCELLED",
            )
            attempt.set_exception(follower_err)
            raise cancel_exc
        except Exception as exc:
            temp_lease_fail: EngineLease | None = None
            async with self._lock:
                self._queued_loads_count = max(0, self._queued_loads_count - 1)
                e = self._entries.get(key)
                if e is not None and e.attempt is attempt:
                    self._entries.pop(key, None)
                    if e.engine is not None:
                        temp_lease_fail = e

            if temp_lease_fail is not None:
                await perform_lease_cleanup(temp_lease_fail, self.cleanup_hook)

            if is_oom_exception(exc):
                oom_err = EngineOOMError(
                    "CUDA out of memory during initialization.", code="ERR_ENGINE_FAILED_OOM"
                )
                attempt.set_exception(oom_err)
                raise oom_err from exc

            attempt.set_exception(exc)
            raise
        finally:
            if current_task is not None:
                self._in_progress_loads.discard(current_task)
            if factory_task is not None:
                self._in_progress_loads.discard(factory_task)

    async def _call_admission_estimator(self, key: EngineKey) -> bool:
        if self.admission_estimator is None:
            return True
        if inspect.iscoroutinefunction(self.admission_estimator):
            res = await self.admission_estimator(key)
        else:
            res = await asyncio.to_thread(self.admission_estimator, key)
            if inspect.isawaitable(res):
                res = await res
        return bool(res)

    async def _enforce_capacity_for_leader(self, key: EngineKey) -> None:
        async with self._lock:
            capacity_count = sum(
                1
                for k, e in self._entries.items()
                if k != key
                and (
                    e.is_admitted
                    or e.state in (EngineState.READY, EngineState.DRAINING, EngineState.UNLOADING)
                )
            )
            if capacity_count < self.max_loaded_models:
                return

        evicted = await self._evict_one_idle_lru()
        if not evicted:
            async with self._lock:
                capacity_count = sum(
                    1
                    for k, e in self._entries.items()
                    if k != key
                    and (
                        e.is_admitted
                        or e.state
                        in (EngineState.READY, EngineState.DRAINING, EngineState.UNLOADING)
                    )
                )
                if capacity_count >= self.max_loaded_models:
                    raise AdmissionVRAMError(
                        "Requested model requires more GPU VRAM or capacity than currently available.",
                        code="ERR_ADMISSION_VRAM",
                    )

    async def _evict_one_idle_lru(self) -> bool:
        target_lease: EngineLease | None = None
        target_key: EngineKey | None = None

        async with self._lock:
            idle_entries = [
                (k, e)
                for k, e in self._entries.items()
                if e.state == EngineState.READY and e.ref_count == 0
            ]
            if not idle_entries:
                return False
            idle_entries.sort(key=lambda item: item[1].last_accessed_at)
            target_key, target_lease = idle_entries[0]
            target_lease.state = EngineState.UNLOADING
            timer = self._timer_tasks.pop(target_key, None)
            if timer and not timer.done():
                timer.cancel()

        if target_lease is not None:
            await perform_lease_cleanup(target_lease, self.cleanup_hook)

        async with self._lock:
            if target_key is not None:
                self._entries.pop(target_key, None)
        return True

    async def _instantiate_engine(self, key: EngineKey) -> Any:
        if self.engine_factory is not None:
            if inspect.iscoroutinefunction(self.engine_factory):
                return await self.engine_factory(key)
            res = await asyncio.to_thread(self.engine_factory, key)
            if inspect.isawaitable(res):
                return await res
            return res

        from forgeai.core.backends.vllm_backend import VLLMBackend

        settings = key.to_settings()
        backend = VLLMBackend(settings, streaming=True, quiet_startup=True)
        await asyncio.to_thread(backend.initialize)
        return backend

    async def release(self, key: EngineKey, keep_alive: Any = None) -> None:
        """Release an engine lease reference and schedule keep_alive TTL eviction."""
        to_unload_lease: EngineLease | None = None
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return

            if keep_alive is not None:
                entry.keep_alive_policy = parse_keep_alive(keep_alive)

            if entry.ref_count > 0:
                entry.ref_count -= 1
            entry.last_accessed_at = self.clock()

            if entry.state == EngineState.DRAINING:
                if entry.ref_count == 0:
                    entry.drained_event.set()
                return

            if entry.state == EngineState.READY and entry.ref_count == 0:
                timer = self._timer_tasks.pop(key, None)
                if timer and not timer.done():
                    timer.cancel()

                policy = entry.keep_alive_policy or parse_keep_alive(self.default_keep_alive)
                if policy.is_immediate_unload:
                    entry.state = EngineState.UNLOADING
                    to_unload_lease = entry
                elif policy.is_indefinite:
                    pass
                elif policy.ttl_seconds > 0:
                    self._timer_tasks[key] = asyncio.create_task(
                        self._schedule_ttl_eviction(key, policy.ttl_seconds)
                    )

        if to_unload_lease is not None:
            await perform_lease_cleanup(to_unload_lease, self.cleanup_hook)
            async with self._lock:
                self._entries.pop(key, None)

    async def _schedule_ttl_eviction(self, key: EngineKey, ttl_seconds: float) -> None:
        to_unload_lease: EngineLease | None = None
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            return

        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state == EngineState.READY and entry.ref_count == 0:
                entry.state = EngineState.UNLOADING
                to_unload_lease = entry

        if to_unload_lease is not None:
            await perform_lease_cleanup(to_unload_lease, self.cleanup_hook)
            async with self._lock:
                self._entries.pop(key, None)

    async def stop(
        self, key_or_tag: EngineKey | str, drain_timeout: float = 10.0
    ) -> None:
        """Transition engine to DRAINING, await request completion, then unload."""
        target_key: EngineKey | None = None
        target_lease: EngineLease | None = None

        async with self._lock:
            for k, entry in self._entries.items():
                if isinstance(key_or_tag, EngineKey):
                    if k == key_or_tag:
                        target_key = k
                        target_lease = entry
                        break
                elif isinstance(key_or_tag, str) and (
                    k.repo_id == key_or_tag or k.snapshot_hash == key_or_tag
                ):
                    target_key = k
                    target_lease = entry
                    break

            if target_key is None or target_lease is None:
                return

            if target_lease.state in (EngineState.UNLOADING, EngineState.FAILED):
                return

            target_lease.state = EngineState.DRAINING
            timer = self._timer_tasks.pop(target_key, None)
            if timer and not timer.done():
                timer.cancel()

            if target_lease.ref_count == 0:
                target_lease.drained_event.set()

        if target_lease.ref_count > 0:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(target_lease.drained_event.wait(), timeout=drain_timeout)

        async with self._lock:
            target_lease.state = EngineState.UNLOADING

        await perform_lease_cleanup(target_lease, self.cleanup_hook)

        async with self._lock:
            self._entries.pop(target_key, None)

    async def mark_engine_failed(self, key: EngineKey, reason: str = "") -> None:
        """Mark a READY engine as FAILED and perform immediate cleanup."""
        target_lease: EngineLease | None = None
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.state = EngineState.FAILED
            timer = self._timer_tasks.pop(key, None)
            if timer and not timer.done():
                timer.cancel()
            target_lease = entry

        await perform_lease_cleanup(target_lease, self.cleanup_hook)

        async with self._lock:
            self._entries.pop(key, None)

    async def list_async(self) -> list[EngineStatus]:
        """Return a lock-safe status snapshot list of all managed engines."""
        async with self._lock:
            now = self.clock()
            return [
                EngineStatus(
                    key=entry.key,
                    state=entry.state,
                    ref_count=entry.ref_count,
                    created_at=entry.created_at,
                    last_accessed_at=entry.last_accessed_at,
                    remaining_keep_alive_seconds=entry.remaining_keep_alive_seconds(now=now),
                )
                for entry in list(self._entries.values())
            ]

    def list(self) -> list[EngineStatus]:
        """Synchronous best-effort status snapshot."""
        now = self.clock()
        return [
            EngineStatus(
                key=entry.key,
                state=entry.state,
                ref_count=entry.ref_count,
                created_at=entry.created_at,
                last_accessed_at=entry.last_accessed_at,
                remaining_keep_alive_seconds=entry.remaining_keep_alive_seconds(now=now),
            )
            for entry in list(self._entries.values())
        ]

    async def shutdown(self) -> None:
        """Shut down supervisor idempotently and unload all managed engines."""
        async with self._lock:
            if self._shutdown_task is not None:
                task = self._shutdown_task
            else:
                self._shutting_down = True
                task = asyncio.create_task(self._do_shutdown())
                self._shutdown_task = task

        await task

    async def _do_shutdown(self) -> None:
        timers = list(self._timer_tasks.values())
        for timer in timers:
            if not timer.done():
                timer.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._timer_tasks.clear()

        while True:
            loads = list(self._in_progress_loads)
            if not loads:
                break
            await asyncio.gather(*loads, return_exceptions=True)

        async with self._lock:
            keys_to_stop = list(self._entries.keys())

        for key in keys_to_stop:
            await self.stop(key, drain_timeout=2.0)


class DevToolEngine:
    """
    Wrapper for the vLLM inference backend with lifecycle management.
    Preserved for backward source compatibility.
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
        self._lock = asyncio.Lock()

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
            quiet_startup=self._quiet_startup,
        )
        self._backend.initialize()
        self._start_time = time.time()

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a model prompt."""
        if not self._backend:
            raise RuntimeError("Engine is not initialized.")
        return self._backend.build_prompt(messages)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        if not self.is_running or not self._backend:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        start = time.time()
        async with self._lock:
            result = await self._backend.generate(prompt, max_tokens, temperature, top_p, stop, top_k)
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
        top_k: int | None = None,
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

        async with self._lock:
            async for chunk in self._backend.generate_stream(
                prompt, max_tokens, temperature, top_p, stop, top_k
            ):

                chunks.append(chunk)
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

    def get_status(self) -> LegacyEngineStatus:
        """Get current engine status."""
        return LegacyEngineStatus(
            model_name=self.settings.model_name,
            backend=self.settings.backend.value if self.settings.backend else "vllm",
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
