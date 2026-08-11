"""Prometheus metric exports for the API service and runtime monitoring."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Bounded label validation sets
VALID_ENGINE_STATES: set[str] = {
    "unloaded",
    "loading",
    "ready",
    "draining",
    "evicted",
    "failed",
    "stopped",
    "other",
}

VALID_EVICTION_REASONS: set[str] = {
    "idle_timeout",
    "vram_pressure",
    "manual",
    "lru",
    "oom",
    "other",
}

VALID_REJECTION_REASONS: set[str] = {
    "vram_exceeded",
    "queue_full",
    "draining",
    "shutting_down",
    "unsupported_config",
    "other",
}

# --- Existing API Metrics (Preserved for backward compatibility) ---

REQUESTS_TOTAL = Counter(
    "forgeai_requests_total",
    "Total number of HTTP requests",
    ["method", "status"],
)

REQUEST_DURATION = Histogram(
    "forgeai_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

TOKENS_GENERATED = Counter(
    "forgeai_tokens_generated_total",
    "Total completion tokens generated",
)

TOKENS_PER_SECOND = Gauge(
    "forgeai_tokens_per_second",
    "Current completion token generation rate",
)

PROMPT_TOKENS = Counter(
    "forgeai_prompt_tokens_total",
    "Total prompt tokens processed",
)

ENGINE_STATUS = Gauge(
    "forgeai_engine_running",
    "Whether the engine is running (1=running, 0=stopped)",
)

ACTIVE_REQUESTS = Gauge(
    "forgeai_active_requests",
    "Number of currently active requests",
)

GPU_MEMORY_USED = Gauge(
    "forgeai_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
)

GPU_MEMORY_TOTAL = Gauge(
    "forgeai_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    ["gpu_id"],
)

GPU_UTILIZATION = Gauge(
    "forgeai_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)


# --- New Low-Cardinality Resource & Performance Metrics ---

PROCESS_RSS_BYTES = Gauge(
    "forgeai_process_rss_bytes",
    "Process RSS memory usage in bytes",
)

GPU_MEMORY_FREE = Gauge(
    "forgeai_gpu_memory_free_bytes",
    "GPU memory free in bytes",
    ["gpu_id"],
)

GPU_MEMORY_RESERVED = Gauge(
    "forgeai_gpu_memory_reserved_bytes",
    "GPU memory reserved in bytes",
    ["gpu_id"],
)

ENGINES_BY_STATE = Gauge(
    "forgeai_engines_by_state",
    "Number of engines by bounded state",
    ["state"],
)

MODEL_LOAD_DURATION = Histogram(
    "forgeai_model_load_duration_seconds",
    "Model load duration in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

MODEL_LOAD_PEAK_MEMORY = Gauge(
    "forgeai_model_load_peak_memory_bytes",
    "Peak GPU memory during model load in bytes",
)

KV_CACHE_CAPACITY_BYTES = Gauge(
    "forgeai_kv_cache_capacity_bytes",
    "KV cache total capacity in bytes",
)

KV_CACHE_USAGE_BYTES = Gauge(
    "forgeai_kv_cache_usage_bytes",
    "KV cache memory usage in bytes",
)

KV_CACHE_UTILIZATION = Gauge(
    "forgeai_kv_cache_utilization_ratio",
    "KV cache utilization ratio (0.0 to 1.0)",
)

QUEUED_REQUESTS = Gauge(
    "forgeai_queued_requests",
    "Number of queued requests waiting for execution",
)

TIME_TO_FIRST_TOKEN = Histogram(
    "forgeai_time_to_first_token_seconds",
    "Time to first token (TTFT) in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

DECODE_THROUGHPUT = Gauge(
    "forgeai_decode_throughput_tokens_per_second",
    "Decode token throughput per second",
)

ENGINE_EVICTIONS_TOTAL = Counter(
    "forgeai_engine_evictions_total",
    "Total engine evictions by bounded reason",
    ["reason"],
)

ADMISSION_REJECTIONS_TOTAL = Counter(
    "forgeai_admission_rejections_total",
    "Total admission rejections by bounded reason",
    ["reason"],
)

OOM_EVENTS_TOTAL = Counter(
    "forgeai_oom_events_total",
    "Total out-of-memory (OOM) events encountered",
)


# --- Helper Functions & Label Validation ---


def record_request(
    method: str,
    status: str,
    duration: float,
    tokens: int = 0,
    prompt_tokens: int = 0,
) -> None:
    """Record metrics for a completed request."""
    REQUESTS_TOTAL.labels(method=method, status=status).inc()
    REQUEST_DURATION.labels(method=method).observe(duration)
    if prompt_tokens > 0:
        PROMPT_TOKENS.inc(prompt_tokens)
    if tokens > 0:
        TOKENS_GENERATED.inc(tokens)
        if duration > 0:
            TOKENS_PER_SECOND.set(tokens / duration)


def update_gpu_metrics(
    gpu_id: int,
    used_bytes: float,
    total_bytes: float,
    util_pct: float,
    free_bytes: float | None = None,
    reserved_bytes: float | None = None,
) -> None:
    """Update GPU metrics for a specific device."""
    gpu_str = str(gpu_id)
    GPU_MEMORY_USED.labels(gpu_id=gpu_str).set(used_bytes)
    GPU_MEMORY_TOTAL.labels(gpu_id=gpu_str).set(total_bytes)
    GPU_UTILIZATION.labels(gpu_id=gpu_str).set(util_pct)

    if free_bytes is not None:
        GPU_MEMORY_FREE.labels(gpu_id=gpu_str).set(free_bytes)
    else:
        GPU_MEMORY_FREE.labels(gpu_id=gpu_str).set(max(0.0, total_bytes - used_bytes))

    if reserved_bytes is not None:
        GPU_MEMORY_RESERVED.labels(gpu_id=gpu_str).set(reserved_bytes)


def update_process_rss(rss_bytes: float) -> None:
    """Update process RSS memory metric."""
    PROCESS_RSS_BYTES.set(rss_bytes)


def record_engine_state(state: str, count: int = 1) -> None:
    """Record count of engines in a specific bounded state."""
    safe_state = state.lower().strip() if state.lower().strip() in VALID_ENGINE_STATES else "other"
    ENGINES_BY_STATE.labels(state=safe_state).set(count)


def record_model_load(duration_seconds: float, peak_memory_bytes: float = 0.0) -> None:
    """Record model load duration and peak VRAM usage."""
    if duration_seconds >= 0:
        MODEL_LOAD_DURATION.observe(duration_seconds)
    if peak_memory_bytes >= 0:
        MODEL_LOAD_PEAK_MEMORY.set(peak_memory_bytes)


def update_kv_cache_metrics(capacity_bytes: float, usage_bytes: float) -> None:
    """Update KV cache capacity, usage, and utilization metrics."""
    KV_CACHE_CAPACITY_BYTES.set(capacity_bytes)
    KV_CACHE_USAGE_BYTES.set(usage_bytes)
    ratio = (usage_bytes / capacity_bytes) if capacity_bytes > 0 else 0.0
    KV_CACHE_UTILIZATION.set(max(0.0, min(1.0, ratio)))


def update_queued_requests(count: int) -> None:
    """Update active queued requests count."""
    QUEUED_REQUESTS.set(max(0, count))


def record_ttft(duration_seconds: float) -> None:
    """Record time to first token (TTFT)."""
    if duration_seconds >= 0:
        TIME_TO_FIRST_TOKEN.observe(duration_seconds)


def update_decode_throughput(tokens_per_second: float) -> None:
    """Update decode throughput rate."""
    DECODE_THROUGHPUT.set(max(0.0, tokens_per_second))


def record_engine_eviction(reason: str) -> None:
    """Record engine eviction by bounded reason."""
    safe_reason = reason.lower().strip() if reason.lower().strip() in VALID_EVICTION_REASONS else "other"
    ENGINE_EVICTIONS_TOTAL.labels(reason=safe_reason).inc()


def record_admission_rejection(reason: str) -> None:
    """Record admission rejection by bounded reason."""
    safe_reason = reason.lower().strip() if reason.lower().strip() in VALID_REJECTION_REASONS else "other"
    ADMISSION_REJECTIONS_TOTAL.labels(reason=safe_reason).inc()


def record_oom_event() -> None:
    """Record an out-of-memory (OOM) event."""
    OOM_EVENTS_TOTAL.inc()


def generate_metrics() -> str:
    """Generate Prometheus-format metrics text."""
    return generate_latest().decode("utf-8")
