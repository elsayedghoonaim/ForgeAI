"""Prometheus metric exports for the API service."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

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


def update_gpu_metrics(gpu_id: int, used_bytes: float, total_bytes: float, util_pct: float) -> None:
    """Update GPU metrics for a specific device."""

    GPU_MEMORY_USED.labels(gpu_id=str(gpu_id)).set(used_bytes)
    GPU_MEMORY_TOTAL.labels(gpu_id=str(gpu_id)).set(total_bytes)
    GPU_UTILIZATION.labels(gpu_id=str(gpu_id)).set(util_pct)


def generate_metrics() -> str:
    """Generate Prometheus-format metrics text."""

    return generate_latest().decode("utf-8")
