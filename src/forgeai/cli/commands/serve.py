"""forgeai serve - start the Ollama-compatible daemon server."""

from __future__ import annotations

from ipaddress import ip_address

import typer

from forgeai.cli.runtime import handle_cli_error


def _is_loopback_host(host: str) -> bool:
    """Return True only for loopback bind addresses/names."""
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(11434, "--port", help="Server port"),
    workers: int = typer.Option(1, "--workers", help="Number of UV workers"),
    enable_auth: bool = typer.Option(False, "--auth", help="Enable API authentication"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
    keep_alive: str = typer.Option("5m", "--keep-alive", help="Default engine keep_alive TTL"),
    max_loaded_models: int = typer.Option(1, "--max-loaded-models", help="Max loaded engines in VRAM"),
    load_concurrency: int = typer.Option(1, "--load-concurrency", help="Model load concurrency limit"),
    request_queue_depth: int = typer.Option(32, "--request-queue-depth", help="Max queued load requests"),
    gpu_utilization: float | None = typer.Option(None, "--gpu-util", help="Target GPU memory utilization"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallel GPUs"),
    max_model_len: int | None = typer.Option(None, "--max-model-len", help="Maximum context length"),
    max_num_seqs: int = typer.Option(4, "--max-num-seqs", help="Max concurrent sequences"),
    kv_cache_dtype: str = typer.Option("auto", "--kv-cache-dtype", help="KV cache quantization format"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager", help="Favor eager execution over compile-heavy startup"),
) -> None:
    """Initialize the ForgeAI API server with health checks and metrics."""

    if workers != 1:
        handle_cli_error("workers must be set to 1. The current runtime keeps one in-process engine per API process.")

    if not (1 <= port <= 65535):
        handle_cli_error(f"Port number {port} out of valid range (1-65535).")

    if max_loaded_models < 1:
        handle_cli_error(f"max_loaded_models must be >= 1, got {max_loaded_models}.")

    if load_concurrency < 1:
        handle_cli_error(f"load_concurrency must be >= 1, got {load_concurrency}.")

    if load_concurrency > max_loaded_models:
        handle_cli_error(
            f"load_concurrency ({load_concurrency}) cannot exceed max_loaded_models ({max_loaded_models})."
        )

    if request_queue_depth < 1:
        handle_cli_error(f"request_queue_depth must be >= 1, got {request_queue_depth}.")

    if gpu_utilization is not None and not (0.1 <= gpu_utilization <= 1.0):
        handle_cli_error(f"gpu_utilization must be between 0.1 and 1.0, got {gpu_utilization}.")

    if tensor_parallel < 1:
        handle_cli_error(f"tensor_parallel must be >= 1, got {tensor_parallel}.")

    if max_model_len is not None and max_model_len < 1:
        handle_cli_error(f"max_model_len must be >= 1, got {max_model_len}.")

    if max_num_seqs < 1:
        handle_cli_error(f"max_num_seqs must be >= 1, got {max_num_seqs}.")

    from pydantic import ValidationError
    from rich.console import Console

    from forgeai.api.runtime import SharedRuntimeAdapter
    from forgeai.api.server import create_app
    from forgeai.core.config import DevToolSettings
    from forgeai.core.engine import EngineManager
    from forgeai.core.telemetry import track_event
    from forgeai.models.loader import CacheManager
    from forgeai.models.registry import ModelRegistry
    from forgeai.monitoring.logging import setup_logging
    from forgeai.security.auth import AuthManager, Role
    from forgeai.security.compliance.audit_logger import AuditLogger
    from forgeai.security.rate_limit import MemoryRateLimiter

    console = Console()

    try:
        bootstrap_settings = DevToolSettings()
        setup_logging(level=log_level, json_output=bootstrap_settings.log_json)

        auth_requested = enable_auth or bootstrap_settings.auth_enabled
        if not auth_requested and not _is_loopback_host(host):
            handle_cli_error(
                "Refusing to bind ForgeAI to a non-loopback interface without authentication. "
                "Use --auth and configure FORGEAI_AUTH_SECRET_KEY plus "
                "FORGEAI_BOOTSTRAP_API_KEY."
            )

        settings_kwargs: dict[str, object] = {
            "host": host,
            "port": port,
            "auth_enabled": auth_requested,
            "default_keep_alive": keep_alive,
            "max_loaded_models": max_loaded_models,
            "load_concurrency": load_concurrency,
            "request_queue_depth": request_queue_depth,
            "tensor_parallel_size": tensor_parallel,
            "max_num_seqs": max_num_seqs,
            "kv_cache_dtype": kv_cache_dtype,
            "enforce_eager": enforce_eager,
        }
        if gpu_utilization is not None:
            settings_kwargs["gpu_memory_utilization"] = gpu_utilization
        if max_model_len is not None:
            settings_kwargs["max_model_len"] = max_model_len

        settings = DevToolSettings(**settings_kwargs)

        auth_manager = None
        if auth_requested:
            if settings.auth_secret_key == "change-me-in-production":
                handle_cli_error("auth is enabled but the secret key is still the default. Set FORGEAI_AUTH_SECRET_KEY before starting the server.")
            if not settings.bootstrap_api_key:
                handle_cli_error("auth is enabled but no bootstrap API key is configured. Set FORGEAI_BOOTSTRAP_API_KEY before starting the server.")

            auth_manager = AuthManager(
                secret_key=settings.auth_secret_key,
                algorithm=settings.auth_algorithm,
                token_expire_minutes=settings.auth_token_expire_minutes,
            )
            try:
                bootstrap_role = Role(settings.bootstrap_api_key_role)
            except ValueError as err:
                handle_cli_error(err)
            auth_manager.register_api_key(
                raw_key=settings.bootstrap_api_key,
                name=settings.bootstrap_api_key_name,
                role=bootstrap_role,
            )
    except (ValueError, TypeError, ValidationError) as err:
        handle_cli_error(err)

    console.print("\n[bold cyan]ForgeAI Server[/bold cyan]")
    console.print(f"  Address:  http://{host}:{port}")
    console.print(f"  Auth:     {'enabled' if auth_requested else 'disabled'}")
    console.print(f"  Docs:     http://{host}:{port}/docs\n")

    track_event("command.serve", {"host": host, "port": port})

    if auth_requested and auth_manager:
        console.print(
            f"[green]OK[/green] registered bootstrap API key "
            f"'{settings.bootstrap_api_key_name}' with role={settings.bootstrap_api_key_role}"
        )

    audit_logger = AuditLogger(settings.audit_log_dir) if settings.audit_logging_enabled else None
    rate_limiter = None
    if settings.rate_limit_enabled:
        rate_limiter = MemoryRateLimiter(
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )

    model_registry = ModelRegistry()
    cache_manager = CacheManager()
    engine_manager = EngineManager(settings=settings)
    runtime_adapter = SharedRuntimeAdapter(
        engine_manager=engine_manager,
        model_registry=model_registry,
        cache_manager=cache_manager,
        settings=settings,
    )

    import uvicorn

    api = create_app(
        runtime_adapter=runtime_adapter,
        enable_auth=auth_requested,
        auth_manager=auth_manager,
        settings=settings,
        audit_logger=audit_logger,
        rate_limiter=rate_limiter,
    )

    try:
        uvicorn.run(api, host=host, port=port, workers=workers, log_level=log_level)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server shutting down...[/yellow]")
