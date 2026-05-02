"""forgeai serve - start the API server."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

console = Console()


def serve(
    model: str = typer.Argument("", help="Model name or HuggingFace repo ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    gpu_utilization: float | None = typer.Option(
        None,
        "--gpu-util",
        help="GPU memory utilization (auto-tuned when omitted)",
    ),
    tensor_parallel: int | None = typer.Option(None, "--tp", help="Tensor parallel size"),
    auto_optimize: bool = typer.Option(False, "--auto-optimize", help="Auto-tune tensor parallel size"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, vllm, llama_cpp"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", help="GPU layers for llama.cpp (-1 = all)"),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window for llama.cpp"),
    enable_auth: bool = typer.Option(False, "--auth", help="Enable API authentication"),
    workers: int = typer.Option(1, "--workers", help="Number of UV workers"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
) -> None:
    """Initialize the ForgeAI API server with health checks and metrics."""

    from forgeai.cli.runtime import print_runtime_tuning, resolve_runtime_tuning
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name
    from forgeai.monitoring.logging import setup_logging
    from forgeai.security.auth import AuthManager, Role
    from forgeai.security.compliance.audit_logger import AuditLogger
    from forgeai.security.rate_limit import MemoryRateLimiter

    bootstrap_settings = DevToolSettings()
    setup_logging(level=log_level, json_output=bootstrap_settings.log_json)

    configured_model = model or bootstrap_settings.model_name
    if not configured_model:
        console.print("[red]ERROR:[/red] no model configured.")
        console.print(
            "  Provide a model argument or set forgeai_MODEL_NAME in the environment."
        )
        raise typer.Exit(code=1)

    if workers != 1:
        console.print("[red]ERROR:[/red] workers must be set to 1.")
        console.print("  The current runtime keeps one in-process engine per API process.")
        raise typer.Exit(code=1)

    resolved = resolve_model_name(configured_model)
    auth_requested = enable_auth or bootstrap_settings.auth_enabled

    console.print("\n[bold cyan]ForgeAI Server[/bold cyan]")
    console.print(f"  Model:    {resolved}")
    console.print(f"  Address:  http://{host}:{port}")
    console.print(f"  Auth:     {'enabled' if auth_requested else 'disabled'}")
    console.print(f"  Docs:     http://{host}:{port}/docs\n")

    track_event("command.serve", {"model": resolved, "port": port})

    tuning = resolve_runtime_tuning(
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_utilization,
        auto_optimize=auto_optimize,
    )
    print_runtime_tuning(tuning)

    if backend in ("llama_cpp", "auto") and not resolved.endswith(".gguf") and backend == "llama_cpp":
        from forgeai.models.gguf_finder import find_gguf_for_model
        console.print(f"[dim]Searching HuggingFace for GGUF variants of {resolved}...[/dim]")
        candidates = find_gguf_for_model(resolved)
        if candidates:
            best = candidates[0]
            console.print(f"[dim]Found GGUF variant: {best.repo_id} / {best.filename}[/dim]")
            try:
                from huggingface_hub import hf_hub_download
                console.print(f"[dim]Downloading {best.filename} (this may take a while)...[/dim]")
                resolved = hf_hub_download(repo_id=best.repo_id, filename=best.filename)
                console.print(f"[green]OK[/green] Downloaded to {resolved}")
            except Exception as e:
                console.print(f"[red]ERROR:[/red] Failed to download GGUF: {e}")
                raise typer.Exit(code=1) from e
        else:
            console.print(f"\n[red]ERROR:[/red] Model \"{resolved}\" is not in GGUF format, and no GGUF variants were found.")
            console.print("\nThe llama_cpp backend requires GGUF models. Options:")
            console.print(f"  1. Use the vllm backend instead:\n     forgeai serve {resolved} --backend vllm")
            console.print("  2. Convert it manually to GGUF using llama.cpp scripts.")
            raise typer.Exit(code=1) from None

    settings = DevToolSettings(
        model_name=resolved,
        backend=backend,
        host=host,
        port=port,
        gpu_memory_utilization=tuning.gpu_memory_utilization,
        tensor_parallel_size=tuning.tensor_parallel_size,
        auth_enabled=auth_requested,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )

    auth_manager = None
    if auth_requested:
        if settings.auth_secret_key == "change-me-in-production":
            console.print(
                "[red]ERROR:[/red] auth is enabled but the secret key is still the default."
            )
            console.print("  Set forgeai_AUTH_SECRET_KEY before starting the server.")
            raise typer.Exit(code=1)
        if not settings.bootstrap_api_key:
            console.print("[red]ERROR:[/red] auth is enabled but no bootstrap API key is configured.")
            console.print("  Set forgeai_BOOTSTRAP_API_KEY before starting the server.")
            raise typer.Exit(code=1)

        auth_manager = AuthManager(
            secret_key=settings.auth_secret_key,
            algorithm=settings.auth_algorithm,
            token_expire_minutes=settings.auth_token_expire_minutes,
        )
        try:
            bootstrap_role = Role(settings.bootstrap_api_key_role)
        except ValueError as err:
            raise typer.Exit(code=1) from err
        auth_manager.register_api_key(
            raw_key=settings.bootstrap_api_key,
            name=settings.bootstrap_api_key_name,
            role=bootstrap_role,
        )
        console.print(
            f"[green]OK[/green] registered bootstrap API key "
            f"'{settings.bootstrap_api_key_name}' with role={bootstrap_role.value}"
        )

    settings.ensure_directories()

    audit_logger = AuditLogger(settings.audit_log_dir) if settings.audit_logging_enabled else None
    rate_limiter = None
    if settings.rate_limit_enabled:
        rate_limiter = MemoryRateLimiter(
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )

    try:
        from forgeai.core.engine import DevToolEngine

        engine = DevToolEngine(settings)
        engine.initialize()
    except Exception as err:
        console.print(f"[red]ERROR:[/red] engine initialization failed: {escape(str(err))}")
        raise typer.Exit(code=1) from err

    import uvicorn

    from forgeai.api.server import create_app

    api = create_app(
        engine=engine,
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
    finally:
        engine.shutdown()
