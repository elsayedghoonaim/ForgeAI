"""FastAPI server factory for the ForgeAI API service."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from forgeai import __version__
from forgeai.api.routes import chat, health, models, ollama
from forgeai.core.engine import EngineManagerError
from forgeai.monitoring.logging import get_logger
from forgeai.monitoring.metrics import ACTIVE_REQUESTS, ENGINE_STATUS, record_request
from forgeai.security.middleware import RateLimitMiddleware, sanitize_request_id

logger = get_logger(__name__)


def _request_id(request: Request, app: FastAPI) -> tuple[str, str]:
    settings = getattr(app.state, "settings", None) or SimpleNamespace(request_id_header="X-Request-ID")
    header_name = getattr(settings, "request_id_header", "X-Request-ID")
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = sanitize_request_id(request.headers.get(header_name))
        request.state.request_id = request_id
    return request_id, header_name


def _error_response(
    request: Request,
    app: FastAPI,
    status_code: int,
    message: str,
    error_type: str,
    details: object | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id, header_name = _request_id(request, app)
    merged_headers = {header_name: request_id}
    if headers:
        merged_headers.update(headers)

    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=status_code,
            headers=merged_headers,
            content={"error": message},
        )

    error_payload: dict[str, object] = {
        "type": error_type,
        "message": message,
        "status_code": status_code,
        "request_id": request_id,
    }
    if details is not None:
        error_payload["details"] = details
    payload: dict[str, object] = {"error": error_payload}
    return JSONResponse(
        status_code=status_code,
        headers=merged_headers,
        content=payload,
    )


class RequestLoggingMiddleware:
    """Pure-ASGI request-id, access-log and metrics middleware.

    Unlike ``BaseHTTPMiddleware`` it does not wrap the response in a second task or stream,
    so client disconnects reach the application and per-chunk overhead is nil. Metrics,
    latency and the access log are finalized exactly once: when the final body message
    (``more_body`` false) is sent, when the client disconnects, or when the app raises or
    returns, whichever happens first. ``ACTIVE_REQUESTS`` is decremented at that point.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        settings = getattr(getattr(app, "state", None), "settings", None)
        header_name = getattr(settings, "request_id_header", "X-Request-ID")

        raw_id: str | None = None
        wanted = header_name.lower().encode("latin-1")
        for name, value in scope.get("headers", []):
            if name.lower() == wanted:
                raw_id = value.decode("latin-1")
                break
        request_id = sanitize_request_id(raw_id)

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["client_ip"] = client_ip

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.monotonic()
        status_code = 0
        finalized = False
        ACTIVE_REQUESTS.inc()

        def finalize(status_override: int | None = None) -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            ACTIVE_REQUESTS.dec()
            status = status_override if status_override is not None else status_code
            elapsed = time.monotonic() - start
            try:
                record_request(
                    method=method,
                    status=str(status),
                    duration=elapsed,
                    tokens=state.get("completion_tokens", 0),
                    prompt_tokens=state.get("prompt_tokens", 0),
                )
                logger.info(
                    "request",
                    method=method,
                    path=path,
                    status=status,
                    duration_ms=round(elapsed * 1000, 1),
                    request_id=request_id,
                    actor=state.get("actor_id"),
                    permission=state.get("required_permission"),
                    client_ip=client_ip,
                )
            except Exception:  # never let observability break request handling
                pass

        async def receive_wrapper() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                finalize(status_code or 499)
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).setdefault(header_name, request_id)
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finalize()

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except BaseException:
            finalize(status_code or 500)
            raise
        finally:
            finalize(status_code or 500)


def create_app(
    engine: Any = None,
    title: str = "ForgeAI API",
    enable_cors: bool = True,
    enable_auth: bool = False,
    auth_manager: Any = None,
    settings: Any = None,
    audit_logger: Any = None,
    rate_limiter: Any = None,
    engine_manager: Any = None,
    model_registry: Any = None,
    cache_manager: Any = None,
    runtime_adapter: Any = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        engine: DevToolEngine instance to serve (legacy).
        title: API title.
        enable_cors: Allow CORS middleware to be installed. It is only active when
            ``settings.cors_allow_origins`` lists explicit origins (empty by default), so no
            untrusted browser origin can reach a local ForgeAI daemon unless configured.
        enable_auth: Enable authentication middleware.
        auth_manager: Authentication manager used by the middleware.
        settings: Runtime settings object.
        audit_logger: Audit logger used by security middleware.
        rate_limiter: Rate limiter used by security middleware.
        engine_manager: EngineManager supervisor instance.
        model_registry: ModelRegistry instance.
        cache_manager: CacheManager instance.
        runtime_adapter: SharedRuntimeAdapter instance.
    """

    if enable_auth and auth_manager is None:
        raise ValueError("Authentication is enabled but no auth_manager was provided.")

    if runtime_adapter is None and (
        engine_manager is not None or model_registry is not None or cache_manager is not None
    ):
        from forgeai.api.runtime import SharedRuntimeAdapter

        runtime_adapter = SharedRuntimeAdapter(
            engine_manager=engine_manager,
            model_registry=model_registry,
            cache_manager=cache_manager,
            settings=settings,
        )

    if runtime_adapter is not None:
        if engine_manager is None:
            engine_manager = runtime_adapter.engine_manager
        if model_registry is None:
            model_registry = runtime_adapter.model_registry
        if cache_manager is None:
            cache_manager = runtime_adapter.cache_manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        mgr = getattr(app.state, "engine_manager", None)
        if mgr is not None and hasattr(mgr, "shutdown"):
            await mgr.shutdown()

    docs_enabled = getattr(settings, "docs_enabled", None)
    if docs_enabled is None:
        docs_enabled = not enable_auth

    app = FastAPI(
        title=title,
        version=__version__,
        description="OpenAI and Ollama compatible API powered by ForgeAI",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
    )

    app.state.engine = engine
    app.state.engine_manager = engine_manager
    app.state.model_registry = model_registry
    app.state.cache_manager = cache_manager
    app.state.runtime_adapter = runtime_adapter
    app.state.auth_manager = auth_manager
    app.state.settings = settings or SimpleNamespace(request_id_header="X-Request-ID")
    app.state.audit_logger = audit_logger
    app.state.rate_limiter = rate_limiter
    ENGINE_STATUS.set(1 if engine is not None and getattr(engine, "is_running", False) else 0)

    @app.exception_handler(EngineManagerError)
    async def handle_engine_manager_error(request: Request, exc: EngineManagerError) -> JSONResponse:
        status_code = 503
        if exc.code == "ERR_QUEUE_FULL":
            status_code = 429
        elif exc.code == "ERR_ENGINE_FAILED_OOM":
            status_code = 503

        request_id, header_name = _request_id(request, request.app)
        headers = {header_name: request_id, "X-ForgeAI-Error-Code": exc.code}
        return _error_response(
            request,
            request.app,
            status_code,
            str(exc),
            exc.code,
            headers=headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            request,
            request.app,
            exc.status_code,
            str(exc.detail),
            "http_error",
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            request.app,
            422,
            "Request validation failed.",
            "validation_error",
            details=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id, _ = _request_id(request, request.app)
        logger.error(
            "request_error",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            error=str(exc),
        )
        return _error_response(
            request,
            request.app,
            500,
            "Internal server error.",
            "internal_error",
        )

    # Middleware order (the last one added is outermost):
    #   CORS -> request id / logging / metrics -> IP rate limit -> auth -> actor rate limit.
    # CORS wraps everything so 401/403/429 responses carry CORS headers, and the logging layer
    # wraps the limiters and auth so those rejections are logged and counted too.
    app.add_middleware(RateLimitMiddleware, mode="actor")
    if enable_auth:
        from forgeai.security.middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware, mode="ip")
    app.add_middleware(RequestLoggingMiddleware)

    cors_origins = list(getattr(settings, "cors_allow_origins", None) or [])
    if enable_cors and cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router, tags=["Health"])
    app.include_router(models.router, prefix="/v1", tags=["Models"])
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])
    app.include_router(ollama.router, prefix="/api", tags=["Ollama"])

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST

            from forgeai.monitoring.metrics import generate_metrics

            return Response(content=generate_metrics(), media_type=CONTENT_TYPE_LATEST)
        except Exception:
            return Response(content="# No metrics available", media_type="text/plain")

    return app
