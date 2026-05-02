"""FastAPI server factory for the ForgeAI API service."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from forgeai import __version__
from forgeai.api.routes import chat, health, models
from forgeai.monitoring.logging import get_logger
from forgeai.monitoring.metrics import ACTIVE_REQUESTS, ENGINE_STATUS, record_request

logger = get_logger(__name__)


def _request_id(request: Request, app: FastAPI) -> tuple[str, str]:
    settings = getattr(app.state, "settings", None) or SimpleNamespace(request_id_header="X-Request-ID")
    header_name = getattr(settings, "request_id_header", "X-Request-ID")
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = request.headers.get(header_name) or uuid4().hex
        request.state.request_id = request_id
    return request_id, header_name


def _error_response(
    request: Request,
    app: FastAPI,
    status_code: int,
    message: str,
    error_type: str,
    details: object | None = None,
) -> JSONResponse:
    request_id, header_name = _request_id(request, app)
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
        headers={header_name: request_id},
        content=payload,
    )


def create_app(
    engine: Any = None,
    title: str = "ForgeAI API",
    enable_cors: bool = True,
    enable_auth: bool = False,
    auth_manager: Any = None,
    settings: Any = None,
    audit_logger: Any = None,
    rate_limiter: Any = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        engine: DevToolEngine instance to serve.
        title: API title.
        enable_cors: Enable CORS middleware.
        enable_auth: Enable authentication middleware.
        auth_manager: Authentication manager used by the middleware.
        settings: Runtime settings object.
        audit_logger: Audit logger used by security middleware.
        rate_limiter: Rate limiter used by security middleware.
    """

    if enable_auth and auth_manager is None:
        raise ValueError("Authentication is enabled but no auth_manager was provided.")

    app = FastAPI(
        title=title,
        version=__version__,
        description="OpenAI-compatible API powered by ForgeAI",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.engine = engine
    app.state.auth_manager = auth_manager
    app.state.settings = settings or SimpleNamespace(request_id_header="X-Request-ID")
    app.state.audit_logger = audit_logger
    app.state.rate_limiter = rate_limiter
    ENGINE_STATUS.set(1 if engine is not None and engine.is_running else 0)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
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

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id, header_name = _request_id(request, request.app)
        request.state.client_ip = request.client.host if request.client else "unknown"

        start = time.time()
        ACTIVE_REQUESTS.inc()
        try:
            response: Response = await call_next(request)
            ACTIVE_REQUESTS.dec()
            elapsed = time.time() - start
            response.headers.setdefault(header_name, request_id)
            record_request(
                method=request.method,
                status=str(response.status_code),
                duration=elapsed,
                tokens=getattr(request.state, "completion_tokens", 0),
                prompt_tokens=getattr(request.state, "prompt_tokens", 0),
            )
            logger.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(elapsed * 1000, 1),
                request_id=request_id,
                actor=getattr(request.state, "actor_id", None),
                permission=getattr(request.state, "required_permission", None),
                client_ip=request.state.client_ip,
            )
            return response
        except Exception:
            ACTIVE_REQUESTS.dec()
            raise

    if enable_auth:
        from forgeai.security.middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware)

    app.include_router(health.router, tags=["Health"])
    app.include_router(models.router, prefix="/v1", tags=["Models"])
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        try:
            from forgeai.monitoring.metrics import generate_metrics

            return Response(content=generate_metrics(), media_type="text/plain")
        except Exception:
            return Response(content="# No metrics available", media_type="text/plain")

    return app
