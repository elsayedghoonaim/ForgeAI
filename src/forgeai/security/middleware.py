"""Security middleware: authn, authz, rate limiting, and audit logging."""

from __future__ import annotations

from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

PUBLIC_PATHS = {"/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"}


def _required_permission(method: str, path: str) -> str | None:
    if path == "/metrics":
        return "monitoring"
    if method == "POST" and path == "/v1/chat/completions":
        return "inference"
    if method == "GET" and path.startswith("/v1/models"):
        return "models"
    return None


def _request_id(request: Request) -> tuple[str, str]:
    settings = getattr(request.app.state, "settings", None)
    header_name = getattr(settings, "request_id_header", "X-Request-ID")
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = request.headers.get(header_name) or uuid4().hex
        request.state.request_id = request_id
    return request_id, header_name


def _json_error(
    request: Request,
    status_code: int,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id, header_name = _request_id(request)
    merged_headers = {header_name: request_id}
    if headers:
        merged_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        headers=merged_headers,
        content={
            "error": {
                "message": message,
                "status_code": status_code,
                "request_id": request_id,
            }
        },
    )


def _audit(
    request: Request,
    *,
    event_type: str,
    actor: str,
    action: str,
    resource: str,
    outcome: str,
    details: dict[str, object] | None = None,
) -> None:
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger is None:
        return
    audit_logger.log(
        event_type=event_type,
        actor=actor,
        action=action,
        resource=resource,
        outcome=outcome,
        details=details,
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces auth, authorization, and rate limits."""

    async def dispatch(self, request: Request, call_next):
        request_id, header_name = _request_id(request)

        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            response = await call_next(request)
            response.headers.setdefault(header_name, request_id)
            return response

        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        if not auth_header and not api_key:
            _audit(
                request,
                event_type="auth",
                actor="anonymous",
                action="authenticate",
                resource=request.url.path,
                outcome="denied",
                details={"reason": "missing_credentials", "request_id": request_id},
            )
            return _json_error(
                request,
                401,
                "Missing authentication. Provide Authorization header or X-API-Key.",
            )

        auth_manager = getattr(request.app.state, "auth_manager", None)
        if auth_manager is None:
            return _json_error(
                request,
                500,
                "Authentication is enabled but no auth manager is configured.",
            )

        actor_id = "anonymous"
        permissions: set[str] = set()
        role = ""

        if api_key:
            key_info = auth_manager.validate_api_key(api_key)
            if key_info is None:
                _audit(
                    request,
                    event_type="auth",
                    actor="anonymous",
                    action="authenticate",
                    resource=request.url.path,
                    outcome="denied",
                    details={"reason": "invalid_api_key", "request_id": request_id},
                )
                return _json_error(request, 401, "Invalid API key")
            actor_id = key_info.key_id
            role = key_info.role.value
            permissions = auth_manager.permissions_for_role(key_info.role)
        elif auth_header.startswith("Bearer "):
            payload = auth_manager.verify_token(auth_header[7:])
            if payload is None:
                _audit(
                    request,
                    event_type="auth",
                    actor="anonymous",
                    action="authenticate",
                    resource=request.url.path,
                    outcome="denied",
                    details={"reason": "invalid_token", "request_id": request_id},
                )
                return _json_error(request, 401, "Invalid or expired token")
            actor_id = payload.sub
            role = payload.role.value
            permissions = set(payload.permissions)
        else:
            return _json_error(request, 401, "Invalid authentication format")

        request.state.actor_id = actor_id
        request.state.role = role
        request.state.permissions = permissions

        required_permission = _required_permission(request.method, request.url.path)
        request.state.required_permission = required_permission
        if required_permission and required_permission not in permissions:
            _audit(
                request,
                event_type="access",
                actor=actor_id,
                action=request.method,
                resource=request.url.path,
                outcome="denied",
                details={
                    "permission": required_permission,
                    "reason": "missing_permission",
                    "request_id": request_id,
                },
            )
            return _json_error(
                request,
                403,
                f"Permission '{required_permission}' is required for this resource.",
            )

        rate_limiter = getattr(request.app.state, "rate_limiter", None)
        if rate_limiter is not None:
            limit_key = f"{actor_id}:{required_permission or request.url.path}"
            allowed, retry_after = rate_limiter.check(limit_key)
            if not allowed:
                _audit(
                    request,
                    event_type="access",
                    actor=actor_id,
                    action=request.method,
                    resource=request.url.path,
                    outcome="denied",
                    details={
                        "permission": required_permission,
                        "reason": "rate_limited",
                        "request_id": request_id,
                    },
                )
                return _json_error(
                    request,
                    429,
                    "Rate limit exceeded.",
                    headers={"Retry-After": str(retry_after)},
                )

        response = await call_next(request)
        response.headers.setdefault(header_name, request_id)

        _audit(
            request,
            event_type="access",
            actor=actor_id,
            action=request.method,
            resource=request.url.path,
            outcome="success" if response.status_code < 400 else "failure",
            details={
                "permission": required_permission,
                "request_id": request_id,
                "status_code": response.status_code,
            },
        )
        return response
