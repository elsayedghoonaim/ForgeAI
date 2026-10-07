"""Security middleware: authn, authz, rate limiting, and audit logging."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PUBLIC_PATHS = {"/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"}
RATE_LIMIT_EXEMPT_PATHS = {"/healthz", "/readyz"}

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sanitize_request_id(value: str | None) -> str:
    """Return ``value`` if it is a safe request ID, otherwise a fresh one."""

    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


def _required_permission(method: str, path: str) -> str:
    """Map paths to required permissions. Defaults to 'admin' (fail-closed)."""
    if path == "/metrics":
        return "monitoring"
    if path in ("/api/generate", "/api/chat", "/api/embed") or (
        method == "POST" and path == "/v1/chat/completions"
    ):
        return "inference"
    if path in ("/api/tags", "/api/show", "/api/ps", "/api/version") or (
        method == "GET" and path.startswith("/v1/models")
    ):
        return "models"
    if path in ("/api/pull", "/api/delete"):
        return "admin"

    # Fail-closed default: require top administrative privileges for any unrecognized path
    return "admin"


def _request_id(request: Request) -> tuple[str, str]:
    settings = getattr(request.app.state, "settings", None)
    header_name = getattr(settings, "request_id_header", "X-Request-ID")
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = sanitize_request_id(request.headers.get(header_name))
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

    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=status_code,
            headers=merged_headers,
            content={"error": message},
        )

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


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class _DeniedAuditThrottle:
    """Per-IP limit on denied-auth audit events so anonymous traffic cannot flood the log."""

    def __init__(self, max_events: int = 10, window_seconds: float = 60.0) -> None:
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._last_sweep = time.monotonic()

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        if now - self._last_sweep >= self.window_seconds:
            self._last_sweep = now
            for key in [
                k for k, v in self._events.items() if not v or now - v[-1] >= self.window_seconds
            ]:
                del self._events[key]
        events = self._events.setdefault(ip, deque())
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.max_events:
            return False
        events.append(now)
        return True


class AuthMiddleware:
    """Pure-ASGI middleware that enforces authentication and authorization.

    Responses are streamed straight through (no body buffering, no extra task), and the
    access audit event is written once the response status is known.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._denied_throttle = _DeniedAuditThrottle()

    def _audit_denied_auth(self, request: Request, reason: str, request_id: str) -> None:
        ip = _client_ip(request)
        if not self._denied_throttle.allow(ip):
            return
        _audit(
            request,
            event_type="auth",
            actor="anonymous",
            action="authenticate",
            resource=request.url.path,
            outcome="denied",
            details={"reason": reason, "request_id": request_id, "client_ip": ip},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id, _header_name = _request_id(request)

        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        denial = self._authenticate(request, request_id)
        if denial is not None:
            await denial(scope, receive, send)
            return

        actor_id = request.state.actor_id
        required_permission = request.state.required_permission

        async def audit_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                _audit(
                    request,
                    event_type="access",
                    actor=actor_id,
                    action=request.method,
                    resource=request.url.path,
                    outcome="success" if status_code < 400 else "failure",
                    details={
                        "permission": required_permission,
                        "request_id": request_id,
                        "status_code": status_code,
                    },
                )
            await send(message)

        await self.app(scope, receive, audit_send)

    def _authenticate(self, request: Request, request_id: str) -> JSONResponse | None:
        """Return an error response if the request is not allowed, else None."""
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        if not auth_header and not api_key:
            self._audit_denied_auth(request, "missing_credentials", request_id)
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
                self._audit_denied_auth(request, "invalid_api_key", request_id)
                return _json_error(request, 401, "Invalid API key")
            actor_id = key_info.key_id
            role = key_info.role.value
            permissions = auth_manager.permissions_for_role(key_info.role)
        elif auth_header.startswith("Bearer "):
            payload = auth_manager.verify_token(auth_header[7:])
            if payload is None:
                self._audit_denied_auth(request, "invalid_token", request_id)
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
        return None


class RateLimitMiddleware:
    """Pure-ASGI rate limiter that works with or without authentication.

    Two instances are installed: ``mode="ip"`` sits outside auth and keys by client IP
    (so failed and anonymous requests count), ``mode="actor"`` sits inside auth and
    keys by the authenticated actor and required permission.
    """

    def __init__(self, app: ASGIApp, mode: str = "ip") -> None:
        if mode not in ("ip", "actor"):
            raise ValueError("mode must be 'ip' or 'actor'")
        self.app = app
        self.mode = mode

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        limiter = getattr(getattr(app, "state", None), "rate_limiter", None)
        path = scope.get("path", "")
        if limiter is None or path in RATE_LIMIT_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        if self.mode == "ip":
            client = scope.get("client")
            key = f"ip:{client[0] if client else 'unknown'}"
        else:
            state = scope.get("state") or {}
            actor = state.get("actor_id")
            if not actor:
                await self.app(scope, receive, send)
                return
            key = f"{actor}:{state.get('required_permission') or path}"

        allowed, retry_after = limiter.check(key)
        if allowed:
            await self.app(scope, receive, send)
            return

        settings = getattr(getattr(app, "state", None), "settings", None)
        header_name = getattr(settings, "request_id_header", "X-Request-ID")
        request_id = (scope.get("state") or {}).get("request_id")
        if not request_id:
            raw_id = None
            for name, value in scope.get("headers", []):
                if name.decode("latin-1").lower() == header_name.lower():
                    raw_id = value.decode("latin-1")
                    break
            request_id = sanitize_request_id(raw_id)
        if path.startswith("/api/"):
            payload: dict[str, object] = {"error": "Rate limit exceeded."}
        else:
            payload = {
                "error": {
                    "message": "Rate limit exceeded.",
                    "status_code": 429,
                    "request_id": request_id,
                }
            }
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(retry_after).encode()),
                    (header_name.lower().encode("latin-1"), request_id.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
