"""
Liveness and readiness probe endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/healthz")
async def liveness():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request):
    """Readiness probe — returns 200 if daemon dependencies are ready."""
    runtime = getattr(request.app.state, "runtime_adapter", None)
    manager = getattr(request.app.state, "engine_manager", None) or (runtime.engine_manager if runtime else None)

    if manager is not None or runtime is not None:
        if manager is not None and getattr(manager, "is_shutting_down", False):
            return Response(
                content='{"status": "not_ready", "reason": "daemon shutting down"}',
                status_code=503,
                media_type="application/json",
            )
        return {"status": "ready"}

    # Legacy engine readiness check
    engine = request.app.state.engine

    if engine is None:
        return Response(
            content='{"status": "not_ready", "reason": "engine not initialized"}',
            status_code=503,
            media_type="application/json",
        )

    if not getattr(engine, "is_running", False):
        return Response(
            content='{"status": "not_ready", "reason": "engine not running"}',
            status_code=503,
            media_type="application/json",
        )

    return {"status": "ready", "model": getattr(getattr(engine, "settings", None), "model_name", "")}
