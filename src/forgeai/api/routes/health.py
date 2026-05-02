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
    """Readiness probe — returns 200 only if the engine is initialized and ready."""
    engine = request.app.state.engine

    if engine is None:
        return Response(
            content='{"status": "not_ready", "reason": "engine not initialized"}',
            status_code=503,
            media_type="application/json",
        )

    if not engine.is_running:
        return Response(
            content='{"status": "not_ready", "reason": "engine not running"}',
            status_code=503,
            media_type="application/json",
        )

    return {"status": "ready", "model": engine.settings.model_name}
