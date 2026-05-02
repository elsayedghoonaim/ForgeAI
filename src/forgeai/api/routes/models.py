"""
Model listing and management endpoints.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "forgeai"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelData] = []


@router.get("/models")
async def list_models(request: Request) -> ModelListResponse:
    """List available models (OpenAI-compatible)."""
    engine = request.app.state.engine
    models = []

    if engine is not None and engine.is_running:
        models.append(
            ModelData(
                id=engine.settings.model_name,
                created=int(engine._start_time or time.time()),
            )
        )

    return ModelListResponse(data=models)


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request) -> ModelData:
    """Get details of a specific model."""
    engine = request.app.state.engine
    if engine and engine.settings.model_name == model_id:
        return ModelData(
            id=model_id,
            created=int(engine._start_time or time.time()),
        )

    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
