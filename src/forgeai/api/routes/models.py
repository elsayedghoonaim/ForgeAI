"""
Model listing and management endpoints.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
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
    runtime = getattr(request.app.state, "runtime_adapter", None)
    registry = getattr(request.app.state, "model_registry", None) or (runtime.model_registry if runtime else None)

    models = []
    if registry is not None:
        cache_manager = getattr(request.app.state, "cache_manager", None) or (runtime.cache_manager if runtime else None)
        records = await asyncio.to_thread(
            registry.list_records, cache_manager=cache_manager, include_size=False
        )
        now_ts = int(time.time())
        for r in records:
            models.append(ModelData(id=r.ref.full_tag, created=now_ts))

    engine = getattr(request.app.state, "engine", None)
    if not models and engine is not None and getattr(engine, "is_running", False):
        model_name = getattr(getattr(engine, "settings", None), "model_name", "default-model")
        start_time = getattr(engine, "_start_time", None) or time.time()
        models.append(ModelData(id=model_name, created=int(start_time)))

    return ModelListResponse(data=models)


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> ModelData:
    """Get details of a specific model."""
    runtime = getattr(request.app.state, "runtime_adapter", None)
    registry = getattr(request.app.state, "model_registry", None) or (runtime.model_registry if runtime else None)

    if registry is not None:
        cache_manager = getattr(request.app.state, "cache_manager", None) or (runtime.cache_manager if runtime else None)
        try:
            record = await asyncio.to_thread(
                registry.get_record, model_id, cache_manager=cache_manager
            )
            return ModelData(id=record.ref.full_tag, created=int(time.time()))
        except KeyError:
            pass
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    engine = getattr(request.app.state, "engine", None)
    if engine and getattr(getattr(engine, "settings", None), "model_name", None) == model_id:
        start_time = getattr(engine, "_start_time", None) or time.time()
        return ModelData(id=model_id, created=int(start_time))

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

