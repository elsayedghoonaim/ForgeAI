"""Ollama-compatible API endpoints for ForgeAI."""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import aclosing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from forgeai import __version__
from forgeai.api.schemas.ollama import (
    OllamaChatRequest,
    OllamaDeleteRequest,
    OllamaEmbedRequest,
    OllamaGenerateRequest,
    OllamaOptions,
    OllamaPullRequest,
    OllamaShowRequest,
    OllamaUnloadRequest,
)
from forgeai.api.streaming import guarded_stream_response
from forgeai.core.backends.base import StreamStats, abort_request, iter_stream
from forgeai.core.engine import is_oom_exception
from forgeai.models.loader import SecureCacheManager, SecurityBlockError
from forgeai.models.manifest import ForgeAIManifest, validate_repo_id_string

router = APIRouter()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _get_runtime_adapter(request: Request) -> Any:
    adapter = getattr(request.app.state, "runtime_adapter", None)
    if adapter is None:
        raise HTTPException(status_code=503, detail="Daemon runtime adapter not initialized.")
    return adapter


def _parse_options(options_raw: Any) -> tuple[float, float, int, int, list[str] | None]:
    """Parse Ollama options into (temperature, top_p, top_k, max_tokens, stop)."""
    if options_raw is None:
        return 0.7, 0.95, 40, 512, None

    if isinstance(options_raw, dict):
        opts = OllamaOptions.model_validate(options_raw)
    elif isinstance(options_raw, OllamaOptions):
        opts = options_raw
    else:
        raise ValueError("Invalid options payload type.")

    temp = opts.temperature if opts.temperature is not None else 0.7
    top_p = opts.top_p if opts.top_p is not None else 0.95
    top_k = opts.top_k if opts.top_k is not None else 40
    max_tokens = opts.num_predict if opts.num_predict is not None else 512

    stop = None
    if isinstance(opts.stop, str):
        stop = [opts.stop]
    elif isinstance(opts.stop, list):
        stop = [str(s) for s in opts.stop]

    return temp, top_p, top_k, max_tokens, stop


def _ndjson_stream_response(
    runtime: Any,
    key: Any,
    engine: Any,
    prompt: str,
    *,
    keep_alive: Any,
    params: dict[str, Any],
    make_chunk: Any,
    make_terminal: Any,
    acquire_start: float,
) -> StreamingResponse:
    """NDJSON streaming response with guaranteed lease release, close and abort."""
    stats = StreamStats()

    async def _events() -> Any:
        eval_start = time.perf_counter()
        source = iter_stream(engine, prompt, stats, **params)
        async with aclosing(source) as chunks:  # type: ignore[type-var]
            async for chunk in chunks:
                yield json.dumps(make_chunk(chunk)) + "\n"
        now = time.perf_counter()
        total_duration_ns = int((now - acquire_start) * 1e9)
        eval_duration_ns = int((now - eval_start) * 1e9)
        yield json.dumps(make_terminal(stats, total_duration_ns, eval_duration_ns)) + "\n"

    async def _on_error(err: Exception) -> list[str]:
        if is_oom_exception(err):
            with suppress(Exception):
                await runtime.engine_manager.mark_engine_failed(key, str(err))
        return [json.dumps({"error": str(err)}) + "\n"]

    return guarded_stream_response(
        _events(),
        release=lambda: runtime.release_lease(key, keep_alive=keep_alive),
        abort=lambda: abort_request(engine, stats.request_id),
        on_error=_on_error,
        media_type="application/x-ndjson",
    )


def _is_zero_keep_alive(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, float)) and val == 0:
        return True
    return bool(isinstance(val, str) and val.strip().lower() in ("0", "0.0", "0s", "0m", "0h"))


def _model_error(model: str, err: Exception) -> JSONResponse:
    """Map lease/resolution errors: unknown model -> 404, malformed tag -> 400."""
    if isinstance(err, KeyError):
        return JSONResponse(status_code=404, content={"error": f"model '{model}' not found"})
    return JSONResponse(status_code=400, content={"error": str(err)})


def _count_tokens(engine: Any, texts: list[str]) -> int:
    """Token count for ``texts`` using the engine's tokenizer when one is reachable.

    Falls back to a whitespace word count (a documented underestimate) when the engine
    exposes no tokenizer or tokenization fails.
    """
    counter = getattr(engine, "count_tokens", None)
    if callable(counter):
        with suppress(Exception):
            return int(sum(counter(t) for t in texts))
    for holder in (engine, getattr(engine, "backend", None), getattr(engine, "_backend", None)):
        tok = getattr(holder, "_tokenizer", None) or getattr(holder, "tokenizer", None)
        encode = getattr(tok, "encode", None)
        if callable(encode):
            with suppress(Exception):
                return sum(len(encode(t)) for t in texts)
    return sum(len(t.split()) for t in texts)


async def _unload_model(runtime: Any, model: str) -> tuple[bool, JSONResponse | None]:
    """Unload ``model`` without ever loading it. Returns (was_loaded, error_response)."""
    try:
        key, _manifest, _record = await asyncio.to_thread(runtime.get_engine_key_for_tag, model)
    except (KeyError, ValueError) as err:
        return False, _model_error(model, err)
    statuses = await runtime.engine_manager.list_async()
    was_loaded = any(st.key == key for st in statuses)
    await runtime.engine_manager.stop(key)
    return was_loaded, None


@router.post("/unload")
async def unload(request: Request, body: OllamaUnloadRequest) -> Response:
    """Unload a running model engine. Reports whether it was actually loaded."""
    runtime = _get_runtime_adapter(request)
    try:
        model = body.model_name
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})
    was_loaded, error = await _unload_model(runtime, model)
    if error is not None:
        return error
    return JSONResponse(content={"model": model, "unloaded": was_loaded})


@router.post("/generate")
async def generate(request: Request, body: OllamaGenerateRequest) -> Response:
    """Ollama-compatible /api/generate endpoint."""
    runtime = _get_runtime_adapter(request)

    # keep_alive=0 with an empty prompt is an unload request, whatever ``stream`` says;
    # it must never load the model.
    if not body.prompt and _is_zero_keep_alive(body.keep_alive):
        _, error = await _unload_model(runtime, body.model)
        if error is not None:
            return error
        return JSONResponse(
            content={
                "model": body.model,
                "created_at": _iso_now(),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "context": body.context or [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
        )

    if body.images and len(body.images) > 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Multimodal image inputs are deferred in this runtime version."},
        )
    if body.suffix is not None:
        return JSONResponse(
            status_code=400,
            content={"error": "Suffix parameter is unsupported by active vLLM chat template engine."},
        )
    if body.template is not None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Per-request custom Jinja template override is unsupported. Use local manifest chat_template."
            },
        )

    acquire_start = time.perf_counter()
    acquired = False
    try:
        lease, manifest, record, key = await runtime.acquire_lease(
            body.model, keep_alive=body.keep_alive
        )
    except (KeyError, ValueError) as err:
        return _model_error(body.model, err)

    acquired = True
    try:
        load_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)


        if body.raw:
            prompt = body.prompt
        else:
            system_content = body.system or manifest.system_prompt
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": body.prompt})
            prompt = lease.engine.build_prompt(messages)

        temp, top_p, top_k, max_tokens, stop = _parse_options(body.options)

        if body.stream:
            def _chunk(text: str) -> dict[str, Any]:
                return {
                    "model": body.model,
                    "created_at": _iso_now(),
                    "response": text,
                    "done": False,
                }

            def _terminal(stats: StreamStats, total_ns: int, eval_ns: int) -> dict[str, Any]:
                return {
                    "model": body.model,
                    "created_at": _iso_now(),
                    "response": "",
                    "done": True,
                    "done_reason": stats.finish_reason or "stop",
                    "context": body.context or [],
                    "total_duration": total_ns,
                    "load_duration": load_duration_ns,
                    "prompt_eval_count": stats.prompt_tokens,
                    "prompt_eval_duration": max(0, total_ns - eval_ns),
                    "eval_count": stats.completion_tokens,
                    "eval_duration": eval_ns,
                }

            response = _ndjson_stream_response(
                runtime,
                key,
                lease.engine,
                prompt,
                keep_alive=body.keep_alive,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "top_p": top_p,
                    "stop": stop,
                    "top_k": top_k,
                },
                make_chunk=_chunk,
                make_terminal=_terminal,
                acquire_start=acquire_start,
            )
            acquired = False  # Ownership transferred to the guarded response
            return response

        gen_start = time.perf_counter()
        result = await lease.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            stop=stop,
            top_k=top_k,
        )
        eval_duration_ns = int((time.perf_counter() - gen_start) * 1e9)
        total_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)

        return JSONResponse(
            content={
                "model": body.model,
                "created_at": _iso_now(),
                "response": getattr(result, "text", ""),
                "done": True,
                "done_reason": getattr(result, "finish_reason", "stop") or "stop",
                "context": body.context or [],
                "total_duration": total_duration_ns,
                "load_duration": load_duration_ns,
                "prompt_eval_count": getattr(result, "prompt_tokens", 0),
                "prompt_eval_duration": max(0, total_duration_ns - eval_duration_ns),
                "eval_count": getattr(result, "completion_tokens", 0),
                "eval_duration": eval_duration_ns,
            }
        )
    finally:
        if acquired:
            await runtime.release_lease(key, keep_alive=body.keep_alive)


@router.post("/chat")
async def chat(request: Request, body: OllamaChatRequest) -> Response:
    """Ollama-compatible /api/chat endpoint."""
    runtime = _get_runtime_adapter(request)

    if not body.messages and _is_zero_keep_alive(body.keep_alive):
        _, error = await _unload_model(runtime, body.model)
        if error is not None:
            return error
        return JSONResponse(
            content={
                "model": body.model,
                "created_at": _iso_now(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }
        )

    for msg in body.messages:
        if msg.images and len(msg.images) > 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Multimodal image inputs are deferred in this runtime version."},
            )

    acquire_start = time.perf_counter()
    acquired = False
    try:
        lease, manifest, record, key = await runtime.acquire_lease(
            body.model, keep_alive=body.keep_alive
        )
    except (KeyError, ValueError) as err:
        return _model_error(body.model, err)

    acquired = True
    try:
        load_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)


        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system and manifest.system_prompt:
            messages.insert(0, {"role": "system", "content": manifest.system_prompt})

        prompt = lease.engine.build_prompt(messages)
        temp, top_p, top_k, max_tokens, stop = _parse_options(body.options)

        if body.stream:
            def _chat_chunk(text: str) -> dict[str, Any]:
                return {
                    "model": body.model,
                    "created_at": _iso_now(),
                    "message": {"role": "assistant", "content": text},
                    "done": False,
                }

            def _chat_terminal(stats: StreamStats, total_ns: int, eval_ns: int) -> dict[str, Any]:
                return {
                    "model": body.model,
                    "created_at": _iso_now(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": stats.finish_reason or "stop",
                    "total_duration": total_ns,
                    "load_duration": load_duration_ns,
                    "prompt_eval_count": stats.prompt_tokens,
                    "prompt_eval_duration": max(0, total_ns - eval_ns),
                    "eval_count": stats.completion_tokens,
                    "eval_duration": eval_ns,
                }

            response = _ndjson_stream_response(
                runtime,
                key,
                lease.engine,
                prompt,
                keep_alive=body.keep_alive,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "top_p": top_p,
                    "stop": stop,
                    "top_k": top_k,
                },
                make_chunk=_chat_chunk,
                make_terminal=_chat_terminal,
                acquire_start=acquire_start,
            )
            acquired = False  # Ownership transferred to the guarded response
            return response

        gen_start = time.perf_counter()
        result = await lease.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            stop=stop,
            top_k=top_k,
        )
        eval_duration_ns = int((time.perf_counter() - gen_start) * 1e9)
        total_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)

        return JSONResponse(
            content={
                "model": body.model,
                "created_at": _iso_now(),
                "message": {
                    "role": "assistant",
                    "content": getattr(result, "text", ""),
                },
                "done": True,
                "done_reason": getattr(result, "finish_reason", "stop") or "stop",
                "total_duration": total_duration_ns,
                "load_duration": load_duration_ns,
                "prompt_eval_count": getattr(result, "prompt_tokens", 0),
                "prompt_eval_duration": max(0, total_duration_ns - eval_duration_ns),
                "eval_count": getattr(result, "completion_tokens", 0),
                "eval_duration": eval_duration_ns,
            }
        )
    finally:
        if acquired:
            await runtime.release_lease(key, keep_alive=body.keep_alive)


@router.post("/embed")
async def embed(request: Request, body: OllamaEmbedRequest) -> Response:
    """Ollama-compatible /api/embed endpoint."""
    runtime = _get_runtime_adapter(request)

    if body.dimensions is not None:
        return JSONResponse(
            status_code=400,
            content={"error": "Custom embedding dimensions parameter is unsupported."},
        )
    if body.truncate is not None:
        return JSONResponse(
            status_code=400,
            content={"error": "Embedding truncate parameter is unsupported."},
        )
    if body.options is not None and isinstance(body.options, dict) and len(body.options) > 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Embedding options parameter is unsupported."},
        )

    acquire_start = time.perf_counter()
    acquired = False
    try:
        lease, manifest, record, key = await runtime.acquire_lease(
            body.model, keep_alive=body.keep_alive
        )
    except (KeyError, ValueError) as err:
        return _model_error(body.model, err)

    acquired = True
    try:
        load_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)
        input_texts = [body.input] if isinstance(body.input, str) else body.input

        try:
            embeddings = await lease.engine.embed(input_texts)
        except NotImplementedError as err:
            return JSONResponse(
                status_code=501,
                content={"error": f"Model '{body.model}' does not support embeddings/pooling: {err}"},
            )
        except Exception as err:
            return JSONResponse(
                status_code=500,
                content={"error": f"Embedding generation failed: {err}"},
            )

        total_duration_ns = int((time.perf_counter() - acquire_start) * 1e9)
        prompt_tokens = _count_tokens(lease.engine, input_texts)

        return JSONResponse(
            content={
                "model": body.model,
                "embeddings": embeddings,
                "total_duration": total_duration_ns,
                "load_duration": load_duration_ns,
                "prompt_eval_count": prompt_tokens,
            }
        )
    finally:
        if acquired:
            await runtime.release_lease(key, keep_alive=body.keep_alive)


@router.get("/tags")
async def list_tags(request: Request) -> Response:
    """Ollama-compatible /api/tags endpoint."""
    runtime = _get_runtime_adapter(request)
    records = await asyncio.to_thread(
        runtime.model_registry.list_records, cache_manager=runtime.cache_manager, include_size=True
    )

    models_list = []
    for record in records:
        try:
            manifest = record.manifest
            if manifest is None:
                raise ValueError("record has no manifest")
            weight_quant = manifest.engine_settings.weight_quantization
            kv_dtype = manifest.kv_cache.dtype
        except Exception:
            weight_quant = "none"
            kv_dtype = "auto"

        try:
            mtime = Path(record.manifest_path).stat().st_mtime
            modified_at_str = datetime.fromtimestamp(mtime, UTC).isoformat()
        except Exception:
            modified_at_str = _iso_now()

        models_list.append(
            {
                "name": record.ref.full_tag,
                "model": record.ref.full_tag,
                "modified_at": modified_at_str,
                "size": record.size_bytes,
                "digest": record.ref.digest,
                "details": {
                    "format": "safetensors",
                    "family": "vLLM",
                    "families": ["vLLM"],
                    "parameter_size": "",
                    "quantization_level": weight_quant,
                    "weight_quantization": weight_quant,
                    "kv_cache_dtype": kv_dtype,
                },
            }
        )

    return JSONResponse(content={"models": models_list})


@router.get("/ps")
async def list_running(request: Request) -> Response:
    """Ollama-compatible /api/ps endpoint."""
    runtime = _get_runtime_adapter(request)
    statuses = await runtime.engine_manager.list_async()

    running_models = []
    now = datetime.now(UTC)
    for st in statuses:
        ttl = st.remaining_keep_alive_seconds
        if ttl is None or st.ref_count > 0 or math.isinf(ttl):
            expires_at_str = "2100-01-01T00:00:00Z"
        else:
            expires_at_str = (now + timedelta(seconds=max(0.0, ttl))).isoformat()

        public_tag = runtime.get_public_tag_for_key(st.key)
        try:
            record = await asyncio.to_thread(
                runtime.model_registry.get_record,
                public_tag,
                cache_manager=runtime.cache_manager,
                include_size=True,
            )
            size = record.size_bytes
            digest = record.ref.digest
        except Exception:
            size = 0
            digest = st.key.snapshot_hash or ""

        running_models.append(
            {
                "name": public_tag,
                "model": public_tag,
                "size": size,
                "size_vram": 0,  # vLLM does not report per-model VRAM; clients treat 0 as unknown
                "digest": digest,
                "details": {
                    "format": "safetensors",
                    "family": "vLLM",
                    "families": ["vLLM"],
                    "parameter_size": "",
                    "quantization_level": st.key.weight_quantization,
                    "weight_quantization": st.key.weight_quantization,
                    "kv_cache_dtype": st.key.kv_cache_dtype,
                },
                "expires_at": expires_at_str,
            }
        )

    return JSONResponse(content={"models": running_models})


@router.post("/show")
async def show_model(request: Request, body: OllamaShowRequest) -> Response:
    """Ollama-compatible /api/show endpoint."""
    runtime = _get_runtime_adapter(request)
    try:
        model_name = body.model_name
        manifest, record = await asyncio.to_thread(runtime.get_manifest_and_record, model_name)
    except (ValueError, KeyError):
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{body.model or body.name}' not found"},
        )

    return JSONResponse(
        content={
            "license": "",
            "modelfile": (
                "# ForgeAI Local Manifest Replacement (YAML)\n"
                "# Native Ollama Modelfile directives are replaced by canonical ForgeAI manifest\n"
                f"{manifest.to_yaml()}"
            ),
            "forgeai_manifest": manifest.model_dump(mode="json"),
            "parameters": (
                f"temperature {manifest.parameters.temperature}\n"
                f"top_p {manifest.parameters.top_p}\n"
                f"top_k {manifest.parameters.top_k}\n"
                f"max_tokens {manifest.parameters.max_tokens}"
            ),
            "template": manifest.chat_template,
            "system": manifest.system_prompt or "",
            "details": {
                "format": "safetensors",
                "family": "vLLM",
                "families": ["vLLM"],
                "parameter_size": "",
                "quantization_level": manifest.engine_settings.weight_quantization,
                "weight_quantization": manifest.engine_settings.weight_quantization,
                "kv_cache_dtype": manifest.kv_cache.dtype,
            },
            "model_info": {
                "general.architecture": "vLLM",
                "forgeai.source_kind": manifest.source_kind,
                "forgeai.model": manifest.model,
                "forgeai.revision": manifest.revision,
                "forgeai.tensor_parallel_size": manifest.engine_settings.tensor_parallel_size,
                "forgeai.pipeline_parallel_size": manifest.engine_settings.pipeline_parallel_size,
                "forgeai.weight_quantization": manifest.engine_settings.weight_quantization,
                "forgeai.kv_cache_dtype": manifest.kv_cache.dtype,
            },
        }
    )


@router.post("/pull")
async def pull_model(request: Request, body: OllamaPullRequest) -> Response:
    """Ollama-compatible /api/pull endpoint."""
    runtime = _get_runtime_adapter(request)
    try:
        raw_model = body.model_name
        if ":" in raw_model:
            parts = raw_model.split(":", 1)
            repo_id = parts[0]
            public_tag = raw_model
        else:
            repo_id = raw_model
            public_tag = f"{raw_model}:latest"

        validate_repo_id_string(repo_id)
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})

    # Save existing manifest if present to preserve atomic state on registration failure
    existing_manifest = None
    with suppress(Exception):
        existing_manifest = await asyncio.to_thread(
            runtime.model_registry.get_manifest, public_tag
        )

    trust_remote_code = bool(
        body.trust_remote_code
        or (existing_manifest is not None and existing_manifest.engine_settings.trust_remote_code)
    )

    async def _download_and_scan() -> str:
        """Download the snapshot, then scan it; a blocked snapshot is purged and raises."""
        path = await asyncio.to_thread(
            runtime.cache_manager.download_snapshot,
            repo_id=repo_id,
            revision="main",
            token=None,
            trust_remote_code=trust_remote_code,
        )
        # SecureCacheManager.download_snapshot already validated and scanned (and purged on
        # block); any other cache manager still needs the explicit scan here.
        if not isinstance(runtime.cache_manager, SecureCacheManager):
            await asyncio.to_thread(runtime.cache_manager.scan_snapshot_or_purge, path, repo_id)
        return str(path)

    def _new_manifest() -> ForgeAIManifest:
        manifest = ForgeAIManifest(name=public_tag, model=repo_id, source_kind="huggingface")
        if trust_remote_code:
            manifest.engine_settings.trust_remote_code = True
        return manifest

    if body.stream:
        async def _stream_pull():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            yield json.dumps({"status": "downloading weights"}) + "\n"

            try:
                await _download_and_scan()
            except Exception as err:
                yield json.dumps({"error": f"Pull failed: {err}"}) + "\n"
                return

            yield json.dumps({"status": "writing manifest"}) + "\n"


            try:
                manifest = _new_manifest()
                record = await asyncio.to_thread(
                    runtime.model_registry.register_manifest,
                    manifest,
                    cache_manager=runtime.cache_manager,
                )
                yield json.dumps({"status": "success", "digest": record.ref.digest}) + "\n"
            except Exception as err:
                if existing_manifest is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(
                            runtime.model_registry.register_manifest,
                            existing_manifest,
                            cache_manager=runtime.cache_manager,
                        )
                yield json.dumps({"error": f"Manifest registration failed: {err}"}) + "\n"

        return StreamingResponse(_stream_pull(), media_type="application/x-ndjson")

    try:
        await _download_and_scan()
        manifest = _new_manifest()
        record = await asyncio.to_thread(
            runtime.model_registry.register_manifest,
            manifest,
            cache_manager=runtime.cache_manager,
        )
        return JSONResponse(content={"status": "success", "digest": record.ref.digest})
    except Exception as err:
        if existing_manifest is not None:
            with suppress(Exception):
                await asyncio.to_thread(
                    runtime.model_registry.register_manifest,
                    existing_manifest,
                    cache_manager=runtime.cache_manager,
                )
        status = 403 if isinstance(err, SecurityBlockError) else 500
        return JSONResponse(status_code=status, content={"error": f"Pull failed: {err}"})


@router.delete("/delete")
async def delete_model(request: Request, body: OllamaDeleteRequest) -> Response:
    """Ollama-compatible /api/delete endpoint."""
    runtime = _get_runtime_adapter(request)
    try:
        model_name = body.model_name
        key, manifest, record = await asyncio.to_thread(
            runtime.get_engine_key_for_tag, model_name
        )
    except (ValueError, KeyError):
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{body.model or body.name}' not found"},
        )

    # Call stop() with the exact EngineKey built for that record, then unregister after stop completes
    await runtime.engine_manager.stop(key)
    await asyncio.to_thread(
        runtime.model_registry.unregister_tag,
        record.ref.full_tag,
        cache_manager=runtime.cache_manager,
    )

    return JSONResponse(content={})


@router.get("/version")
async def version() -> Response:
    """Ollama-compatible /api/version endpoint."""
    return JSONResponse(content={"version": f"{__version__}-forgeai"})
