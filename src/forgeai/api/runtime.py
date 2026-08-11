"""Shared runtime adapter connecting ModelRegistry, CacheManager, and EngineManager."""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from forgeai.core.config import DevToolSettings
from forgeai.core.engine import EngineKey, EngineLease, EngineManager
from forgeai.models.loader import CacheManager
from forgeai.models.manifest import ForgeAIManifest
from forgeai.models.registry import ModelRecord, ModelRegistry, parse_tag


class SharedRuntimeAdapter:
    """
    Shared runtime adapter binding ModelRegistry, CacheManager, and EngineManager.
    Resolves tags, builds deterministic EngineKeys, and manages lease lifecycle.
    """

    def __init__(
        self,
        engine_manager: EngineManager | None = None,
        model_registry: ModelRegistry | None = None,
        cache_manager: CacheManager | None = None,
        settings: DevToolSettings | None = None,
    ) -> None:
        self.settings = settings or DevToolSettings()
        self.engine_manager = engine_manager or EngineManager(settings=self.settings)
        self.model_registry = model_registry or ModelRegistry()
        self.cache_manager = cache_manager or CacheManager()
        self._key_to_tag: dict[EngineKey, str] = {}

    def resolve_tag(self, tag: str | None) -> str:
        """Resolve requested model tag to full tag format (name:tag_version)."""
        if not tag or not tag.strip():
            raise ValueError("Model tag cannot be empty.")
        name, version = parse_tag(tag.strip())
        return f"{name}:{version}"

    def get_manifest_and_record(self, tag: str) -> tuple[ForgeAIManifest, ModelRecord]:
        """Resolve tag to manifest and ModelRecord. Raises KeyError if not found."""
        resolved_tag = self.resolve_tag(tag)
        try:
            record = self.model_registry.get_record(resolved_tag, cache_manager=self.cache_manager)
            manifest = self.model_registry.get_manifest(resolved_tag)
            return manifest, record
        except KeyError:
            if ":" not in tag:
                record = self.model_registry.get_record(f"{tag}:latest", cache_manager=self.cache_manager)
                manifest = self.model_registry.get_manifest(f"{tag}:latest")
                return manifest, record
            raise

    def build_engine_key(self, manifest: ForgeAIManifest, record: ModelRecord) -> EngineKey:
        """Construct deterministic EngineKey from manifest and record."""
        snapshot_path = record.snapshot_path
        repo_id = (
            snapshot_path
            if snapshot_path and record.size_bytes > 0
            else manifest.model
        )

        chat_template_digest = (
            hashlib.sha256(manifest.chat_template.encode("utf-8")).hexdigest()
            if manifest.chat_template
            else ""
        )

        # Snapshot identity combines resolved snapshot path, revision, and manifest digest
        snapshot_identity = hashlib.sha256(
            f"{snapshot_path}:{manifest.revision}:{record.ref.digest}".encode("utf-8")
        ).hexdigest()

        key = EngineKey(
            repo_id=repo_id,
            snapshot_hash=snapshot_identity,
            revision=manifest.revision,
            tokenizer=manifest.tokenizer_override or "",
            tokenizer_revision="",
            chat_template_digest=chat_template_digest,
            tensor_parallel_size=manifest.engine_settings.tensor_parallel_size,
            pipeline_parallel_size=manifest.engine_settings.pipeline_parallel_size,
            max_model_len=self.settings.max_model_len or 8192,
            max_num_seqs=self.settings.max_num_seqs,
            dtype=self.settings.dtype,
            weight_quantization=manifest.engine_settings.weight_quantization,
            kv_cache_dtype=manifest.kv_cache.dtype,
            gpu_memory_utilization=manifest.engine_settings.gpu_memory_utilization,
            enforce_eager=manifest.engine_settings.enforce_eager,
            trust_remote_code=manifest.engine_settings.trust_remote_code,
            device_runtime_id="cuda:0",
        )


        self._key_to_tag[key] = record.ref.full_tag
        return key

    def get_engine_key_for_tag(
        self, tag: str
    ) -> tuple[EngineKey, ForgeAIManifest, ModelRecord]:
        """Resolve public tag and build its exact EngineKey."""
        manifest, record = self.get_manifest_and_record(tag)
        key = self.build_engine_key(manifest, record)
        return key, manifest, record

    def get_public_tag_for_key(self, key: EngineKey) -> str:
        """Return public tag registered for key, or key.repo_id if unmapped."""
        return self._key_to_tag.get(key, key.repo_id)

    async def acquire_lease(
        self, tag: str, keep_alive: Any = None
    ) -> tuple[EngineLease, ForgeAIManifest, ModelRecord, EngineKey]:
        """Pre-acquire an engine lease before returning HTTP responses."""
        key, manifest, record = self.get_engine_key_for_tag(tag)
        lease = await self.engine_manager.acquire(key, keep_alive=keep_alive)
        return lease, manifest, record, key

    async def release_lease(self, key: EngineKey, keep_alive: Any = None) -> None:
        """Release engine lease reference."""
        await self.engine_manager.release(key, keep_alive=keep_alive)

    @asynccontextmanager
    async def acquire(
        self, tag: str, keep_alive: Any = None
    ) -> AsyncIterator[tuple[EngineLease, ForgeAIManifest, ModelRecord, EngineKey]]:
        """Async context manager acquiring and releasing an engine lease."""
        lease, manifest, record, key = await self.acquire_lease(tag, keep_alive=keep_alive)
        try:
            yield lease, manifest, record, key
        finally:
            await self.release_lease(key, keep_alive=keep_alive)
