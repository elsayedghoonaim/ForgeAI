"""
Unit and integration tests for ForgeAI typed local model manifest,
manifest registry, and canonical HuggingFace CacheManager.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from forgeai.models.loader import CacheManager, download_model
from forgeai.models.manifest import (
    EngineSettings,
    ForgeAIManifest,
    GenerationDefaults,
    KVCacheSettings,
    validate_repo_id_string,
)
from forgeai.models.registry import (
    ModelRecord,
    ModelRegistry,
    format_tag,
    parse_tag,
    validate_tag_string,
)


class TestManifestSchemas:
    """Tests for typed Pydantic models in src/forgeai/models/manifest.py."""

    def test_generation_defaults_defaults(self) -> None:
        defaults = GenerationDefaults()
        assert defaults.temperature == 0.7
        assert defaults.top_p == 0.95
        assert defaults.top_k == 40
        assert defaults.max_tokens == 4096
        assert defaults.stop == []

    def test_engine_settings_defaults(self) -> None:
        engine = EngineSettings()
        assert engine.tensor_parallel_size == 1
        assert engine.pipeline_parallel_size == 1
        assert engine.gpu_memory_utilization == 0.85
        assert engine.enforce_eager is False
        assert engine.weight_quantization == "none"
        assert engine.trust_remote_code is False

    def test_kv_cache_settings_dtypes(self) -> None:
        kv = KVCacheSettings(dtype="turboquant_4bit_nc")
        assert kv.dtype == "turboquant_4bit_nc"

        kv_fp8 = KVCacheSettings(dtype="fp8")
        assert kv_fp8.dtype == "fp8"

        with pytest.raises(ValidationError):
            KVCacheSettings(dtype="invalid_dtype")  # type: ignore[arg-type]

    def test_manifest_orthogonality(self) -> None:
        manifest = ForgeAIManifest(
            name="gemma-2-9b-it:latest",
            model="google/gemma-2-9b-it",
            engine_settings=EngineSettings(weight_quantization="awq"),
            kv_cache=KVCacheSettings(dtype="turboquant_4bit_nc"),
        )
        assert manifest.engine_settings.weight_quantization == "awq"
        assert manifest.kv_cache.dtype == "turboquant_4bit_nc"

    def test_manifest_rejects_gguf(self) -> None:
        with pytest.raises(ValueError, match="GGUF model format is unsupported"):
            ForgeAIManifest(
                name="test-gguf",
                model="models/llama-3-8b.gguf",
            )

    def test_manifest_rejects_bin_pt(self) -> None:
        with pytest.raises(ValueError, match="PyTorch pickled weights"):
            ForgeAIManifest(
                name="test-bin",
                model="meta-llama/Llama-2-7b/pytorch_model.bin",
            )

        with pytest.raises(ValueError, match="PyTorch pickled weights"):
            ForgeAIManifest(
                name="test-pt",
                model="model.pt",
            )

    def test_manifest_local_dir_validation(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "my_local_model"
        local_dir.mkdir()

        with pytest.raises(ValueError, match="missing required config.json"):
            ForgeAIManifest(
                source_kind="local_dir",
                name="local-model:latest",
                model=str(local_dir),
            )

        (local_dir / "config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="missing .safetensors file"):
            ForgeAIManifest(
                source_kind="local_dir",
                name="local-model:latest",
                model=str(local_dir),
            )

        (local_dir / "model.safetensors").write_text("fake weights", encoding="utf-8")

        manifest = ForgeAIManifest(
            source_kind="local_dir",
            name="local-model:latest",
            model=str(local_dir),
        )
        assert manifest.source_kind == "local_dir"
        assert manifest.model == str(local_dir)

    def test_deterministic_yaml_serialization(self) -> None:
        manifest = ForgeAIManifest(
            name="gemma-2-9b-it:latest",
            model="google/gemma-2-9b-it",
        )
        yaml_str1 = manifest.to_yaml()
        yaml_str2 = manifest.to_yaml()
        assert yaml_str1 == yaml_str2
        assert "gemma-2-9b-it" in yaml_str1

        deserialized = ForgeAIManifest.from_yaml(yaml_str1)
        assert deserialized.name == manifest.name
        assert deserialized.model == manifest.model

    def test_hf_repo_id_validation_traversal_absolute(self) -> None:
        with pytest.raises(ValueError, match="Invalid Hugging Face repository ID"):
            validate_repo_id_string("../etc/passwd")

        with pytest.raises(ValueError, match="Invalid Hugging Face repository ID"):
            validate_repo_id_string("/absolute/path/repo")

        with pytest.raises(ValueError, match="Invalid Hugging Face repository ID"):
            validate_repo_id_string("C:\\Windows\\System32")

        with pytest.raises(ValueError, match="Invalid Hugging Face repository ID"):
            validate_repo_id_string("owner/repo/extra_component")


class TestModelRegistry:
    """Tests for ModelRegistry catalog and atomic publishing."""

    def test_parse_and_format_tag(self) -> None:
        name, tag = parse_tag("gemma-2-9b-it:v1")
        assert name == "gemma-2-9b-it"
        assert tag == "v1"

        name2, tag2 = parse_tag("gemma-2-9b-it")
        assert name2 == "gemma-2-9b-it"
        assert tag2 == "latest"

        assert format_tag("gemma-2-9b-it", "v1") == "gemma-2-9b-it:v1"
        assert format_tag("gemma-2-9b-it", "") == "gemma-2-9b-it:latest"

    def test_manifest_filename_collisions_and_namespaces(self, tmp_path: Path) -> None:
        registry = ModelRegistry(manifests_dir=tmp_path / "manifests")

        m_colon = ForgeAIManifest(name="foo:bar", model="org/model1")
        m_underscore = ForgeAIManifest(name="foo_bar", model="org/model2")
        m_namespace = ForgeAIManifest(name="team/foo:bar", model="org/model3")

        r1 = registry.register_manifest(m_colon)
        r2 = registry.register_manifest(m_underscore)
        r3 = registry.register_manifest(m_namespace)

        assert r1.manifest_path != r2.manifest_path
        assert r1.manifest_path != r3.manifest_path
        assert r2.manifest_path != r3.manifest_path

        p1 = Path(r1.manifest_path)
        p2 = Path(r2.manifest_path)
        p3 = Path(r3.manifest_path)

        assert p1.is_relative_to(registry.manifests_dir)
        assert p2.is_relative_to(registry.manifests_dir)
        assert p3.is_relative_to(registry.manifests_dir)

        assert registry.get_manifest("foo:bar").model == "org/model1"
        assert registry.get_manifest("foo_bar").model == "org/model2"
        assert registry.get_manifest("team/foo:bar").model == "org/model3"

        with pytest.raises(ValueError, match="Invalid tag namespace structure"):
            validate_tag_string("team//model:tag")

        with pytest.raises(ValueError, match="Invalid tag namespace structure"):
            validate_tag_string("team/sub/model:tag")

        with pytest.raises(ValueError, match="Invalid characters in tag"):
            validate_tag_string("team/model:tag$")

        with pytest.raises(ValueError, match="Invalid characters in tag"):
            validate_tag_string("invalid@name:tag")

    def test_registry_register_and_get(self, tmp_path: Path) -> None:
        manifests_dir = tmp_path / "manifests"
        registry = ModelRegistry(manifests_dir=manifests_dir)

        manifest = ForgeAIManifest(
            name="test-model:latest",
            model="meta-llama/Llama-3-8B-Instruct",
        )

        record = registry.register_manifest(manifest)
        assert isinstance(record, ModelRecord)
        assert record.ref.name == "test-model"
        assert record.ref.tag == "latest"
        assert Path(record.manifest_path).exists()

        retrieved = registry.get_manifest("test-model:latest")
        assert retrieved.name == "test-model:latest"
        assert retrieved.model == "meta-llama/Llama-3-8B-Instruct"

    def test_registry_atomic_write_no_corruption(self, tmp_path: Path) -> None:
        registry = ModelRegistry(manifests_dir=tmp_path / "manifests")
        manifest = ForgeAIManifest(
            name="atomic-test:latest",
            model="org/atomic-repo",
        )
        registry.register_manifest(manifest)

        manifest_path = Path(registry.get_record("atomic-test:latest").manifest_path)
        original_content = manifest_path.read_text(encoding="utf-8")

        with (
            patch("os.replace", side_effect=OSError("Atomic replace error")),
            pytest.raises(OSError, match="Atomic replace error"),
        ):
            registry.register_manifest(manifest)

        assert manifest_path.read_text(encoding="utf-8") == original_content

        tmp_dir = registry.manifests_dir.parent / "tmp"
        if tmp_dir.exists():
            tmp_files = list(tmp_dir.glob("*.tmp"))
            assert len(tmp_files) == 0

    def test_registry_list_and_unregister(self, tmp_path: Path) -> None:
        registry = ModelRegistry(manifests_dir=tmp_path / "manifests")

        m1 = ForgeAIManifest(name="model1:latest", model="org/model1")
        m2 = ForgeAIManifest(name="model2:v1", model="org/model2")

        registry.register_manifest(m1)
        registry.register_manifest(m2)

        records = registry.list_records()
        assert len(records) == 2
        names = {r.ref.name for r in records}
        assert names == {"model1", "model2"}

        unregistered = registry.unregister_tag("model1:latest")
        assert unregistered is not None
        assert unregistered.ref.name == "model1"

        records_after = registry.list_records()
        assert len(records_after) == 1
        assert records_after[0].ref.name == "model2"

    def test_registry_path_traversal_rejection(self, tmp_path: Path) -> None:
        registry = ModelRegistry(manifests_dir=tmp_path / "manifests")

        with pytest.raises(ValueError, match="path traversal"):
            registry.get_manifest("../../../etc/passwd")

        with pytest.raises(ValueError, match="path traversal"):
            registry.get_manifest("foo/bar\\")


class TestCacheManager:
    """Tests for canonical CacheManager, locking, snapshot reuse, and GC."""

    def test_cache_manager_paths(self, tmp_path: Path) -> None:
        forgeai_home = tmp_path / ".forgeai"
        cache_mgr = CacheManager(forgeai_home=forgeai_home)

        assert cache_mgr.forgeai_home == forgeai_home.resolve()
        assert cache_mgr.hf_home == (forgeai_home / "hf").resolve()
        assert cache_mgr.hub_dir == (forgeai_home / "hf" / "hub").resolve()
        assert cache_mgr.locks_dir == (forgeai_home / "locks").resolve()
        assert cache_mgr.tmp_dir == (forgeai_home / "tmp").resolve()

        assert cache_mgr.hub_dir.exists()
        assert cache_mgr.locks_dir.exists()
        assert cache_mgr.tmp_dir.exists()

    def test_snapshot_download_mock(self, tmp_path: Path) -> None:
        forgeai_home = tmp_path / ".forgeai"
        cache_mgr = CacheManager(forgeai_home=forgeai_home)

        fake_snap = cache_mgr.hub_dir / "models--meta-llama--Llama-3-8B" / "snapshots" / "abc123"

        with patch(
            "huggingface_hub.snapshot_download", return_value=str(fake_snap)
        ) as mock_download:
            res = cache_mgr.download_snapshot(
                repo_id="meta-llama/Llama-3-8B",
                revision="v1.0",
            )
            assert res == str(fake_snap)
            mock_download.assert_called_once_with(
                repo_id="meta-llama/Llama-3-8B",
                cache_dir=str(cache_mgr.hf_home),
                revision="v1.0",
                token=None,
                ignore_patterns=["*.md", "*.txt", "LICENSE*", ".git*"],
            )

    def test_download_model_legacy_cache_dir_forwarding(self, tmp_path: Path) -> None:
        custom_cache = tmp_path / "custom_hf_cache"

        fake_snap = custom_cache / "hub" / "models--org--repo" / "snapshots" / "def456"

        with (
            patch("forgeai.models.loader.CacheManager") as mock_cache_cls,
            patch("huggingface_hub.snapshot_download", return_value=str(fake_snap)),
        ):
            mock_mgr = mock_cache_cls.return_value
            mock_mgr.download_snapshot.return_value = str(fake_snap)

            res = download_model(
                repo_id="org/repo",
                cache_dir=str(custom_cache),
                enable_safety_scan=False,
            )

            assert res == str(fake_snap)
            mock_cache_cls.assert_called_once_with(
                forgeai_home=custom_cache.resolve().parent / ".forgeai",
                hf_home=custom_cache.resolve(),
            )

    def test_snapshot_reuse(self, tmp_path: Path) -> None:
        forgeai_home = tmp_path / ".forgeai"
        cache_mgr = CacheManager(forgeai_home=forgeai_home)

        repo_id = "test-owner/test-model"
        repo_dir = cache_mgr.get_repo_dir(repo_id)
        snap_dir = repo_dir / "snapshots" / "commit123"
        snap_dir.mkdir(parents=True)

        refs_dir = repo_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("commit123", encoding="utf-8")

        with patch("huggingface_hub.snapshot_download") as mock_hf:
            downloaded = cache_mgr.download_snapshot(repo_id, revision="main")
            assert downloaded == str(snap_dir)
            mock_hf.assert_not_called()

    def test_garbage_collect_defensive_and_symlinks(self, tmp_path: Path) -> None:
        forgeai_home = tmp_path / ".forgeai"
        cache_mgr = CacheManager(forgeai_home=forgeai_home)

        m1_snap = cache_mgr.hub_dir / "models--org--m1" / "snapshots" / "active_commit"
        m1_snap.mkdir(parents=True)
        (m1_snap / "model.safetensors").write_text("m1 weights", encoding="utf-8")

        m2_snap = cache_mgr.hub_dir / "models--org--m2" / "snapshots" / "old_commit"
        m2_snap.mkdir(parents=True)
        (m2_snap / "model.safetensors").write_text("m2 weights", encoding="utf-8")

        outside_dir = tmp_path / "outside_sensitive_data"
        outside_dir.mkdir(parents=True)
        (outside_dir / "secret.txt").write_text("do not delete", encoding="utf-8")

        outside_symlink = cache_mgr.hub_dir / "models--org--m3" / "snapshots" / "bad_symlink"
        outside_symlink.parent.mkdir(parents=True)

        symlink_created = False
        with suppress(OSError, NotImplementedError):
            os.symlink(outside_dir, outside_symlink, target_is_directory=True)
            symlink_created = True

        removed = cache_mgr.garbage_collect_unreferenced(active_snapshots=[str(m1_snap)])

        assert removed >= 1
        assert m1_snap.exists()
        assert not m2_snap.exists()
        assert outside_dir.exists()
        assert (outside_dir / "secret.txt").exists()

        if symlink_created:
            assert not outside_symlink.exists() and not outside_symlink.is_symlink()
