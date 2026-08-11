"""
Model discovery, multimodal detection, and local manifest registry.

Auto-detects model capabilities from HuggingFace Hub configuration,
including multimodal support (vision, audio).
Manages model tag manifests in YAML format under FORGEAI_HOME/manifests.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from forgeai.models.manifest import ForgeAIManifest

if TYPE_CHECKING:
    from forgeai.models.loader import CacheManager

console = Console()


@dataclass
class ModelRef:
    """Reference to a registered model tag and manifest content digest."""

    name: str
    tag: str
    digest: str

    @property
    def full_tag(self) -> str:
        if self.tag:
            return f"{self.name}:{self.tag}"
        return f"{self.name}:latest"


@dataclass
class ModelRecord:
    """Record describing a registered model, manifest location, and snapshot details."""

    ref: ModelRef
    manifest_path: str
    snapshot_path: str
    size_bytes: int


def parse_tag(tag_str: str) -> tuple[str, str]:
    """Parse tag string into (name, tag_version) using rsplit."""
    if ":" in tag_str:
        parts = tag_str.rsplit(":", 1)
        return parts[0], parts[1]
    return tag_str, "latest"


def format_tag(name: str, tag: str) -> str:
    """Format name and tag into full tag string."""
    if tag:
        return f"{name}:{tag}"
    return f"{name}:latest"


def validate_tag_string(tag: str) -> None:
    """Validate that tag is non-empty and contains safe characters without traversal."""
    if not tag or not tag.strip():
        raise ValueError("Model tag cannot be empty.")
    if "\0" in tag or "\\" in tag or ".." in tag:
        raise ValueError(f"Invalid tag name with potential path traversal: {tag!r}")
    if (
        tag.startswith("/")
        or tag.startswith(":")
        or tag.endswith(":")
        or tag.endswith("/")
        or (len(tag) > 1 and tag[1] == ":")
    ):
        raise ValueError(f"Invalid tag name structure: {tag!r}")

    name_part, version_part = parse_tag(tag)
    if not name_part or not version_part:
        raise ValueError(f"Invalid tag name or version: {tag!r}")

    comp_pattern = re.compile(r"^[A-Za-z0-9._-]+$")
    if not comp_pattern.match(version_part):
        raise ValueError(f"Invalid characters in tag version: {tag!r}")

    name_components = name_part.split("/")
    if len(name_components) > 2 or any(not c for c in name_components):
        raise ValueError(f"Invalid tag namespace structure: {tag!r}")

    for comp in name_components:
        if not comp_pattern.match(comp):
            raise ValueError(f"Invalid characters in tag name component: {tag!r}")


class ModelRegistry:
    """
    Catalog of model manifests stored in YAML format under ${FORGEAI_HOME}/manifests.
    """

    def __init__(self, manifests_dir: str | Path | None = None) -> None:
        if manifests_dir:
            self.manifests_dir = Path(manifests_dir).expanduser().resolve()
        else:
            forgeai_home = Path(
                os.environ.get("FORGEAI_HOME", os.path.expanduser("~/.forgeai"))
            ).expanduser().resolve()
            self.manifests_dir = forgeai_home / "manifests"

        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _tag_to_filename(self, tag: str) -> str:
        """Convert a model tag to a safe, deterministic, URL-quoted filename."""
        validate_tag_string(tag)
        safe_name = urllib.parse.quote(tag, safe="")
        return f"{safe_name}.yaml"

    def _get_manifest_path(self, tag: str) -> Path:
        filename = self._tag_to_filename(tag)
        path = (self.manifests_dir / filename).resolve()
        try:
            if not path.is_relative_to(self.manifests_dir.resolve()):
                raise ValueError(f"Path traversal detected for tag: {tag!r}")
        except ValueError as err:
            raise ValueError(f"Path traversal detected for tag: {tag!r}") from err
        return path

    def get_manifest(self, tag: str) -> ForgeAIManifest:
        """Load and return ForgeAIManifest for tag."""
        path = self._get_manifest_path(tag)
        if not path.exists() and ":" not in tag:
            path = self._get_manifest_path(f"{tag}:latest")

        if not path.exists():
            raise KeyError(f"Model manifest not found for tag: {tag!r}")

        yaml_content = path.read_text(encoding="utf-8")
        manifest: ForgeAIManifest = ForgeAIManifest.from_yaml(yaml_content)
        return manifest

    def register_manifest(
        self,
        manifest: ForgeAIManifest,
        cache_manager: CacheManager | None = None,
    ) -> ModelRecord:
        """
        Deterministically serialize and atomically publish a manifest.
        A failed write must not corrupt an existing manifest.
        """
        tag = manifest.name
        target_path = self._get_manifest_path(tag)

        yaml_content = manifest.to_yaml()
        digest = hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()

        tmp_dir = self.manifests_dir.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        temp_fd, temp_path_str = tempfile.mkstemp(dir=str(tmp_dir), suffix=".yaml.tmp")
        temp_path = Path(temp_path_str)

        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(yaml_content)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, target_path)
        except Exception:
            if temp_path.exists():
                with suppress(Exception):
                    os.remove(temp_path)
            raise

        snapshot_path, size_bytes = self._resolve_snapshot_info(
            manifest, cache_manager=cache_manager
        )
        name_part, tag_part = parse_tag(tag)
        ref = ModelRef(name=name_part, tag=tag_part, digest=digest)

        return ModelRecord(
            ref=ref,
            manifest_path=str(target_path),
            snapshot_path=str(snapshot_path),
            size_bytes=size_bytes,
        )

    def unregister_tag(
        self,
        tag: str,
        cache_manager: CacheManager | None = None,
    ) -> ModelRecord | None:
        """Unregister tag by removing manifest file. Returns ModelRecord before removal if found."""
        try:
            record = self.get_record(tag, cache_manager=cache_manager)
            path = Path(record.manifest_path)
            if path.exists():
                path.unlink()
            return record
        except KeyError:
            return None

    def get_record(self, tag: str, cache_manager: CacheManager | None = None) -> ModelRecord:
        """Get ModelRecord for a specific tag."""
        manifest = self.get_manifest(tag)
        path = self._get_manifest_path(tag)
        if not path.exists() and ":" not in tag:
            path = self._get_manifest_path(f"{tag}:latest")
        yaml_content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()

        snapshot_path, size_bytes = self._resolve_snapshot_info(
            manifest, cache_manager=cache_manager
        )
        name_part, tag_part = parse_tag(manifest.name)
        ref = ModelRef(name=name_part, tag=tag_part, digest=digest)

        return ModelRecord(
            ref=ref,
            manifest_path=str(path),
            snapshot_path=str(snapshot_path),
            size_bytes=size_bytes,
        )

    def list_records(self, cache_manager: CacheManager | None = None) -> list[ModelRecord]:
        """List records for all registered model manifests."""
        records: list[ModelRecord] = []
        if not self.manifests_dir.exists():
            return records

        for yaml_file in sorted(self.manifests_dir.glob("*.yaml")):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                manifest = ForgeAIManifest.from_yaml(content)
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                snapshot_path, size_bytes = self._resolve_snapshot_info(
                    manifest, cache_manager=cache_manager
                )
                name_part, tag_part = parse_tag(manifest.name)
                ref = ModelRef(name=name_part, tag=tag_part, digest=digest)
                records.append(
                    ModelRecord(
                        ref=ref,
                        manifest_path=str(yaml_file),
                        snapshot_path=str(snapshot_path),
                        size_bytes=size_bytes,
                    )
                )
            except Exception:
                continue

        return records

    def _resolve_snapshot_info(
        self,
        manifest: ForgeAIManifest,
        cache_manager: CacheManager | None = None,
    ) -> tuple[str, int]:
        """Resolve model snapshot directory path and size in bytes."""
        if manifest.source_kind == "local_dir":
            local_path = Path(manifest.model).resolve()
            if local_path.exists():
                size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
                return str(local_path), size
            return manifest.model, 0

        # source_kind == "huggingface"
        if cache_manager is not None:
            snap_path = cache_manager.get_snapshot_path(manifest.model, manifest.revision)
            if snap_path and snap_path.exists():
                size = sum(f.stat().st_size for f in snap_path.rglob("*") if f.is_file())
                return str(snap_path), size
            hub_path = cache_manager.hub_dir / f"models--{manifest.model.replace('/', '--')}"
            return str(hub_path), 0

        forgeai_home = Path(
            os.environ.get("FORGEAI_HOME", os.path.expanduser("~/.forgeai"))
        ).expanduser().resolve()
        hub_dir = (
            Path(os.environ.get("HF_HOME", str(forgeai_home / "hf"))).expanduser().resolve()
            / "hub"
        )
        model_dir = hub_dir / f"models--{manifest.model.replace('/', '--')}"
        return str(model_dir), 0


@dataclass
class ModelInfo:
    """Metadata about a discovered model."""

    name: str
    repo_id: str
    architecture: str = ""
    param_count: float | None = None  # In billions
    max_position_embeddings: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    hidden_size: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    model_type: str = ""
    quantization: str | None = None
    is_multimodal: bool = False
    multimodal_types: list[str] = field(default_factory=list)
    supports_vision: bool = False
    supports_audio: bool = False
    local_path: str | None = None


def discover_model(repo_id: str, cache_dir: str | None = None) -> ModelInfo:
    """
    Discover model metadata from HuggingFace Hub.

    Downloads and parses config.json to extract architecture details
    and detect multimodal capabilities.
    """
    info = ModelInfo(name=repo_id.split("/")[-1], repo_id=repo_id)

    try:
        from huggingface_hub import hf_hub_download

        # Download config.json
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=cache_dir,
        )

        import json

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        info = _parse_config(info, config)
        console.print(f"[green]✓[/green] Discovered: [bold]{info.name}[/bold] ({info.architecture})")

    except ImportError:
        console.print("[dim]huggingface_hub not available — limited discovery[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Model discovery failed: {e}[/yellow]")

    return info


def _parse_config(info: ModelInfo, config: dict[str, Any]) -> ModelInfo:
    """Parse model config.json into ModelInfo."""
    info.model_type = config.get("model_type", "")

    # Architecture detection
    architectures = config.get("architectures", [])
    if architectures:
        info.architecture = architectures[0]

    # Model dimensions
    info.hidden_size = config.get("hidden_size", 0)
    info.num_layers = config.get("num_hidden_layers", 0)
    info.num_attention_heads = config.get("num_attention_heads", 0)
    info.num_kv_heads = config.get("num_key_value_heads", info.num_attention_heads)
    info.max_position_embeddings = config.get("max_position_embeddings", 0)
    info.vocab_size = config.get("vocab_size", 0)

    # Head dimension
    if info.hidden_size and info.num_attention_heads:
        info.head_dim = info.hidden_size // info.num_attention_heads

    # Parameter count estimation
    if info.hidden_size and info.num_layers and info.vocab_size:
        # Rough estimation: ~12 * hidden_size^2 * num_layers + vocab * hidden
        params = (
            12 * info.hidden_size**2 * info.num_layers
            + info.vocab_size * info.hidden_size
        )
        info.param_count = params / 1e9

    # Quantization detection
    quant_config = config.get("quantization_config", {})
    if quant_config:
        info.quantization = quant_config.get("quant_method", None)

    # Multimodal detection
    info = _detect_multimodal(info, config)

    return info


def _detect_multimodal(info: ModelInfo, config: dict[str, Any]) -> ModelInfo:
    """Detect multimodal capabilities from config."""
    arch = info.architecture.lower()
    model_type = info.model_type.lower()

    # Vision models
    vision_indicators = [
        "vision" in arch,
        "vl" in model_type,
        "visual" in arch,
        "image" in arch,
        "llava" in arch,
        "internvl" in model_type,
        "qwen2_vl" in model_type,
        config.get("vision_config") is not None,
        config.get("visual_config") is not None,
        config.get("image_size") is not None,
    ]

    if any(vision_indicators):
        info.is_multimodal = True
        info.supports_vision = True
        info.multimodal_types.append("vision")

    # Audio models
    audio_indicators = [
        "audio" in arch,
        "whisper" in model_type,
        "speech" in arch,
        config.get("audio_config") is not None,
    ]

    if any(audio_indicators):
        info.is_multimodal = True
        info.supports_audio = True
        info.multimodal_types.append("audio")

    return info


def is_multimodal(repo_id: str, cache_dir: str | None = None) -> bool:
    """Quick check if a model supports multimodal inputs."""
    info = discover_model(repo_id, cache_dir)
    return info.is_multimodal
