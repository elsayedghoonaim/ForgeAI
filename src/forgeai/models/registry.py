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
import threading
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
    manifest: ForgeAIManifest | None = None


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


_FileKey = tuple[int, int]  # (st_mtime_ns, st_size)


@dataclass
class _ManifestEntry:
    """Parsed manifest plus its digest, valid for one (mtime_ns, size) of the file."""

    file_key: _FileKey
    manifest: ForgeAIManifest
    digest: str


def _dir_size(path: Path) -> int:
    """Total size of the files below ``path`` (symlinks to files are followed)."""
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


class ModelRegistry:
    """
    Catalog of model manifests stored in YAML format under ${FORGEAI_HOME}/manifests.

    Parsed manifests, digests and resolved snapshot paths are cached per manifest file,
    keyed by ``(path, mtime_ns, size)``, so request handling does not re-read YAML or
    re-scan snapshot directories. Snapshot sizes are computed lazily (listing endpoints
    only) and cached per snapshot path. All caches are invalidated by
    ``register_manifest``, ``unregister_tag`` and ``invalidate``.
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

        self._lock = threading.RLock()
        self._entries: dict[Path, _ManifestEntry] = {}
        self._snapshots: dict[tuple[Path, _FileKey, str], str] = {}
        self._sizes: dict[str, tuple[int, int]] = {}  # snapshot path -> (dir mtime_ns, size)

    # ------------------------------------------------------------------ caches

    def invalidate(self, path: Path | None = None) -> None:
        """Drop cached state for one manifest path, or for everything."""
        with self._lock:
            if path is None:
                self._entries.clear()
                self._snapshots.clear()
                self._sizes.clear()
                return
            self._entries.pop(path, None)
            for key in [k for k in self._snapshots if k[0] == path]:
                del self._snapshots[key]
            # Sizes are keyed by snapshot path (shared across tags); drop them all, they
            # are cheap to recompute and only ever needed by listing endpoints.
            self._sizes.clear()

    def _load_entry(self, path: Path) -> _ManifestEntry:
        """Return the parsed manifest for ``path``, re-reading only if the file changed."""
        st = path.stat()  # FileNotFoundError propagates to callers
        file_key = (st.st_mtime_ns, st.st_size)
        with self._lock:
            cached = self._entries.get(path)
            if cached is not None and cached.file_key == file_key:
                return cached

        content = path.read_text(encoding="utf-8")
        manifest = ForgeAIManifest.from_yaml(content)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        entry = _ManifestEntry(file_key=file_key, manifest=manifest, digest=digest)
        with self._lock:
            self._entries[path] = entry
        return entry

    # ------------------------------------------------------------------ paths

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

    def _find_manifest_path(self, tag: str) -> Path:
        path = self._get_manifest_path(tag)
        if not path.exists() and ":" not in tag:
            path = self._get_manifest_path(f"{tag}:latest")
        if not path.exists():
            raise KeyError(f"Model manifest not found for tag: {tag!r}")
        return path

    # -------------------------------------------------------------------- API

    def get_manifest(self, tag: str) -> ForgeAIManifest:
        """Load and return ForgeAIManifest for tag (a private copy of the cached one)."""
        path = self._find_manifest_path(tag)
        try:
            entry = self._load_entry(path)
        except FileNotFoundError as err:
            raise KeyError(f"Model manifest not found for tag: {tag!r}") from err
        return entry.manifest.model_copy(deep=True)

    def register_manifest(
        self,
        manifest: ForgeAIManifest,
        cache_manager: CacheManager | None = None,
        *,
        include_size: bool = False,
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
        finally:
            self.invalidate(target_path)

        # Seed the cache with what was just written so the next request does not re-parse.
        with suppress(OSError):
            st = target_path.stat()
            with self._lock:
                self._entries[target_path] = _ManifestEntry(
                    file_key=(st.st_mtime_ns, st.st_size),
                    manifest=manifest.model_copy(deep=True),
                    digest=digest,
                )

        snapshot_path, size_bytes = self._resolve_snapshot_info(
            target_path, manifest, cache_manager=cache_manager, include_size=include_size
        )
        name_part, tag_part = parse_tag(tag)
        ref = ModelRef(name=name_part, tag=tag_part, digest=digest)

        return ModelRecord(
            ref=ref,
            manifest_path=str(target_path),
            snapshot_path=str(snapshot_path),
            size_bytes=size_bytes,
            manifest=manifest.model_copy(deep=True),
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
            try:
                if path.exists():
                    path.unlink()
            finally:
                self.invalidate(path)
            return record
        except KeyError:
            return None

    def get_record(
        self,
        tag: str,
        cache_manager: CacheManager | None = None,
        *,
        include_size: bool = False,
    ) -> ModelRecord:
        """Get ModelRecord (including the parsed manifest) for a specific tag.

        ``size_bytes`` is 0 unless ``include_size`` is set; only listing endpoints need it.
        """
        path = self._find_manifest_path(tag)
        try:
            entry = self._load_entry(path)
        except FileNotFoundError as err:
            raise KeyError(f"Model manifest not found for tag: {tag!r}") from err
        return self._build_record(path, entry, cache_manager, include_size)

    def list_records(
        self,
        cache_manager: CacheManager | None = None,
        *,
        include_size: bool = True,
    ) -> list[ModelRecord]:
        """List records for all registered model manifests."""
        records: list[ModelRecord] = []
        if not self.manifests_dir.exists():
            return records

        seen: set[Path] = set()
        for yaml_file in sorted(self.manifests_dir.glob("*.yaml")):
            seen.add(yaml_file)
            try:
                entry = self._load_entry(yaml_file)
                records.append(self._build_record(yaml_file, entry, cache_manager, include_size))
            except Exception:
                continue

        with self._lock:
            for stale in [p for p in self._entries if p not in seen]:
                self._entries.pop(stale, None)
        return records

    def _build_record(
        self,
        path: Path,
        entry: _ManifestEntry,
        cache_manager: CacheManager | None,
        include_size: bool,
    ) -> ModelRecord:
        manifest = entry.manifest
        snapshot_path, size_bytes = self._resolve_snapshot_info(
            path, manifest, cache_manager=cache_manager, include_size=include_size
        )
        name_part, tag_part = parse_tag(manifest.name)
        ref = ModelRef(name=name_part, tag=tag_part, digest=entry.digest)
        return ModelRecord(
            ref=ref,
            manifest_path=str(path),
            snapshot_path=str(snapshot_path),
            size_bytes=size_bytes,
            manifest=manifest.model_copy(deep=True),
        )

    def _snapshot_size(self, snapshot_path: str) -> int:
        """Lazily compute and cache the size of a snapshot directory."""
        try:
            dir_mtime = os.stat(snapshot_path).st_mtime_ns
        except OSError:
            return 0
        with self._lock:
            cached = self._sizes.get(snapshot_path)
            if cached is not None and cached[0] == dir_mtime:
                return cached[1]
        size = _dir_size(Path(snapshot_path))
        with self._lock:
            self._sizes[snapshot_path] = (dir_mtime, size)
        return size

    def _resolve_snapshot_info(
        self,
        manifest_path: Path,
        manifest: ForgeAIManifest,
        cache_manager: CacheManager | None = None,
        include_size: bool = False,
    ) -> tuple[str, int]:
        """Resolve the model snapshot directory path and (optionally) its size in bytes."""
        try:
            st = manifest_path.stat()
            file_key: _FileKey = (st.st_mtime_ns, st.st_size)
        except OSError:
            file_key = (0, 0)

        if manifest.source_kind == "local_dir":
            local_path = Path(manifest.model).resolve()
            if local_path.exists():
                size = self._snapshot_size(str(local_path)) if include_size else 0
                return str(local_path), size
            return manifest.model, 0

        # source_kind == "huggingface"
        hub_dir = (
            cache_manager.hub_dir
            if cache_manager is not None
            else Path(
                os.environ.get(
                    "HF_HOME",
                    str(
                        Path(os.environ.get("FORGEAI_HOME", os.path.expanduser("~/.forgeai")))
                        .expanduser()
                        .resolve()
                        / "hf"
                    ),
                )
            )
            .expanduser()
            .resolve()
            / "hub"
        )
        hub_path = hub_dir / f"models--{manifest.model.replace('/', '--')}"

        if cache_manager is None:
            return str(hub_path), 0

        cache_key = (manifest_path, file_key, str(hub_dir))
        with self._lock:
            cached_snap = self._snapshots.get(cache_key)
        if cached_snap is not None and os.path.isdir(cached_snap):
            snap_str = cached_snap
        else:
            snap_path = cache_manager.get_snapshot_path(manifest.model, manifest.revision)
            if not (snap_path and snap_path.exists()):
                with self._lock:
                    self._snapshots.pop(cache_key, None)
                return str(hub_path), 0
            snap_str = str(snap_path)
            with self._lock:
                self._snapshots[cache_key] = snap_str
        size = self._snapshot_size(snap_str) if include_size else 0
        return snap_str, size


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
