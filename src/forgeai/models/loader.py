"""
Model download management, canonical HuggingFace caching, and snapshot purging.

Handles HuggingFace Hub downloads with progress tracking, local caching,
canonical CacheManager storage abstraction, and safety scanning.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from rich.console import Console

from forgeai.models.manifest import validate_repo_id_string
from forgeai.utils.helpers import format_bytes

console = Console()

# Only fetch what the vLLM engine needs. Excluding ``*.bin`` / ``*.pt`` keeps repos that ship
# both pickle and safetensors weights from downloading everything twice (and from tripping
# the pickle scanner). ``*.txt`` is required for merges.txt / vocab.txt.
DOWNLOAD_ALLOW_PATTERNS: list[str] = [
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "chat_template*",
    "*.jinja",
]
# Custom modelling code is only fetched when trust_remote_code is explicitly requested.
REMOTE_CODE_ALLOW_PATTERNS: list[str] = ["*.py"]


class SecurityBlockError(ValueError):
    """Raised when a downloaded snapshot fails the safety scan (and has been purged)."""


def _is_valid_repo_id(repo_id: str) -> bool:
    """Return whether a Hugging Face repository identifier is safe and supported."""
    if not repo_id or len(repo_id) > 200:
        return False
    try:
        validate_repo_id_string(repo_id)
        return True
    except ValueError:
        return False


def _validate_repo_id_or_raise(repo_id: str) -> None:
    """Raise a stable public validation error for invalid repository IDs."""
    if not _is_valid_repo_id(repo_id):
        raise ValueError(f"Invalid HuggingFace repository ID format: {repo_id!r}")


@contextmanager
def _acquire_file_lock(lock_path: Path):
    """Cross-platform file locking using msvcrt (Windows) or fcntl (Unix)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as f:
        if sys.platform == "win32":
            import msvcrt

            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write("\0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                with suppress(Exception):
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                with suppress(Exception):
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class CacheManager:
    """
    Canonical Hugging Face cache manager rooted at FORGEAI_HOME.

    `download_snapshot()` is intentionally a low-level cache primitive. Public
    runtime/model acquisition must use `download_snapshot_secure()` (or
    SecureCacheManager), which adds the fail-closed model safety policy.
    """

    def __init__(
        self,
        forgeai_home: str | Path | None = None,
        hf_home: str | Path | None = None,
    ) -> None:
        if forgeai_home:
            self.forgeai_home = Path(forgeai_home).expanduser().resolve()
        else:
            self.forgeai_home = Path(
                os.environ.get("FORGEAI_HOME", os.path.expanduser("~/.forgeai"))
            ).expanduser().resolve()

        if hf_home:
            self.hf_home = Path(hf_home).expanduser().resolve()
        elif forgeai_home:
            self.hf_home = (self.forgeai_home / "hf").resolve()
        else:
            self.hf_home = Path(
                os.environ.get("HF_HOME", str(self.forgeai_home / "hf"))
            ).expanduser().resolve()

        self.hub_dir = self.hf_home / "hub"
        self.locks_dir = self.forgeai_home / "locks"
        self.tmp_dir = self.forgeai_home / "tmp"

        self.hub_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _repo_to_dir_name(self, repo_id: str) -> str:
        return f"models--{repo_id.replace('/', '--')}"

    def get_repo_dir(self, repo_id: str) -> Path:
        return self.hub_dir / self._repo_to_dir_name(repo_id)

    def get_snapshot_path(self, repo_id: str, revision: str = "main") -> Path | None:
        """Find local snapshot directory for a given repo_id and revision, if present."""
        repo_dir = self.get_repo_dir(repo_id)
        if not repo_dir.exists():
            return None

        ref_file = repo_dir / "refs" / revision
        if ref_file.exists() and ref_file.is_file():
            commit_hash = ref_file.read_text(encoding="utf-8").strip()
            snap_path = repo_dir / "snapshots" / commit_hash
            if snap_path.exists() and snap_path.is_dir():
                return snap_path

        direct_snap = repo_dir / "snapshots" / revision
        if direct_snap.exists() and direct_snap.is_dir():
            return direct_snap

        return None

    def _remove_rejected_snapshot(self, local_path: Path) -> None:
        """Best-effort purge of a rejected snapshot (and its blobs) contained by HF_HOME."""
        try:
            if not local_path.parent.resolve().joinpath(local_path.name).is_relative_to(
                self.hf_home.resolve()
            ):
                return
            self.purge_snapshot(local_path)
        except Exception:
            pass

    def purge_snapshot(self, snapshot_path: str | Path) -> None:
        """Delete a snapshot, the blobs only it referenced, and any refs pointing at it."""
        import shutil

        snap = Path(snapshot_path)
        hub = self.hub_dir.resolve()
        # snapshots/<commit>: resolve the parent only, the snapshot dir itself may be a symlink
        snap_abs = snap.parent.resolve() / snap.name
        model_dir = snap_abs.parent.parent
        if snap_abs.parent.name != "snapshots" or not model_dir.is_relative_to(hub):
            # Not a hub-layout snapshot (e.g. a plain directory): remove just that directory.
            if snap_abs.is_dir() and not snap_abs.is_symlink():
                shutil.rmtree(snap_abs, ignore_errors=True)
            return
        commit = snap_abs.name

        if snap_abs.is_symlink():
            snap_abs.unlink()
        elif snap_abs.is_dir():
            shutil.rmtree(snap_abs, ignore_errors=True)

        refs_dir = model_dir / "refs"
        if refs_dir.is_dir():
            for ref in refs_dir.rglob("*"):
                with suppress(OSError):
                    if ref.is_file() and ref.read_text(encoding="utf-8").strip() == commit:
                        ref.unlink()

        self._prune_unreferenced_blobs(model_dir)

    def scan_snapshot_or_purge(self, snapshot_path: str | Path, repo_id: str) -> dict[str, Any]:
        """Run the safety scan; on failure purge the snapshot and raise SecurityBlockError."""
        from forgeai.models.safety_scanner import scan_model_weights

        scan_result = scan_model_weights(str(snapshot_path))
        if not scan_result["safe"]:
            self.purge_snapshot(snapshot_path)
            raise SecurityBlockError(
                f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                f"Reason: {scan_result.get('reason', 'Unknown')}"
            )
        return scan_result

    @staticmethod
    def _prune_unreferenced_blobs(model_dir: Path) -> int:
        """Delete blobs in ``model_dir/blobs`` that no remaining snapshot links to."""
        blobs_dir = model_dir / "blobs"
        snapshots_dir = model_dir / "snapshots"
        if not blobs_dir.is_dir() or blobs_dir.is_symlink() or not snapshots_dir.is_dir():
            return 0

        blobs_resolved = blobs_dir.resolve()
        referenced: set[Path] = set()
        for root, _dirs, files in os.walk(snapshots_dir, followlinks=False):
            for name in files:
                try:
                    referenced.add(Path(os.path.realpath(os.path.join(root, name))))
                except OSError:
                    continue

        removed = 0
        for blob in blobs_resolved.iterdir():
            if blob.is_symlink() or not blob.is_file() or blob.name.endswith(".incomplete"):
                continue
            if blob.resolve() in referenced:
                continue
            try:
                blob.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def _validate_snapshot_security(
        self,
        local_path: str | Path,
        repo_id: str,
        *,
        enable_safety_scan: bool = True,
    ) -> str:
        """Validate a model snapshot using the fail-closed runtime policy."""
        path = Path(local_path).expanduser().resolve()

        try:
            if not path.exists():
                raise SecurityBlockError(
                    f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                    "Downloaded snapshot path does not exist."
                )

            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            unsafe_suffixes = {".bin", ".pt", ".pth", ".ckpt"}
            unsafe_weights = [p for p in files if p.suffix.lower() in unsafe_suffixes]
            if unsafe_weights:
                names = ", ".join(sorted(p.name for p in unsafe_weights[:5]))
                raise SecurityBlockError(
                    f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                    "Pickle-backed model weights are not permitted. "
                    f"Found: {names}. Use safetensors-only model artifacts."
                )

            safetensors = [p for p in files if p.suffix.lower() == ".safetensors"]
            if not safetensors:
                raise SecurityBlockError(
                    f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                    "Snapshot contains no .safetensors model weights."
                )

            if enable_safety_scan:
                try:
                    self.scan_snapshot_or_purge(path, repo_id)
                except SecurityBlockError:
                    raise
                except Exception as err:
                    raise SecurityBlockError(
                        f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                        f"Scanner error: {err}"
                    ) from err

            return str(path)
        except Exception:
            self._remove_rejected_snapshot(path)
            raise

    def download_snapshot(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> str:
        """Download/reuse a raw Hugging Face cache snapshot.

        Only files the engine needs are fetched (``DOWNLOAD_ALLOW_PATTERNS``); ``*.py`` files
        are only fetched when ``trust_remote_code`` is True.
        """
        _validate_repo_id_or_raise(repo_id)

        existing = self.get_snapshot_path(repo_id, revision)
        if existing and existing.exists():
            return str(existing)

        lock_path = self.locks_dir / f"{self._repo_to_dir_name(repo_id)}.lock"

        with _acquire_file_lock(lock_path):
            existing = self.get_snapshot_path(repo_id, revision)
            if existing and existing.exists():
                return str(existing)

            try:
                from huggingface_hub import snapshot_download
            except ImportError as err:
                raise RuntimeError(
                    "huggingface_hub is required for model downloads.\n"
                    "Install with: pip install huggingface-hub"
                ) from err

            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            return str(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.hf_home),
                    revision=revision,
                    token=token,
                    allow_patterns=DOWNLOAD_ALLOW_PATTERNS
                    + (REMOTE_CODE_ALLOW_PATTERNS if trust_remote_code else []),
                )
            )

    def download_snapshot_secure(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        *,
        enable_safety_scan: bool = True,
        trust_remote_code: bool = False,
    ) -> str:
        """Download/reuse and then validate a snapshot before runtime use."""
        local_path = self.download_snapshot(
            repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code
        )
        return self._validate_snapshot_security(
            local_path,
            repo_id,
            enable_safety_scan=enable_safety_scan,
        )

    async def pull_snapshot(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> str:
        """Async wrapper for the secure model-acquisition path."""
        return await asyncio.to_thread(
            self.download_snapshot_secure,
            repo_id,
            revision,
            token,
            trust_remote_code=trust_remote_code,
        )


class SecureCacheManager(CacheManager):
    """Cache manager whose public download operation is always security validated."""

    def download_snapshot(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> str:
        raw_path = super().download_snapshot(
            repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code
        )
        return self._validate_snapshot_security(raw_path, repo_id, enable_safety_scan=True)

    def download_snapshot_secure(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        *,
        enable_safety_scan: bool = True,
        trust_remote_code: bool = False,
    ) -> str:
        raw_path = super().download_snapshot(
            repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code
        )
        return self._validate_snapshot_security(
            raw_path,
            repo_id,
            enable_safety_scan=enable_safety_scan,
        )


def download_model(
    repo_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    enable_safety_scan: bool = True,
    trust_remote_code: bool = False,
) -> str:
    """Download a model from HuggingFace Hub and enforce the runtime safety policy."""
    _validate_repo_id_or_raise(repo_id)

    if cache_dir:
        cache_path = Path(cache_dir).expanduser().resolve()
        # Preserve the historical cache layout except when its parent is the
        # filesystem root (e.g. cache_dir=/tmp), which would otherwise try to
        # create /.forgeai and fail for non-root users.
        forgeai_home = (
            cache_path / ".forgeai"
            if cache_path.parent == cache_path.anchor or cache_path.parent == Path(cache_path.anchor)
            else cache_path.parent / ".forgeai"
        )
        manager = CacheManager(
            forgeai_home=forgeai_home,
            hf_home=cache_path,
        )
    else:
        manager = CacheManager()

    console.print(f"\n[bold]Downloading:[/bold] {repo_id}")
    if revision:
        console.print(f"  Revision: {revision}")
    if enable_safety_scan:
        console.print("\n[bold]Running safety scan...[/bold]")

    # Keep the low-level call contract stable for existing callers/tests, then
    # apply the same fail-closed format/scanner validation before returning.
    local_path = manager.download_snapshot(
        repo_id,
        revision=revision or "main",
        token=token,
        trust_remote_code=trust_remote_code,
    )
    manager._validate_snapshot_security(
        local_path,
        repo_id,
        enable_safety_scan=enable_safety_scan,
    )

    console.print(f"\n[green]✓[/green] Model saved to: {local_path}")
    if enable_safety_scan:
        console.print("[green]✓[/green] Safety scan passed")
    return local_path


def _dir_size_no_follow(root: Path) -> int:
    """Sum file sizes under ``root`` without following symlinks.

    HF caches store each file once in ``blobs/`` and expose it through symlinks in
    ``snapshots/``; following links would count every blob once per snapshot. Hard-linked
    files are counted once via (st_dev, st_ino).
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            ident = (st.st_dev, st.st_ino)
            if ident in seen:
                continue
            seen.add(ident)
            total += st.st_size
    return total


def get_cached_models(cache_dir: str | None = None) -> list[dict[str, str]]:
    """List all locally cached models in HuggingFace hub."""
    if cache_dir:
        hub_path = Path(cache_dir)
    else:
        manager = CacheManager()
        hub_path = manager.hub_dir

    models: list[dict[str, str]] = []
    if not hub_path.exists():
        return models

    for entry in hub_path.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name.replace("models--", "").split("--")
            repo_id = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]
            total_size = _dir_size_no_follow(entry)

            models.append({
                "repo_id": repo_id,
                "path": str(entry),
                "size": format_bytes(total_size),
            })

    return models


def delete_cached_model(repo_id: str, cache_dir: str | None = None) -> bool:
    """Delete a cached model directory."""
    _validate_repo_id_or_raise(repo_id)

    import shutil

    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        manager = CacheManager()
        cache_path = manager.hub_dir

    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_path = cache_path / dir_name

    if model_path.exists():
        shutil.rmtree(model_path)
        console.print(f"[green]✓[/green] Deleted cached model: {repo_id}")
        return True

    console.print(f"[yellow]⚠[/yellow] Model not found in cache: {repo_id}")
    return False
