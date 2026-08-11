"""
Model download management, canonical HuggingFace caching, and garbage collection.

Handles HuggingFace Hub downloads with progress tracking, local caching,
canonical CacheManager storage abstraction, and safety scanning.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager, suppress
from pathlib import Path

from rich.console import Console

from forgeai.models.manifest import validate_repo_id_string
from forgeai.utils.helpers import format_bytes

console = Console()


def _is_valid_repo_id(repo_id: str) -> bool:
    """
    Validate HuggingFace Hub repository ID format.
    Returns True if valid, False otherwise.
    """
    try:
        validate_repo_id_string(repo_id)
        return True
    except ValueError:
        return False


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

    Manages canonical HF_HOME=${FORGEAI_HOME}/hf, file locks under
    ${FORGEAI_HOME}/locks, and partial temporary work under ${FORGEAI_HOME}/tmp.
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

        # Check ref pointer file
        ref_file = repo_dir / "refs" / revision
        if ref_file.exists() and ref_file.is_file():
            commit_hash = ref_file.read_text(encoding="utf-8").strip()
            snap_path = repo_dir / "snapshots" / commit_hash
            if snap_path.exists() and snap_path.is_dir():
                return snap_path

        # Direct commit hash check
        direct_snap = repo_dir / "snapshots" / revision
        if direct_snap.exists() and direct_snap.is_dir():
            return direct_snap

        # If snapshots dir has subdirectories, check if one matches
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            snaps = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if len(snaps) == 1:
                return snaps[0]

        return None

    def download_snapshot(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
    ) -> str:
        """
        Download snapshot using Hugging Face snapshot_download.
        Reuses existing snapshot if repo_id/revision is already present in canonical HF cache.
        """
        validate_repo_id_string(repo_id)

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

            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(self.hf_home),
                revision=revision,
                token=token,
                ignore_patterns=["*.md", "*.txt", "LICENSE*", ".git*"],
            )

            return str(local_path)

    async def pull_snapshot(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
    ) -> str:
        """Async wrapper for download_snapshot."""
        return await asyncio.to_thread(self.download_snapshot, repo_id, revision, token)

    def garbage_collect_unreferenced(self, active_snapshots: list[str]) -> int:
        """
        Remove unreferenced Hugging Face snapshot directories within the canonical hub.

        Defensive safety rules:
        - Never delete active/referenced snapshots.
        - Never delete anything outside self.hub_dir.
        - Never follow or delete outside symlink targets.
        - Returns count of snapshot directories removed.
        """
        hub_resolved = self.hub_dir.resolve()
        if not hub_resolved.exists():
            return 0

        active_paths: set[Path] = set()
        for snap_str in active_snapshots:
            if not snap_str:
                continue
            try:
                p = Path(snap_str).resolve()
                active_paths.add(p)
            except Exception:
                pass

        removed_count = 0

        for model_dir in hub_resolved.glob("models--*"):
            if not model_dir.is_dir():
                continue

            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists() or not snapshots_dir.is_dir():
                continue

            for snap_dir in list(snapshots_dir.iterdir()):
                if not (snap_dir.is_dir() or snap_dir.is_symlink()):
                    continue

                # Symlink safety check: if snap_dir is a symlink, check target containment
                if snap_dir.is_symlink():
                    try:
                        target_resolved = snap_dir.resolve()
                        if not target_resolved.is_relative_to(hub_resolved):
                            # Points outside hub_dir: unlink symlink only, never rmtree target!
                            snap_dir.unlink()
                            removed_count += 1
                            continue
                    except Exception:
                        snap_dir.unlink()
                        removed_count += 1
                        continue

                snap_resolved = snap_dir.resolve()

                # Containment check: must be inside self.hub_dir
                try:
                    if not snap_resolved.is_relative_to(hub_resolved):
                        continue
                except ValueError:
                    continue

                # Must not be hub_dir itself or forgeai_home
                if snap_resolved == hub_resolved or snap_resolved == self.forgeai_home.resolve():
                    continue

                # Check if referenced directly
                if snap_resolved in active_paths or str(snap_resolved) in active_snapshots:
                    continue

                # Check if any active path is nested inside snap_resolved
                is_active = False
                for act_path in active_paths:
                    try:
                        if act_path.is_relative_to(snap_resolved):
                            is_active = True
                            break
                    except ValueError:
                        pass

                if is_active:
                    continue

                # Defensive deletion of unreferenced snapshot
                try:
                    import shutil

                    shutil.rmtree(snap_resolved)
                    removed_count += 1
                except Exception:
                    pass

        return removed_count


def download_model(
    repo_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    enable_safety_scan: bool = True,
) -> str:
    """
    Download a model from HuggingFace Hub using canonical CacheManager.

    Args:
        repo_id: HuggingFace repository ID.
        cache_dir: Local cache directory.
        revision: Specific model revision/branch.
        token: HuggingFace API token for private models.
        enable_safety_scan: Run safety scanner after download.

    Returns:
        Local path to the downloaded model.
    """
    validate_repo_id_string(repo_id)

    if cache_dir:
        cache_path = Path(cache_dir).expanduser().resolve()
        manager = CacheManager(
            forgeai_home=cache_path.parent / ".forgeai",
            hf_home=cache_path,
        )
    else:
        manager = CacheManager()

    console.print(f"\n[bold]Downloading:[/bold] {repo_id}")
    if revision:
        console.print(f"  Revision: {revision}")

    local_path = manager.download_snapshot(repo_id, revision=revision or "main", token=token)

    console.print(f"\n[green]✓[/green] Model saved to: {local_path}")

    if enable_safety_scan:
        console.print("\n[bold]Running safety scan...[/bold]")
        try:
            from forgeai.models.safety_scanner import scan_model_weights

            scan_result = scan_model_weights(local_path)
            if scan_result["safe"]:
                console.print("[green]✓[/green] Safety scan passed")
            else:
                console.print(
                    f"[red]✗[/red] Safety scan flagged issues:\n"
                    f"  {scan_result.get('reason', 'Unknown')}"
                )
                try:
                    import shutil

                    if os.path.isdir(local_path):
                        shutil.rmtree(local_path)
                    elif os.path.isfile(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

                raise ValueError(
                    f"SECURITY BLOCK: Model safety scan failed for {repo_id}. "
                    f"Reason: {scan_result.get('reason', 'Unknown')}"
                )
        except ValueError:
            raise
        except Exception as e:
            console.print(f"[yellow]⚠ Safety scan skipped: {e}[/yellow]")

    return local_path


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
            total_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())

            models.append({
                "repo_id": repo_id,
                "path": str(entry),
                "size": format_bytes(total_size),
            })

    return models


def delete_cached_model(repo_id: str, cache_dir: str | None = None) -> bool:
    """Delete a cached model directory."""
    validate_repo_id_string(repo_id)

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
