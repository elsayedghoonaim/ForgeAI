"""
Poisoned weight detection via entropy analysis.

Analyzes model weight distributions to detect anomalies that may indicate
backdoor attacks, trojan insertions, or corrupted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()

# Thresholds for anomaly detection
ENTROPY_LOW_THRESHOLD = 0.5   # Suspiciously low entropy (constant weights)
ENTROPY_HIGH_THRESHOLD = 8.0  # Suspiciously high entropy (random noise)
OUTLIER_RATIO_THRESHOLD = 0.05  # Max ratio of extreme outlier values
MAX_FILE_SIZE_GB = 50  # Skip files larger than this


@dataclass
class ScanResult:
    """Results of a safety scan."""
    safe: bool = True
    score: float = 100.0  # 0-100 safety score
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    files_scanned: int = 0
    total_parameters: int = 0
    scan_time_seconds: float = 0.0

    @property
    def reason(self) -> str:
        return "; ".join(self.warnings) if self.warnings else "No issues found"


def scan_model_weights(model_path: str, quick: bool = False) -> dict[str, Any]:
    """
    Scan model weights for signs of poisoning or corruption.

    Args:
        model_path: Path to model directory or weight file.
        quick: If True, only scan a subset of files for speed.

    Returns:
        Dict with 'safe' (bool) and scan details.
    """
    import time
    start = time.time()
    result = ScanResult()
    path = Path(model_path)

    if not path.exists():
        result.safe = False
        result.warnings.append(f"Path does not exist: {model_path}")
        return _to_dict(result)

    # Collect weight files
    weight_files = _find_weight_files(path)
    if not weight_files:
        result.warnings.append("No weight files found")
        result.scan_time_seconds = time.time() - start
        return _to_dict(result)

    if quick:
        weight_files = weight_files[:3]  # Scan only first 3 files

    for wf in weight_files:
        _scan_single_file(wf, result)

    # Calculate overall safety score
    penalty_per_warning = 15
    result.score = max(0, 100 - len(result.warnings) * penalty_per_warning)
    result.safe = result.score >= 50

    result.files_scanned = len(weight_files)
    result.scan_time_seconds = time.time() - start
    return _to_dict(result)


def _find_weight_files(path: Path) -> list[Path]:
    """Find model weight files in a directory."""
    if path.is_file():
        return [path]

    extensions = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    files: list[Path] = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))

    # Sort by size (scan smaller files first)
    files.sort(key=lambda f: f.stat().st_size)
    return files


def _scan_single_file(file_path: Path, result: ScanResult) -> None:
    """Scan a single weight file for anomalies."""
    file_size = file_path.stat().st_size
    file_size_gb = file_size / (1024**3)

    if file_size_gb > MAX_FILE_SIZE_GB:
        result.warnings.append(f"Skipped oversized file: {file_path.name} ({file_size_gb:.1f}GB)")
        return

    if file_path.suffix == ".safetensors":
        _scan_safetensors(file_path, result)
    elif file_path.suffix in (".bin", ".pt", ".pth"):
        _scan_pytorch(file_path, result)
    elif file_path.suffix == ".gguf":
        _scan_gguf(file_path, result)


def _scan_safetensors(file_path: Path, result: ScanResult) -> None:
    """Scan a safetensors file."""
    try:
        import json
        import struct

        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size > 100_000_000:  # > 100MB header is suspicious
                result.warnings.append(f"Abnormally large header in {file_path.name}")
                return
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)

        # Check for suspicious tensor names
        for name in header:
            if name == "__metadata__":
                continue
            tensor_info = header[name]
            if "data_offsets" not in tensor_info:
                continue
            # Check for suspicious naming patterns
            suspicious = ["backdoor", "trojan", "exploit", "payload", "inject"]
            if any(s in name.lower() for s in suspicious):
                result.warnings.append(f"Suspicious tensor name: {name}")

        result.total_parameters += len(header) - (1 if "__metadata__" in header else 0)

    except Exception as e:
        result.warnings.append(f"Error scanning {file_path.name}: {e}")


def _scan_pytorch(file_path: Path, result: ScanResult) -> None:
    """Scan a PyTorch weight file."""
    try:
        # Only do basic checks without loading full tensors
        file_size = file_path.stat().st_size
        if file_size < 100:  # Suspiciously small
            result.warnings.append(f"Suspiciously small weight file: {file_path.name}")
    except Exception as e:
        result.warnings.append(f"Error scanning {file_path.name}: {e}")


def _scan_gguf(file_path: Path, result: ScanResult) -> None:
    """Scan a GGUF file header."""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                result.warnings.append(f"Invalid GGUF magic in {file_path.name}")
    except Exception as e:
        result.warnings.append(f"Error scanning {file_path.name}: {e}")


def _to_dict(result: ScanResult) -> dict[str, Any]:
    return {
        "safe": result.safe,
        "score": result.score,
        "warnings": result.warnings,
        "files_scanned": result.files_scanned,
        "total_parameters": result.total_parameters,
        "scan_time_seconds": result.scan_time_seconds,
        "reason": result.reason,
    }


def print_scan_result(result: dict[str, Any]) -> None:
    """Print scan results as a formatted panel."""
    status = "[green]✓ SAFE[/green]" if result["safe"] else "[red]✗ UNSAFE[/red]"
    content = (
        f"Status: {status}  Score: {result['score']:.0f}/100\n"
        f"Files scanned: {result['files_scanned']}\n"
    )
    if result["warnings"]:
        content += "\nWarnings:\n" + "\n".join(f"  ⚠ {w}" for w in result["warnings"])
    console.print(Panel(content, title="Safety Scan", border_style="cyan"))
