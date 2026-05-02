"""
Auto-detection of quantization formats (AWQ, GPTQ, GGUF).

Identifies quantization type from file signatures, config files,
and file extensions to route models to the correct backend.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from forgeai.core.config import BackendType, QuantizationType

console = Console()
GGUF_MAGIC = b"GGUF"


@dataclass
class QuantizationInfo:
    """Detected quantization metadata."""
    format: QuantizationType
    backend: BackendType
    bits: int | None = None
    group_size: int | None = None
    method: str = ""
    source: str = ""


def detect_quantization(model_path: str) -> QuantizationInfo:
    """Auto-detect quantization format from files, config, or naming patterns."""
    path = Path(model_path)

    # 1. GGUF file check
    if path.is_file() and path.suffix == ".gguf":
        return _detect_gguf(path)
    if path.is_dir():
        gguf_files = list(path.glob("*.gguf"))
        if gguf_files:
            return _detect_gguf(gguf_files[0])

    # 2. config.json check
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if config_path.exists():
        info = _detect_from_config(config_path)
        if info:
            return info

    # 3. Filename patterns
    name = path.name.lower() if path.is_file() else str(path).lower()
    if "awq" in name:
        return QuantizationInfo(format=QuantizationType.AWQ, backend=BackendType.VLLM,
                                bits=4, method="awq", source="filename pattern")
    if "gptq" in name:
        return QuantizationInfo(format=QuantizationType.GPTQ, backend=BackendType.VLLM,
                                bits=4, method="gptq", source="filename pattern")

    return QuantizationInfo(format=QuantizationType.NONE, backend=BackendType.VLLM,
                            source="no quantization detected")


def _detect_gguf(file_path: Path) -> QuantizationInfo:
    info = QuantizationInfo(format=QuantizationType.GGUF, backend=BackendType.LLAMA_CPP,
                            source="gguf file signature")
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic == GGUF_MAGIC:
                version = struct.unpack("<I", f.read(4))[0]
                info.method = f"gguf_v{version}"
        name = file_path.name.lower()
        for pattern, bits in [("q4", 4), ("q5", 5), ("q8", 8), ("q3", 3), ("q2", 2), ("f16", 16)]:
            if pattern in name:
                info.bits = bits
                info.method = f"gguf_{pattern}"
                break
    except (OSError, struct.error):
        pass
    return info


def _detect_from_config(config_path: Path) -> QuantizationInfo | None:
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        qc = config.get("quantization_config", {})
        if not qc:
            return None
        method = qc.get("quant_method", "").lower()
        bits = qc.get("bits")
        group_size = qc.get("group_size")
        if method == "awq":
            return QuantizationInfo(format=QuantizationType.AWQ, backend=BackendType.VLLM,
                                    bits=bits, group_size=group_size, method="awq",
                                    source="config.json")
        elif method == "gptq":
            return QuantizationInfo(format=QuantizationType.GPTQ, backend=BackendType.VLLM,
                                    bits=bits, group_size=group_size, method="gptq",
                                    source="config.json")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def print_quantization_info(info: QuantizationInfo) -> None:
    console.print(f"  Format: [bold]{info.format.value}[/bold]  Backend: {info.backend.value}")
    if info.bits:
        console.print(f"  Bits: {info.bits}  Method: {info.method}")
