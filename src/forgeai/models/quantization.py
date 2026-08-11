"""
Auto-detection of quantization formats (AWQ, GPTQ).

Identifies quantization type from file signatures, config files,
and file extensions to route models to the vLLM backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from forgeai.core.config import BackendType, QuantizationType

console = Console()


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

    # 1. GGUF file check rejection
    if (path.is_file() and path.suffix.lower() == ".gguf") or (path.is_dir() and list(path.glob("*.gguf"))):
        raise ValueError(
            f"ERROR: GGUF model format is unsupported in ForgeAI v2.0+ (model: {model_path!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )

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
