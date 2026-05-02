"""
Model discovery and multimodal detection.

Auto-detects model capabilities from HuggingFace Hub configuration,
including multimodal support (vision, audio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

console = Console()


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
        params = (12 * info.hidden_size**2 * info.num_layers +
                  info.vocab_size * info.hidden_size)
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
