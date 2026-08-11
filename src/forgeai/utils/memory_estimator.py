"""
VRAM usage prediction algorithms.

Predicts memory requirements based on model architecture, parameter count,
model precision, and KV-cache resource profile. Returns theoretical/planning
estimates and safe max_model_len values to prevent OOM errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

from forgeai.core.resource_profiles import ProfileUnavailableError, get_resource_profile

console = Console()

# Bytes per parameter for each model weight precision
BYTES_PER_PARAM = {
    "float32": 4.0,
    "fp32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}


@dataclass
class VRAMEstimate:
    """Estimated VRAM usage breakdown (Theoretical / Planning Value)."""

    model_params_mb: float
    kv_cache_mb: float
    activation_mb: float
    overhead_mb: float
    total_mb: float
    safe_max_model_len: int
    fits_in_vram: bool
    available_vram_mb: float
    recommended_gpu_memory_utilization: float
    kv_cache_dtype: str = "auto"
    kv_capacity_ratio: float = 1.0
    bytes_per_kv_token: float = 0.0
    is_theoretical_estimate: bool = True

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024


def estimate_vram(
    param_count_billions: float,
    precision: str = "float16",
    max_model_len: int = 4096,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    tensor_parallel_size: int = 1,
    available_vram_mb: float = 24_000,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 4,
    kv_cache_dtype: str = "auto",
) -> VRAMEstimate:
    """
    Estimate VRAM requirements for loading and serving a model.

    Args:
        param_count_billions: Model parameters in billions (e.g., 7.0 for 7B).
        precision: Model weight precision / quantization format.
        max_model_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key-value attention heads.
        head_dim: Dimension of each attention head.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        available_vram_mb: Available VRAM in MB per GPU.
        gpu_memory_utilization: Target GPU memory utilization.
        max_num_seqs: Maximum concurrent sequences.
        kv_cache_dtype: KV cache profile / dtype (auto/bf16, fp8, turboquant_k8v4,
                        turboquant_4bit_nc, turboquant_3bit_nc).

    Returns:
        VRAMEstimate with full breakdown and planning fields.

    Raises:
        ValueError: If kv_cache_dtype is unknown or unsupported.
    """
    # Look up KV cache profile (rejects unknown KV dtypes)
    try:
        kv_profile = get_resource_profile(kv_cache_dtype)
    except ProfileUnavailableError as err:
        raise ValueError(f"Unknown or unsupported KV cache dtype '{kv_cache_dtype}': {err}") from err

    precision_lower = precision.lower()
    bytes_per_param = BYTES_PER_PARAM.get(precision_lower, 2.0)

    # 1. Model weights
    total_params = param_count_billions * 1e9
    params_per_gpu = total_params / tensor_parallel_size
    model_params_mb = (params_per_gpu * bytes_per_param) / (1024 * 1024)

    # 2. KV cache (independent calculation based on KV profile policy)
    # Per token per layer per GPU: 2 vectors (Key + Value) * (num_kv_heads / TP) * head_dim elements
    kv_elements_per_token_per_layer = 2 * (num_kv_heads / tensor_parallel_size) * head_dim
    bytes_per_token_per_layer = kv_elements_per_token_per_layer * kv_profile.bytes_per_element
    bytes_per_kv_token = bytes_per_token_per_layer * num_layers
    kv_total_bytes = bytes_per_kv_token * max_model_len * max_num_seqs
    kv_cache_mb = kv_total_bytes / (1024 * 1024)

    # 3. Activation memory (rough estimate — ~10% of model weights)
    activation_mb = model_params_mb * 0.10

    # 4. Overhead (CUDA context, framework, etc.)
    overhead_mb = 500.0  # ~500MB baseline

    total_mb = model_params_mb + kv_cache_mb + activation_mb + overhead_mb

    # Calculate usable VRAM
    usable_vram_mb = available_vram_mb * gpu_memory_utilization

    # Calculate safe max_model_len
    per_seq_kv_per_token_bytes = bytes_per_kv_token * max_num_seqs
    if per_seq_kv_per_token_bytes > 0:
        remaining_for_kv = max(0.0, usable_vram_mb - model_params_mb - activation_mb - overhead_mb)
        remaining_bytes = remaining_for_kv * 1024 * 1024
        safe_tokens = int(remaining_bytes / per_seq_kv_per_token_bytes)
        safe_max_model_len = max(128, min(safe_tokens, 131072))
    else:
        safe_max_model_len = max_model_len

    fits = total_mb <= usable_vram_mb

    # Calculate recommended utilization
    if available_vram_mb > 0:
        recommended_util = min(0.95, (total_mb / available_vram_mb) + 0.05)
    else:
        recommended_util = 0.85

    return VRAMEstimate(
        model_params_mb=model_params_mb,
        kv_cache_mb=kv_cache_mb,
        activation_mb=activation_mb,
        overhead_mb=overhead_mb,
        total_mb=total_mb,
        safe_max_model_len=safe_max_model_len,
        fits_in_vram=fits,
        available_vram_mb=available_vram_mb,
        recommended_gpu_memory_utilization=recommended_util,
        kv_cache_dtype=kv_profile.kv_cache_dtype,
        kv_capacity_ratio=kv_profile.expected_capacity_ratio,
        bytes_per_kv_token=bytes_per_kv_token,
        is_theoretical_estimate=True,
    )


# Common model presets for quick estimation
MODEL_PRESETS: dict[str, dict] = {
    "llama-7b": {"param_count_billions": 6.7, "num_layers": 32, "num_kv_heads": 32, "head_dim": 128},
    "llama-13b": {"param_count_billions": 13.0, "num_layers": 40, "num_kv_heads": 40, "head_dim": 128},
    "llama-70b": {"param_count_billions": 70.0, "num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
    "mistral-7b": {"param_count_billions": 7.3, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
    "mixtral-8x7b": {"param_count_billions": 46.7, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
    "qwen2-7b": {"param_count_billions": 7.6, "num_layers": 28, "num_kv_heads": 4, "head_dim": 128},
    "qwen2-72b": {"param_count_billions": 72.7, "num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
    "phi-3-mini": {"param_count_billions": 3.8, "num_layers": 32, "num_kv_heads": 8, "head_dim": 96},
    "gemma-7b": {"param_count_billions": 8.5, "num_layers": 28, "num_kv_heads": 1, "head_dim": 256},
}


def estimate_from_preset(
    preset_name: str,
    precision: str = "float16",
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    available_vram_mb: float = 24_000,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 4,
    kv_cache_dtype: str = "auto",
    **kwargs,
) -> VRAMEstimate | None:
    """Estimate VRAM from a known model preset."""
    preset = MODEL_PRESETS.get(preset_name.lower())
    if not preset:
        return None

    return estimate_vram(
        precision=precision,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        available_vram_mb=available_vram_mb,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        **preset,
        **kwargs,
    )


def print_estimate(estimate: VRAMEstimate, model_name: str = "Model") -> None:
    """Display VRAM estimate as a formatted panel."""
    status = "[green]✓ FITS[/green]" if estimate.fits_in_vram else "[red]✗ OOM RISK[/red]"

    content = (
        f"[bold]{model_name}[/bold]  {status}\n"
        f"[dim]Theoretical planning estimate (not measured validation)[/dim]\n\n"
        f"  Model weights:   {estimate.model_params_mb:>10,.0f} MB\n"
        f"  KV cache ({estimate.kv_cache_dtype}, ~{estimate.kv_capacity_ratio:.1f}x): {estimate.kv_cache_mb:>10,.0f} MB\n"
        f"  Activations:     {estimate.activation_mb:>10,.0f} MB\n"
        f"  Overhead:        {estimate.overhead_mb:>10,.0f} MB\n"
        f"  {'─' * 32}\n"
        f"  [bold]Total:           {estimate.total_mb:>10,.0f} MB ({estimate.total_gb:.1f} GB)[/bold]\n"
        f"  Available VRAM:  {estimate.available_vram_mb:>10,.0f} MB\n\n"
        f"  Safe max_model_len:  {estimate.safe_max_model_len:,}\n"
        f"  Recommended util:    {estimate.recommended_gpu_memory_utilization:.0%}"
    )

    console.print(Panel(content, title="VRAM Estimate (Theoretical Planning)", border_style="cyan"))
