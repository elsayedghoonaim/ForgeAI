"""
Curated model alias mappings.

Maps short, memorable names to full HuggingFace repository IDs
to reduce user error and improve CLI ergonomics.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

# Curated alias → full repo ID mapping
MODEL_ALIASES: dict[str, str] = {
    # Meta Llama
    "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama3-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama3.2-90b-vision": "meta-llama/Llama-3.2-90B-Vision-Instruct",

    # Mistral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # Qwen
    "qwen2": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2-vl": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
    "qwen-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",

    # Google
    "gemma": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",

    # Microsoft
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "phi3-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi3-vision": "microsoft/Phi-3.5-vision-instruct",

    # DeepSeek
    "deepseek-v2": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-coder": "deepseek-ai/DeepSeek-Coder-V2-Instruct",

    # Nanbeige
    "nanbeige-3b": "Nanbeige/Nanbeige4.1-3B",
    "nanbeige4.1-3b": "Nanbeige/Nanbeige4.1-3B",

    # Coding
    "codellama": "meta-llama/CodeLlama-7b-Instruct-hf",
    "starcoder2": "bigcode/starcoder2-7b",

    # Vision / Multimodal
    "llava": "llava-hf/llava-v1.6-mistral-7b-hf",
    "internvl2": "OpenGVLab/InternVL2-8B",
    "glm-ocr": "THUDM/glm-4v-9b",
}


def resolve_model_name(name: str) -> str:
    """
    Resolve a model alias or short name to a full repo ID.

    If the name contains a '/' it's treated as a direct repo ID.
    Otherwise, it's looked up in the alias mapping.
    """
    if "/" in name:
        return name

    alias = name.lower().strip()
    if alias in MODEL_ALIASES:
        resolved = MODEL_ALIASES[alias]
        console.print(f"[dim]Resolved alias: {name} → {resolved}[/dim]")
        return resolved

    # Return as-is if not found — might be a direct repo ID
    return name


def list_aliases() -> None:
    """Print all available model aliases as a Rich table."""
    table = Table(title="Model Aliases", show_lines=True)
    table.add_column("Alias", style="cyan bold")
    table.add_column("Repository ID", style="white")
    table.add_column("Category", style="dim")

    categories = {
        "llama": "Meta Llama",
        "mistral": "Mistral AI",
        "mixtral": "Mistral AI",
        "qwen": "Qwen",
        "gemma": "Google",
        "phi": "Microsoft",
        "deepseek": "DeepSeek",
        "code": "Coding",
        "star": "Coding",
        "llava": "Vision",
        "intern": "Vision",
        "glm": "Vision",
    }

    for alias, repo_id in sorted(MODEL_ALIASES.items()):
        category = "Other"
        for prefix, cat in categories.items():
            if alias.startswith(prefix):
                category = cat
                break
        table.add_row(alias, repo_id, category)

    console.print(table)
