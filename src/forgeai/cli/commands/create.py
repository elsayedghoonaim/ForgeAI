"""
forgeai create — Register a local model from a ForgeAI YAML manifest.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

from forgeai.cli.runtime import handle_cli_error
from forgeai.models.manifest import ForgeAIManifest
from forgeai.models.registry import ModelRegistry

console = Console()
app = typer.Typer(invoke_without_command=True)


def _is_ollama_modelfile(content: str) -> bool:
    """Detect if content is an Ollama Modelfile instead of a YAML manifest."""
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return False
    directives = {"from", "parameter", "system", "template", "license", "adapter"}
    first_word = lines[0].split()[0].lower() if lines[0].split() else ""
    if first_word in directives:
        return True
    for line in lines:
        if line.split()[0].lower() in directives:
            return True
    return False


@app.callback(invoke_without_command=True)
def create(
    model: str = typer.Argument(..., help="Model tag name to register"),
    file: Path = typer.Option(
        ...,
        "-f",
        "--file",
        help="Path to ForgeAI YAML manifest file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Create and register a local model tag from a ForgeAI YAML manifest."""
    try:
        content = file.read_text(encoding="utf-8")
    except Exception as err:
        handle_cli_error(f"Failed to read manifest file '{file}': {err}")

    if _is_ollama_modelfile(content):
        handle_cli_error(
            "Ollama Modelfile text is unsupported. Please provide a ForgeAI YAML manifest."
        )

    try:
        yaml_data = yaml.safe_load(content)
    except Exception as err:
        handle_cli_error(f"Invalid YAML content in manifest file '{file}': {err}")

    if not isinstance(yaml_data, dict):
        handle_cli_error(f"Manifest file '{file}' must contain a YAML dictionary.")

    # Explicit GGUF rejection before schema validation
    model_ref = str(yaml_data.get("model") or model).strip()
    model_ref_lower = model_ref.lower()
    model_tag_lower = model.lower()
    if (
        model_ref_lower.endswith(".gguf")
        or ".gguf" in model_ref_lower
        or model_tag_lower.endswith(".gguf")
        or ".gguf" in model_tag_lower
    ):
        handle_cli_error(
            f"ERROR: GGUF model format is unsupported in ForgeAI v2.0+ (model: {model_ref!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )

    # Override public name consistently with MODEL argument
    yaml_data["name"] = model

    try:
        manifest = ForgeAIManifest.model_validate(yaml_data)
    except ValueError as err:
        handle_cli_error(str(err))
    except Exception as err:
        handle_cli_error(f"Manifest validation failed: {err}")

    try:
        registry = ModelRegistry()
        registry.register_manifest(manifest)
    except Exception as err:
        handle_cli_error(f"Failed to register model manifest: {err}")

    console.print(f"Created model '{model}' from '{file}'")
