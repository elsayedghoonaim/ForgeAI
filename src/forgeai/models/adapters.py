"""
Dynamic LoRA / PEFT adapter hot-swapping.

Allows loading and unloading fine-tuned adapters on a running engine
without restarting or spinning up new instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class AdapterInfo:
    """Metadata about a loaded adapter."""
    name: str
    path: str
    adapter_type: str = "lora"  # lora, peft, qlora
    rank: int = 0
    target_modules: list[str] = field(default_factory=list)
    is_active: bool = False


class AdapterManager:
    """
    Manages dynamic LoRA/PEFT adapter loading and unloading.

    Enables running multiple fine-tuned adapters on the same base model
    simultaneously (e.g., agriculture Q&A + satellite telemetry).
    """

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterInfo] = {}
        self._engine: Any = None

    def set_engine(self, engine: Any) -> None:
        """Attach to an inference engine."""
        self._engine = engine

    def load_adapter(
        self,
        name: str,
        path: str,
        adapter_type: str = "lora",
    ) -> AdapterInfo:
        """
        Load a LoRA/PEFT adapter onto the running engine.

        Args:
            name: Unique identifier for this adapter.
            path: Path to adapter weights (local or HF repo).
            adapter_type: Type of adapter (lora, peft, qlora).

        Returns:
            AdapterInfo for the loaded adapter.
        """
        if name in self._adapters:
            console.print(f"[yellow]⚠ Adapter '{name}' already loaded. Replacing.[/yellow]")
            self.unload_adapter(name)

        info = AdapterInfo(name=name, path=path, adapter_type=adapter_type, is_active=True)

        # Attempt to detect adapter config
        try:
            import json
            from pathlib import Path as AdapterPath
            config_path = AdapterPath(path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                info.rank = config.get("r", config.get("lora_rank", 0))
                info.target_modules = config.get("target_modules", [])
        except Exception:
            pass

        # Load into engine if available
        if self._engine is not None:
            try:
                if hasattr(self._engine, "load_lora"):
                    self._engine.load_lora(path)
                elif hasattr(self._engine, "add_adapter"):
                    self._engine.add_adapter(name, path)
            except Exception as e:
                console.print(f"[yellow]⚠ Engine adapter loading failed: {e}[/yellow]")

        self._adapters[name] = info
        console.print(f"[green]✓[/green] Loaded adapter: [bold]{name}[/bold] ({adapter_type})")
        return info

    def unload_adapter(self, name: str) -> bool:
        """Unload an adapter from the engine."""
        if name not in self._adapters:
            console.print(f"[yellow]⚠ Adapter '{name}' not found.[/yellow]")
            return False

        if self._engine is not None:
            try:
                if hasattr(self._engine, "remove_adapter"):
                    self._engine.remove_adapter(name)
            except Exception as e:
                console.print(f"[yellow]⚠ Engine adapter unloading failed: {e}[/yellow]")

        del self._adapters[name]
        console.print(f"[green]✓[/green] Unloaded adapter: {name}")
        return True

    def list_adapters(self) -> list[AdapterInfo]:
        """List all loaded adapters."""
        return list(self._adapters.values())

    def get_adapter(self, name: str) -> AdapterInfo | None:
        """Get info about a specific adapter."""
        return self._adapters.get(name)

    def print_adapters(self) -> None:
        """Display loaded adapters as a table."""
        if not self._adapters:
            console.print("[dim]No adapters loaded.[/dim]")
            return

        table = Table(title="Loaded Adapters", show_lines=True)
        table.add_column("Name", style="cyan bold")
        table.add_column("Type", style="white")
        table.add_column("Rank")
        table.add_column("Targets")
        table.add_column("Active", justify="center")

        for adapter in self._adapters.values():
            table.add_row(
                adapter.name,
                adapter.adapter_type,
                str(adapter.rank) if adapter.rank else "—",
                ", ".join(adapter.target_modules[:3]) or "—",
                "✓" if adapter.is_active else "✗",
            )
        console.print(table)
