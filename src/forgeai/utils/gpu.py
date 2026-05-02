"""
Multi-GPU topology detection, interconnect analysis, and auto tensor parallelism.

Detects hardware layout, NUMA nodes, and PCIe bottlenecks to automatically
calculate the correct --tensor-parallel-size flag. Turns complex tensor
parallelism into a zero-config experience.
"""

from __future__ import annotations

import math
import os
import platform
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
GPU_MEMORY_STARTUP_RESERVE_MB = 128.0
GPU_MEMORY_AUTOTUNE_MARGIN = 0.01


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    index: int
    name: str
    uuid: str = ""
    total_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    temperature_c: int = 0
    utilization_pct: int = 0
    pcie_gen: int = 0
    pcie_width: int = 0
    numa_node: int = -1
    compute_capability: tuple[int, int] = (0, 0)

    @property
    def memory_utilization(self) -> float:
        if self.total_memory_mb > 0:
            return self.used_memory_mb / self.total_memory_mb
        return 0.0


@dataclass
class GPUTopology:
    """Full GPU topology of the system."""
    gpus: list[GPUInfo] = field(default_factory=list)
    driver_version: str = ""
    cuda_version: str = ""
    numa_aware: bool = False
    has_nvlink: bool = False
    recommended_tp_size: int = 1

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_memory_mb(self) -> float:
        return sum(g.total_memory_mb for g in self.gpus)

    @property
    def total_free_memory_mb(self) -> float:
        return sum(g.free_memory_mb for g in self.gpus)


def detect_gpus() -> GPUTopology:
    """
    Detect all GPUs and build a topology map.

    Uses NVIDIA's NVML Python bindings (imported as ``pynvml``) for detection.
    Returns an empty topology gracefully if no GPUs or pynvml is unavailable.
    """
    topology = GPUTopology()

    try:
        import pynvml

        pynvml.nvmlInit()
        topology.driver_version = pynvml.nvmlSystemGetDriverVersion()

        device_count = pynvml.nvmlDeviceGetCount()
        numa_nodes = set()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu = GPUInfo(
                index=i,
                name=pynvml.nvmlDeviceGetName(handle),
                uuid=pynvml.nvmlDeviceGetUUID(handle),
                total_memory_mb=mem_info.total / (1024 * 1024),
                free_memory_mb=mem_info.free / (1024 * 1024),
                used_memory_mb=mem_info.used / (1024 * 1024),
            )

            # Temperature
            import contextlib
            with contextlib.suppress(pynvml.NVMLError):
                gpu.temperature_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu.utilization_pct = util.gpu
            except pynvml.NVMLError:
                pass

            # PCIe info
            try:
                gpu.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                gpu.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            except pynvml.NVMLError:
                pass

            # Compute capability
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                gpu.compute_capability = (major, minor)
            except pynvml.NVMLError:
                pass

            # NUMA node (Linux only)
            if platform.system() == "Linux":
                try:
                    pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
                    numa_path = f"/sys/bus/pci/devices/{pci_bus_id.decode()}/numa_node"
                    if os.path.exists(numa_path):
                        with open(numa_path) as f:
                            gpu.numa_node = int(f.read().strip())
                            numa_nodes.add(gpu.numa_node)
                except Exception:
                    pass

            topology.gpus.append(gpu)

        # Detect NUMA topology
        topology.numa_aware = len(numa_nodes) > 1

        # Detect NVLink (check P2P capability)
        if device_count >= 2:
            try:
                for link_type in range(6):  # NVML_NVLINK_MAX_LINKS
                    try:
                        status = pynvml.nvmlDeviceGetNvLinkState(
                            pynvml.nvmlDeviceGetHandleByIndex(0), link_type
                        )
                        if status:
                            topology.has_nvlink = True
                            break
                    except pynvml.NVMLError:
                        continue
            except Exception:
                pass

        # Calculate recommended tensor parallel size
        topology.recommended_tp_size = _calculate_tp_size(topology)

        pynvml.nvmlShutdown()

    except ImportError:
        console.print("[dim]nvidia-ml-py not installed — GPU detection unavailable[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ GPU detection failed: {e}[/yellow]")

    return topology


def get_target_gpus(topology: GPUTopology, tensor_parallel_size: int = 1) -> list[GPUInfo]:
    """Return the GPUs that would be used for the requested tensor parallel size."""

    if topology.gpu_count == 0:
        return []

    gpu_count = max(1, min(tensor_parallel_size, topology.gpu_count))
    return topology.gpus[:gpu_count]


def recommend_gpu_memory_utilization(
    topology: GPUTopology,
    tensor_parallel_size: int = 1,
    requested_max: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    reserve_mb: float | None = None,
    safety_margin: float | None = None,
) -> float:
    """
    Recommend a safe ``--gpu-util`` value from current free VRAM.

    The calculation is intentionally slightly conservative to avoid hitting the
    exact startup boundary that causes vLLM worker initialization failures.
    """

    target_gpus = get_target_gpus(topology, tensor_parallel_size=tensor_parallel_size)
    if not target_gpus:
        return requested_max

    recommended = requested_max
    for gpu in target_gpus:
        if gpu.total_memory_mb <= 0:
            continue

        effective_reserve_mb = reserve_mb or derive_startup_reserve_mb(gpu)
        effective_safety_margin = safety_margin or derive_startup_safety_margin(gpu)
        safe_fraction = (gpu.free_memory_mb - effective_reserve_mb) / gpu.total_memory_mb
        safe_fraction = min(requested_max, safe_fraction - effective_safety_margin)
        safe_fraction = max(0.10, min(0.95, safe_fraction))
        safe_fraction = math.floor(safe_fraction * 100) / 100
        recommended = min(recommended, safe_fraction)

    return max(0.10, min(0.95, recommended))


def derive_startup_reserve_mb(gpu: GPUInfo) -> float:
    """Estimate non-model startup headroom from the GPU size."""

    return max(
        GPU_MEMORY_STARTUP_RESERVE_MB,
        gpu.total_memory_mb * 0.05,
    )


def derive_startup_safety_margin(gpu: GPUInfo) -> float:
    """Estimate a utilization margin from the GPU size and reserve."""

    if gpu.total_memory_mb <= 0:
        return GPU_MEMORY_AUTOTUNE_MARGIN

    reserve_fraction = derive_startup_reserve_mb(gpu) / gpu.total_memory_mb
    return max(
        GPU_MEMORY_AUTOTUNE_MARGIN,
        reserve_fraction / 4,
    )


def _calculate_tp_size(topology: GPUTopology) -> int:
    """
    Calculate optimal tensor parallel size based on topology.

    Rules:
    1. Default to 1 (single GPU).
    2. Use all GPUs if they are homogeneous and on the same NUMA node (or NVLink connected).
    3. If GPUs span NUMA nodes without NVLink, warn and limit to GPUs on one node.
    4. Round down to nearest power of 2.
    """
    if topology.gpu_count <= 1:
        return 1

    # Check GPU homogeneity
    names = {g.name for g in topology.gpus}
    if len(names) > 1:
        console.print(
            "[yellow]⚠ Heterogeneous GPUs detected. Using single GPU for safety.[/yellow]"
        )
        return 1

    # NUMA-aware decision
    if topology.numa_aware and not topology.has_nvlink:
        # Group GPUs by NUMA node — use the largest group
        numa_groups: dict[int, list[GPUInfo]] = {}
        for gpu in topology.gpus:
            numa_groups.setdefault(gpu.numa_node, []).append(gpu)
        largest_group = max(numa_groups.values(), key=len)
        tp = len(largest_group)
        console.print(
            f"[yellow]⚠ Cross-NUMA detected without NVLink. "
            f"Limiting TP to {tp} GPUs on NUMA node {largest_group[0].numa_node}[/yellow]"
        )
    else:
        tp = topology.gpu_count

    # Round down to power of 2
    power = 1
    while power * 2 <= tp:
        power *= 2

    return power


def print_gpu_table(topology: GPUTopology) -> None:
    """Display GPU topology as a Rich table."""
    if not topology.gpus:
        console.print("[yellow]No GPUs detected.[/yellow]")
        return

    table = Table(title="GPU Topology", show_lines=True)
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Name", style="bold")
    table.add_column("Memory", justify="right")
    table.add_column("Used", justify="right")
    table.add_column("Free", justify="right")
    table.add_column("Temp", justify="center")
    table.add_column("Util", justify="center")
    table.add_column("NUMA", justify="center")
    table.add_column("PCIe", justify="center")

    for gpu in topology.gpus:
        mem_pct = gpu.memory_utilization * 100
        table.add_row(
            str(gpu.index),
            gpu.name,
            f"{gpu.total_memory_mb:,.0f} MB",
            f"{gpu.used_memory_mb:,.0f} MB ({mem_pct:.0f}%)",
            f"{gpu.free_memory_mb:,.0f} MB",
            f"{gpu.temperature_c}°C" if gpu.temperature_c else "N/A",
            f"{gpu.utilization_pct}%" if gpu.utilization_pct else "N/A",
            str(gpu.numa_node) if gpu.numa_node >= 0 else "N/A",
            f"Gen{gpu.pcie_gen} x{gpu.pcie_width}" if gpu.pcie_gen else "N/A",
        )

    console.print(table)
    console.print(f"Driver: {topology.driver_version}  |  "
                  f"NVLink: {'Yes' if topology.has_nvlink else 'No'}  |  "
                  f"Recommended TP: {topology.recommended_tp_size}")
