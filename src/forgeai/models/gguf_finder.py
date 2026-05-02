"""HuggingFace GGUF discovery utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()

@dataclass
class GGUFCandidate:
    """A discovered GGUF file candidate."""
    repo_id: str
    filename: str
    quantization: str
    size_mb: float = 0.0

def find_gguf_for_model(model_name: str) -> list[GGUFCandidate]:
    """
    Search HuggingFace for GGUF variants of a model.
    Returns a ranked list of candidates.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        console.print("[red]huggingface_hub is not installed.[/red]")
        return []

    api = HfApi()
    candidates: list[GGUFCandidate] = []
    
    # Extract base name if it's a full repo path
    base_name = model_name.split("/")[-1]
    
    # 1. Check the original repo first
    _search_repo_for_gguf(api, model_name, candidates)
    if candidates:
        return _rank_candidates(candidates)
        
    # 2. Check common GGUF variant repos
    org = model_name.split("/")[0] if "/" in model_name else None
    
    possible_repos = []
    if org:
        possible_repos.append(f"{org}/{base_name}-GGUF")
    possible_repos.append(f"bartowski/{base_name}-GGUF")
    possible_repos.append(f"TheBloke/{base_name}-GGUF")
    possible_repos.append(f"{base_name}-GGUF")
    
    for repo in possible_repos:
        if _search_repo_for_gguf(api, repo, candidates):
            # If we found candidates in one repo, don't keep searching others to save time
            break
            
    return _rank_candidates(candidates)

def _search_repo_for_gguf(api: Any, repo_id: str, candidates: list[GGUFCandidate]) -> bool:
    """Search a specific repo for .gguf files and append to candidates. Returns True if found."""
    try:
        files = api.list_repo_files(repo_id=repo_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        for file in gguf_files:
            # Extract quantization type (usually in the filename like Q4_K_M)
            quant = "unknown"
            parts = file.upper().split(".")
            for part in parts:
                if "Q" in part or "F16" in part or "F32" in part:
                    quant = part
                    break
                    
            candidates.append(GGUFCandidate(
                repo_id=repo_id,
                filename=file,
                quantization=quant
            ))
            
        return len(gguf_files) > 0
    except Exception:
        # Repo might not exist or be private
        return False

def _rank_candidates(candidates: list[GGUFCandidate]) -> list[GGUFCandidate]:
    """Rank candidates preferring Q4_K_M, then other Q4, then Q5, then anything else."""
    
    def score_quant(quant: str) -> int:
        q = quant.upper()
        if "Q4_K_M" in q: return 10
        if "Q4" in q: return 9
        if "Q5_K_M" in q: return 8
        if "Q5" in q: return 7
        if "Q8_0" in q: return 6
        if "Q6" in q: return 5
        if "Q3" in q: return 4
        if "Q2" in q: return 3
        if "F16" in q: return 2
        if "F32" in q: return 1
        return 0
        
    return sorted(candidates, key=lambda c: score_quant(c.quantization), reverse=True)
