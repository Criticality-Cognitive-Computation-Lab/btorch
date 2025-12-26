import pytest
import torch


def compile_or_skip(module: torch.nn.Module) -> torch.nn.Module:
    """Compile a module or skip if torch.compile is unavailable."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch build.")
    try:
        return torch.compile(module)
    except Exception as exc:  # pragma: no cover - backend dependent
        pytest.skip(f"torch.compile failed: {exc}")
