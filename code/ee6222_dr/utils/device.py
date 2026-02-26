"""Device resolution helpers."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Resolve device string to cpu/cuda."""
    normalized = device.lower().strip()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")

    if normalized == "cpu":
        return "cpu"

    try:
        import torch

        if normalized == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
