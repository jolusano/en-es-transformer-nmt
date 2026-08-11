"""Device selection that works unchanged on a MacBook, on Colab and on CPU."""

from __future__ import annotations

import platform

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device.

    Parameters
    ----------
    preference
        ``"auto"`` (default), ``"cuda"``, ``"mps"`` or ``"cpu"``.  ``"auto"``
        prefers CUDA, then Apple's Metal Performance Shaders, then CPU.

    Notes
    -----
    The project is developed on an Apple-silicon MacBook (MPS) and trained on
    Colab (CUDA), so every tensor-creating call site takes its device from this
    single function rather than hard-coding ``"cuda"``.
    """
    preference = preference.lower()

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is visible.")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but Metal backend is unavailable.")

    return torch.device(preference)


def supports_amp(device: torch.device) -> bool:
    """Whether mixed-precision autocasting is worth enabling on ``device``.

    Only CUDA gets a genuine speed-up.  MPS autocast exists but, as of
    PyTorch 2.x, is slower than fp32 for models of this size and occasionally
    produces NaNs in attention softmax, so it is deliberately disabled.
    """
    return device.type == "cuda"


def describe_device(device: torch.device) -> dict[str, object]:
    """Return a JSON-serialisable description of the hardware.

    Stored alongside every run so that timing numbers quoted in the report can
    be attributed to the machine that produced them.
    """
    info: dict[str, object] = {
        "type": device.type,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }

    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        info["name"] = properties.name
        info["total_memory_gb"] = round(properties.total_memory / 1024**3, 2)
        info["capability"] = f"{properties.major}.{properties.minor}"
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU (Metal)"
    else:
        info["name"] = "CPU"

    return info


def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters of ``module``.

    Shared (tied) weights are counted once: iterating ``parameters()`` yields
    each tensor a single time even when several modules reference it.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
