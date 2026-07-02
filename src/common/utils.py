"""Shared utilities for configuration, reproducibility, and output paths."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def set_seed(seed: int) -> None:
    """Set common random seeds. Torch is optional for the physics fallback."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar into a closed interval."""
    return max(low, min(high, value))


def project_root() -> Path:
    """Return the repository root when called from src modules."""
    return Path(__file__).resolve().parents[1]


def output_path(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Resolve the configured output directory."""
    if override is not None:
        return ensure_dir(override)
    raw = config.get("project", {}).get("output_dir", "outputs")
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    return ensure_dir(p)


def as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for CSV-loaded PLC tags."""
    try:
        return float(value)
    except Exception:
        return default


def binary_from_tag(value: Any) -> int:
    """Convert common PLC tag encodings to 0/1."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"open", "opened", "on", "true", "1", "running"}:
            return 1
        if normalized in {"closed", "close", "off", "false", "0", "stopped"}:
            return 0
    return 1 if as_float(value, 0.0) >= 0.5 else 0
