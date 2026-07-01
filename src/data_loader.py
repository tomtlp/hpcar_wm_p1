"""Best-effort SWaT CSV loader.

Synthetic mode is the default and mandatory path. CSV mode only attempts to
locate P1-like columns and falls back gracefully when the file is unsuitable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .swat_loader import find_tag_columns


REQUIRED_TAGS = ["LIT101", "FIT101", "MV101", "P101", "P102"]


def find_p1_columns(columns: list[str]) -> dict[str, str]:
    """Case-insensitively map required P1 tag names to CSV columns."""
    return find_tag_columns(columns, REQUIRED_TAGS)


def load_swat_csv(csv_path: str | Path | None) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Load SWaT CSV data if the required P1 columns are available."""
    if csv_path is None:
        print("[data_loader] No CSV path supplied; using synthetic mode.")
        return None, {}
    path = Path(csv_path)
    if not path.exists():
        print(f"[data_loader] CSV not found: {path}. Falling back to synthetic mode.")
        return None, {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[data_loader] Could not read CSV ({exc}). Falling back to synthetic mode.")
        return None, {}

    mapping = find_p1_columns(list(df.columns))
    missing = [tag for tag in REQUIRED_TAGS if tag not in mapping]
    if missing:
        print(
            "[data_loader] Required P1 columns missing "
            f"({', '.join(missing)}). Falling back to synthetic mode."
        )
        return None, mapping
    print(f"[data_loader] Loaded SWaT-like CSV with P1 columns: {mapping}")
    return df, mapping


def swat_initial_level(df: pd.DataFrame | None, mapping: dict[str, str]) -> float | None:
    if df is None or "LIT101" not in mapping or df.empty:
        return None
    try:
        return float(pd.to_numeric(df[mapping["LIT101"]], errors="coerce").dropna().iloc[0])
    except Exception:
        return None


def describe_dataset(df: pd.DataFrame | None, mapping: dict[str, str]) -> dict[str, Any]:
    if df is None:
        return {"mode": "synthetic", "rows": 0, "columns": {}}
    return {"mode": "swat_csv", "rows": int(len(df)), "columns": mapping}
