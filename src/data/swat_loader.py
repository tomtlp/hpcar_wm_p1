"""Real SWaT file discovery and robust P1 column mapping."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_P1_TAGS = ["LIT101", "FIT101", "MV101", "P101", "P102"]
LABEL_CANDIDATES = ["normalattack", "normal_attack", "label", "attack", "class", "y", "marker"]
TIMESTAMP_CANDIDATES = ["timestamp", "time", "date", "datetime"]
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def normalize_column_name(name: Any) -> str:
    """Normalize a SWaT column name for fuzzy matching while preserving digits."""
    text = str(name).strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^A-Za-z0-9]+", "", text).upper()


def compact_name(name: Any) -> str:
    return normalize_column_name(name).lower()


def find_tag_columns(columns: list[Any], required_tags: list[str] | None = None) -> dict[str, str]:
    """Find P1 tag columns despite whitespace, underscores, and case variants."""
    required = required_tags or REQUIRED_P1_TAGS
    normalized_to_original: dict[str, str] = {normalize_column_name(col): str(col) for col in columns}
    mapping: dict[str, str] = {}
    for tag in required:
        tag_norm = normalize_column_name(tag)
        if tag_norm in normalized_to_original:
            mapping[tag] = normalized_to_original[tag_norm]
            continue
        matches = [original for norm, original in normalized_to_original.items() if tag_norm in norm]
        if matches:
            mapping[tag] = matches[0]
    return mapping


def detect_timestamp_column(columns: list[Any]) -> str | None:
    for col in columns:
        norm = compact_name(col)
        if any(candidate in norm for candidate in TIMESTAMP_CANDIDATES):
            return str(col)
    return None


def detect_label_column(columns: list[Any]) -> str | None:
    for col in columns:
        norm = compact_name(col)
        if norm in LABEL_CANDIDATES or any(candidate == norm for candidate in LABEL_CANDIDATES):
            return str(col)
    for col in columns:
        norm = compact_name(col)
        if "normal" in norm and "attack" in norm:
            return str(col)
    return None


def detect_file_role(path: Path) -> str:
    name = path.name.lower()
    if "list" in name and "attack" in name:
        return "attack_list"
    if "attack" in name and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return "attack_csv"
    if "normal" in name and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return "normal_csv"
    return "unknown"


def discover_swat_files(
    swat_dir: str | Path,
    output_dir: str | Path,
    max_rows: int | None = 20000,
) -> pd.DataFrame:
    """Recursively inventory SWaT-like files under a directory."""
    root = Path(swat_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        df = pd.DataFrame(
            columns=[
                "path",
                "extension",
                "size",
                "rows",
                "columns",
                "role",
                "timestamp_column",
                "label_column",
                "p1_columns",
                "warning",
            ]
        )
        df.to_csv(output / "swat_file_inventory.csv", index=False)
        return df

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        role = detect_file_role(path)
        row: dict[str, Any] = {
            "path": str(path),
            "extension": ext,
            "size": int(path.stat().st_size),
            "rows": pd.NA,
            "columns": pd.NA,
            "role": role,
            "timestamp_column": None,
            "label_column": None,
            "p1_columns": "{}",
            "warning": "",
        }
        if ext in SUPPORTED_EXTENSIONS:
            try:
                preview = read_swat_table(path, max_rows=max_rows, preview=True)
                row["rows"] = int(len(preview))
                row["columns"] = int(len(preview.columns))
                row["timestamp_column"] = detect_timestamp_column(list(preview.columns))
                row["label_column"] = detect_label_column(list(preview.columns))
                row["p1_columns"] = json.dumps(find_tag_columns(list(preview.columns)), ensure_ascii=False)
            except Exception as exc:
                row["warning"] = f"unreadable: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output / "swat_file_inventory.csv", index=False)
    return df


def read_swat_table(path: str | Path, max_rows: int | None = None, preview: bool = False) -> pd.DataFrame:
    """Read CSV/XLS/XLSX with defensive defaults."""
    p = Path(path)
    nrows = 200 if preview else max_rows
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, nrows=nrows, low_memory=False)
        return _repair_header_if_needed(df)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p, nrows=nrows)
        repaired = _repair_header_if_needed(df)
        if find_tag_columns(list(repaired.columns)):
            return repaired
        raw = pd.read_excel(p, nrows=nrows, header=None)
        return _header_from_data_rows(raw)
    raise ValueError(f"Unsupported SWaT file extension: {p.suffix}")


def _repair_header_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if find_tag_columns(list(df.columns)):
        return df
    return _header_from_data_rows(df.reset_index(drop=True))


def _header_from_data_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    scan_rows = min(10, len(raw))
    best_idx = None
    best_score = 0
    for idx in range(scan_rows):
        values = [str(v) for v in raw.iloc[idx].tolist()]
        score = len(find_tag_columns(values))
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is None or best_score < 3:
        return raw
    columns = [str(v).strip() if str(v) != "nan" else f"unnamed_{i}" for i, v in enumerate(raw.iloc[best_idx].tolist())]
    repaired = raw.iloc[best_idx + 1 :].copy()
    repaired.columns = columns
    repaired = repaired.reset_index(drop=True)
    return repaired


def choose_role_file(
    inventory: pd.DataFrame,
    role: str,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    if inventory.empty or "role" not in inventory:
        return None
    candidates = inventory[inventory["role"] == role].copy()
    if candidates.empty:
        return None
    candidates["size"] = pd.to_numeric(candidates["size"], errors="coerce").fillna(0)
    return Path(str(candidates.sort_values("size", ascending=False).iloc[0]["path"]))


def write_column_mapping(
    output_dir: str | Path,
    normal_mapping: dict[str, str],
    attack_mapping: dict[str, str],
    original_columns: dict[str, list[str]] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "normal": normal_mapping,
        "attack": attack_mapping,
        "original_columns": original_columns or {},
    }
    path = output / "swat_column_mapping.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_missing_columns(output_dir: str | Path, missing: list[str]) -> Path:
    path = Path(output_dir) / "swat_missing_columns.txt"
    path.write_text(
        "Missing required P1 columns: " + ", ".join(missing) + "\n",
        encoding="utf-8",
    )
    return path
