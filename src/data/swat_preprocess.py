"""Preprocessing utilities for real SWaT P1 logs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..swat_loader import (
    REQUIRED_P1_TAGS,
    detect_label_column,
    detect_timestamp_column,
    find_tag_columns,
)


def parse_label_series(series: pd.Series) -> pd.Series:
    """Map common Normal/Attack or binary labels to 0/1."""
    if series is None:
        return pd.Series(dtype="float")
    values = series.copy()
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        unique = set(numeric.dropna().astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return numeric.astype("float")
    text = values.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index, dtype="float")
    mapped[text.str.contains("attack|abnormal|anomaly|1", regex=True)] = 1.0
    mapped[text.str.contains("normal|benign|0", regex=True)] = 0.0
    return mapped


def label_profile(df: pd.DataFrame, label_col: str | None, output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if label_col is None or label_col not in df:
        profile = pd.DataFrame(
            [
                {
                    "detected_label_column": None,
                    "unique_values": "",
                    "counts": "{}",
                    "attack_ratio": np.nan,
                    "first_attack_index": np.nan,
                    "last_attack_index": np.nan,
                }
            ]
        )
    else:
        labels = parse_label_series(df[label_col])
        attacks = labels == 1
        counts = df[label_col].value_counts(dropna=False).to_dict()
        profile = pd.DataFrame(
            [
                {
                    "detected_label_column": label_col,
                    "unique_values": json.dumps([str(v) for v in df[label_col].dropna().unique()[:20]], ensure_ascii=False),
                    "counts": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
                    "attack_ratio": float(attacks.mean()) if len(attacks) else np.nan,
                    "first_attack_index": int(np.where(attacks.to_numpy())[0][0]) if attacks.any() else np.nan,
                    "last_attack_index": int(np.where(attacks.to_numpy())[0][-1]) if attacks.any() else np.nan,
                }
            ]
        )
    profile.to_csv(output / "swat_label_profile.csv", index=False)
    return profile


def infer_actuator_mapping(
    df: pd.DataFrame,
    tag: str,
    fit_col: str | None,
    configured: dict[str, Any] | None = None,
) -> tuple[dict[Any, int], dict[str, Any]]:
    """Infer binary actuator mapping from config, values, and process evidence."""
    configured = configured or {}
    positive_key = "open_values" if tag == "MV101" else "on_values"
    negative_key = "closed_values" if tag == "MV101" else "off_values"
    positive = set(str(v) for v in configured.get(positive_key, []) or [])
    negative = set(str(v) for v in configured.get(negative_key, []) or [])
    raw = df[tag]
    unique = [v for v in raw.dropna().unique().tolist()]
    mapping: dict[Any, int] = {}
    evidence = "configured"
    warning = ""
    if positive or negative:
        for value in unique:
            if str(value) in positive:
                mapping[value] = 1
            elif str(value) in negative:
                mapping[value] = 0
    if len(mapping) < len(unique):
        evidence = "binary_numeric_or_process_effect"
        if tag == "MV101" and fit_col and fit_col in df and len(unique) >= 2:
            means = df.groupby(tag)[fit_col].mean(numeric_only=True).sort_values()
            if len(means) >= 2:
                closed_value = means.index[0]
                open_value = means.index[-1]
                mapping = {value: int(value == open_value) for value in unique}
                evidence = "higher_mean_FIT101"
        if not mapping and len(unique) == 2:
            ordered = sorted(unique, key=lambda v: str(v))
            mapping = {ordered[0]: 0, ordered[1]: 1}
        elif not mapping:
            warning = "uncertain_nonbinary_values"
            mapping = {value: int(idx == len(unique) - 1) for idx, value in enumerate(sorted(unique, key=lambda v: str(v)))}
    meta = {
        "tag": tag,
        "raw_unique_values": json.dumps([str(v) for v in unique[:30]], ensure_ascii=False),
        "inferred_binary_mapping": json.dumps({str(k): int(v) for k, v in mapping.items()}, ensure_ascii=False),
        "evidence_used": evidence,
        "warning": warning,
    }
    return mapping, meta


def preprocess_swat_dataframe(
    df: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    role: str = "attack",
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Normalize columns, encode labels/actuators, and impute P1 values."""
    cfg = config or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    before_rows = len(df)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()
    df = df.drop_duplicates()
    mapping = find_tag_columns(list(df.columns), cfg.get("required_tags", REQUIRED_P1_TAGS))
    timestamp_col = cfg.get("timestamp_column") or detect_timestamp_column(list(df.columns))
    label_col = cfg.get("label_column") or detect_label_column(list(df.columns))
    if timestamp_col and timestamp_col in df:
        parsed = pd.to_datetime(df[timestamp_col], errors="coerce", format="%d/%m/%Y %I:%M:%S %p")
        if parsed.notna().mean() < 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.5:
            df["_timestamp"] = parsed
            df = df.sort_values("_timestamp")
    if label_col and label_col in df:
        df["_label"] = parse_label_series(df[label_col])
    else:
        df["_label"] = np.nan

    out = pd.DataFrame(index=df.index)
    if "_timestamp" in df:
        out["timestamp"] = df["_timestamp"]
    out["label"] = df["_label"]
    missing_before: dict[str, int] = {}
    for tag, original in mapping.items():
        if tag in {"LIT101", "FIT101"}:
            out[tag] = pd.to_numeric(df[original], errors="coerce")
        else:
            out[tag] = df[original]
        missing_before[tag] = int(out[tag].isna().sum())

    for tag in ["LIT101", "FIT101"]:
        if tag in out:
            out[tag] = out[tag].interpolate(limit=5).ffill().bfill()
    actuator_rows: list[dict[str, Any]] = []
    actuator_cfg = cfg.get("actuator_mapping", {})
    for tag, binary_col in {
        "MV101": "MV101_open_binary",
        "P101": "P101_on_binary",
        "P102": "P102_on_binary",
    }.items():
        if tag not in out:
            continue
        raw_values = out[tag].copy()
        out[f"{tag}_raw"] = raw_values
        out[tag] = out[tag].ffill().bfill()
        mapping_binary, meta = infer_actuator_mapping(out, tag, "FIT101", actuator_cfg.get(tag, {}))
        out[binary_col] = out[tag].map(mapping_binary).astype("float").ffill().bfill().fillna(0).astype(int)
        actuator_rows.append(meta)

    # Compatibility columns expected by existing causal logic.
    if "MV101_open_binary" in out:
        out["mv101_state"] = out["MV101_open_binary"]
        out["mv101_command"] = out["MV101_open_binary"]
    if "P101_on_binary" in out:
        out["p101_state"] = out["P101_on_binary"]
        out["p101_command"] = out["P101_on_binary"]
    if "P102_on_binary" in out:
        out["p102_state"] = out["P102_on_binary"]
        out["p102_command"] = out["P102_on_binary"]
    if "LIT101" in out:
        out["lit101_obs"] = out["LIT101"]
    if "FIT101" in out:
        out["fit101_obs"] = out["FIT101"]
    out = out.reset_index(drop=True)
    out["t"] = np.arange(len(out))

    actuator_df = pd.DataFrame(actuator_rows)
    if not actuator_df.empty:
        actuator_df.to_csv(output / "swat_actuator_mapping.csv", index=False)
    report_rows = []
    for col in sorted(set(mapping) | {"label"}):
        report_rows.append(
            {
                "role": role,
                "column": col,
                "rows_before": before_rows,
                "rows_after": len(out),
                "missing_before": missing_before.get(col, int(df[label_col].isna().sum()) if col == "label" and label_col else np.nan),
                "missing_after": int(out[col].isna().sum()) if col in out else np.nan,
                "dropped_rows": before_rows - len(out),
                "selected_time_range": f"{out['timestamp'].min()}..{out['timestamp'].max()}" if "timestamp" in out else "",
                "selected_p1_columns": json.dumps(mapping, ensure_ascii=False),
            }
        )
    report_df = pd.DataFrame(report_rows)
    report_path = output / "swat_preprocess_report.csv"
    if report_path.exists():
        try:
            report_df = pd.concat([pd.read_csv(report_path), report_df], ignore_index=True)
        except Exception:
            pass
    report_df.to_csv(report_path, index=False)
    label_profile(df, label_col, output)
    return out, mapping, actuator_df
