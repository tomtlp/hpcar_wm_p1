"""Attack-list parsing and label-transition window inference for SWaT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..swat_loader import read_swat_table


def _find_col(columns: list[str], keywords: list[str]) -> str | None:
    lowered = {col.lower().replace(" ", "").replace("_", ""): col for col in columns}
    for key in keywords:
        key_norm = key.lower().replace(" ", "").replace("_", "")
        for norm, original in lowered.items():
            if key_norm in norm:
                return original
    return None


def parse_attack_windows(
    attack_list_file: str | Path | None,
    attack_df: pd.DataFrame | None,
    output_dir: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if attack_list_file and Path(attack_list_file).exists():
        try:
            table = read_swat_table(attack_list_file)
            if all(isinstance(c, int) for c in table.columns) and not table.empty:
                table = table.copy()
                table.columns = [str(v).strip() for v in table.iloc[0].tolist()]
                table = table.iloc[1:].reset_index(drop=True)
            cols = [str(c) for c in table.columns]
            start_col = _find_col(cols, ["starttime", "attackstart", "start"])
            end_col = _find_col(cols, ["endtime", "attackend", "end"])
            target_col = _find_col(cols, ["target", "attackpoint", "affectedtag", "tag"])
            desc_col = _find_col(cols, ["description", "attacktype", "type"])
            id_col = _find_col(cols, ["attacknumber", "attackno", "number", "id"])
            for idx, row in table.iterrows():
                rows.append(
                    {
                        "attack_id": row.get(id_col, idx + 1) if id_col else idx + 1,
                        "start_time": row.get(start_col, pd.NA) if start_col else pd.NA,
                        "end_time": row.get(end_col, pd.NA) if end_col else pd.NA,
                        "start_index": pd.NA,
                        "end_index": pd.NA,
                        "target_tags": row.get(target_col, "unknown") if target_col else "unknown",
                        "description": row.get(desc_col, "") if desc_col else "",
                        "source_file": str(attack_list_file),
                        "alignment_status": "parsed_unaligned",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "attack_id": "parse_error",
                    "start_time": pd.NA,
                    "end_time": pd.NA,
                    "start_index": pd.NA,
                    "end_index": pd.NA,
                    "target_tags": "unknown",
                    "description": f"Could not parse attack list: {exc}",
                    "source_file": str(attack_list_file),
                    "alignment_status": "error",
                }
            )
        if rows and attack_df is not None and "timestamp" in attack_df:
            _align_rows_to_timestamps(rows, attack_df)

    if not rows and attack_df is not None and "label" in attack_df:
        labels = pd.to_numeric(attack_df["label"], errors="coerce")
        is_attack = labels == 1
        values = is_attack.to_numpy()
        idx = 0
        attack_id = 1
        while idx < len(values):
            if not values[idx]:
                idx += 1
                continue
            start = idx
            while idx < len(values) and values[idx]:
                idx += 1
            end = idx - 1
            rows.append(
                {
                    "attack_id": attack_id,
                    "start_time": attack_df["timestamp"].iloc[start] if "timestamp" in attack_df else pd.NA,
                    "end_time": attack_df["timestamp"].iloc[end] if "timestamp" in attack_df else pd.NA,
                    "start_index": int(start),
                    "end_index": int(end),
                    "duration": int(end - start + 1),
                    "target_tags": "unknown",
                    "description": "inferred_from_label_transition",
                    "source_file": "label_column",
                    "alignment_status": "label_aligned",
                    "alignment_status_detail": "label_transition_inside_loaded_range",
                    "exclude_from_eval": False,
                }
            )
            attack_id += 1

    if not rows:
        rows.append(
            {
                "attack_id": "unlabeled_sequence",
                "start_time": pd.NA,
                "end_time": pd.NA,
                "start_index": 0 if attack_df is not None and len(attack_df) else pd.NA,
                "end_index": len(attack_df) - 1 if attack_df is not None and len(attack_df) else pd.NA,
                "duration": len(attack_df) if attack_df is not None else 0,
                "target_tags": "unknown",
                "description": "no_attack_list_or_label",
                "source_file": "",
                "alignment_status": "unlabeled",
                "alignment_status_detail": "no_attack_list_or_label",
                "exclude_from_eval": False,
            }
        )

    window_df = pd.DataFrame(rows)
    window_df.to_csv(Path(output_dir) / "swat_attack_windows.csv", index=False)
    return window_df


def _align_rows_to_timestamps(rows: list[dict[str, Any]], attack_df: pd.DataFrame) -> None:
    timestamps = pd.Series(pd.to_datetime(attack_df["timestamp"], errors="coerce").to_numpy(), index=np.arange(len(attack_df)))
    valid_timestamps = timestamps.dropna()
    if valid_timestamps.empty:
        return
    loaded_start_time = valid_timestamps.min()
    loaded_end_time = valid_timestamps.max()
    loaded_start_text = str(loaded_start_time)
    loaded_end_text = str(loaded_end_time)
    for row in rows:
        row["loaded_start_time"] = loaded_start_text
        row["loaded_end_time"] = loaded_end_text
        row["original_start_time"] = str(row.get("start_time", ""))
        row["original_end_time"] = str(row.get("end_time", ""))
        row["exclude_from_eval"] = False
        row["duration"] = 0
        start = _parse_window_time(row.get("start_time"), loaded_start_time)
        end = _parse_window_time(row.get("end_time"), start)
        if start is None or pd.isna(start) or end is None or pd.isna(end):
            row["alignment_status"] = "timestamp_parse_failed"
            row["alignment_status_detail"] = "could_not_parse_start_or_end_time"
            row["exclude_from_eval"] = True
            print(f"[swat_windows] Excluding attack window {row.get('attack_id')}: timestamp_parse_failed")
            continue
        if end < start:
            start, end = end, start
        if end < loaded_start_time:
            _mark_excluded(row, "out_of_loaded_range_before", start, end)
            continue
        if start > loaded_end_time:
            _mark_excluded(row, "out_of_loaded_range_after", start, end)
            continue

        clipped_start = max(start, loaded_start_time)
        clipped_end = min(end, loaded_end_time)
        status = "inside_loaded_range" if clipped_start == start and clipped_end == end else "partial_overlap"
        overlap = valid_timestamps[(valid_timestamps >= clipped_start) & (valid_timestamps <= clipped_end)]
        if overlap.empty:
            _mark_excluded(row, "no_loaded_rows_overlap", start, end)
            continue

        start_idx = int(overlap.index[0])
        end_idx = int(overlap.index[-1])
        last_idx = int(valid_timestamps.index[-1])
        original_outside = start < loaded_start_time or end > loaded_end_time
        if original_outside and (start_idx >= last_idx - 4 or end_idx >= last_idx - 4) and start > loaded_end_time:
            _mark_excluded(row, "out_of_loaded_range_after", start, end)
            continue

        row["start_index"] = start_idx
        row["end_index"] = max(start_idx, end_idx)
        row["duration"] = int(row["end_index"] - row["start_index"] + 1)
        row["alignment_status"] = status
        row["alignment_status_detail"] = f"{status}: clipped_to_loaded_timestamp_range"


def _parse_window_time(value: Any, date_anchor: Any) -> Any:
    if pd.isna(value):
        return pd.NaT
    if _looks_time_only(value):
        if date_anchor is None or pd.isna(date_anchor):
            return pd.NaT
        combined = f"{pd.Timestamp(date_anchor).date()} {value}"
        return pd.to_datetime(combined, errors="coerce")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed
    if date_anchor is not None and pd.notna(date_anchor):
        combined = f"{pd.Timestamp(date_anchor).date()} {value}"
        return pd.to_datetime(combined, errors="coerce")
    return pd.NaT


def _looks_time_only(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if ":" not in text:
        return False
    return not any(sep in text for sep in ["-", "/", "."])


def _mark_excluded(row: dict[str, Any], status: str, start: Any, end: Any) -> None:
    row["start_index"] = pd.NA
    row["end_index"] = pd.NA
    row["duration"] = 0
    row["alignment_status"] = status
    row["alignment_status_detail"] = f"{status}: original_start={start}, original_end={end}"
    row["exclude_from_eval"] = True
    print(f"[swat_windows] Excluding attack window {row.get('attack_id')}: {status}")
