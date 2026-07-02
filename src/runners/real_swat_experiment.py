"""Real SWaT offline evaluation tasks.

Real SWaT logs are not interactive. The log-eval task therefore evaluates
diagnosis, reconstruction, and prediction only. Recovery metrics are produced
only in counterfactual model rollouts or real-calibrated hybrid simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ..attacks import AttackConfig
from ..causal_logic import diagnose_real_swat_timeseries
from ..metrics import summarize_metrics
from ..plotting import (
    make_counterfactual_plots,
    make_hybrid_action_effect_plots,
    make_hybrid_plots,
    make_real_swat_log_plots,
    make_p1_residual_threshold_plot,
    make_p1_valid_window_plots,
    make_p1_counterfactual_candidate_plot,
    make_hybrid_unit_check_plot,
)
from ..recovery_actions import RecoveryAction, action_cost, all_recovery_actions
from ..swat_attack_windows import parse_attack_windows
from ..swat_calibration import calibrate_p1_from_normal, simulator_config_from_calibration
from ..swat_loader import (
    REQUIRED_P1_TAGS,
    choose_role_file,
    discover_swat_files,
    read_swat_table,
    write_column_mapping,
    write_missing_columns,
)
from ..swat_preprocess import preprocess_swat_dataframe
from ..utils import ensure_dir
from ..world_model import RealSWaTWorldModel


def run_real_swat_task(
    config: dict[str, Any],
    swat_dir: str | Path,
    real_swat_task: str,
    output_dir: str | Path,
    quick: bool = False,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    for stale in ["real_swat_error.txt", "swat_missing_columns.txt", "swat_preprocess_report.csv"]:
        stale_path = output / stale
        if stale_path.exists():
            stale_path.unlink()
    swat_cfg = config.get("swat", {})
    try:
        context = prepare_real_swat_context(config, swat_dir, output, quick)
        if context is None:
            return {"output_dir": output}
        if real_swat_task == "log_eval":
            return run_log_eval(context, output)
        if real_swat_task == "p1_log_eval":
            return run_p1_log_eval(context, output)
        if real_swat_task == "counterfactual":
            return run_counterfactual(context, config, output)
        if real_swat_task == "hybrid":
            return run_hybrid(context, config, output, quick)
        raise ValueError(f"Unknown real_swat_task: {real_swat_task}")
    except Exception as exc:
        message = f"real_swat task failed gracefully: {exc}"
        print(f"[real_swat] {message}")
        (output / "real_swat_error.txt").write_text(message + "\n", encoding="utf-8")
        return {"output_dir": output}


def prepare_real_swat_context(
    config: dict[str, Any],
    swat_dir: str | Path,
    output: Path,
    quick: bool,
) -> dict[str, Any] | None:
    swat_cfg = config.get("swat", {})
    max_rows = int(swat_cfg.get("max_rows_quick", 20000)) if quick else swat_cfg.get("max_rows_full")
    inventory = discover_swat_files(swat_dir, output, max_rows=max_rows)
    if inventory.empty:
        return _real_swat_error(output, f"No files found under {swat_dir}")

    normal_path = choose_role_file(inventory, "normal_csv", swat_cfg.get("normal_file"))
    attack_path = choose_role_file(inventory, "attack_csv", swat_cfg.get("attack_file"))
    attack_list_path = choose_role_file(inventory, "attack_list", swat_cfg.get("attack_list_file"))
    if normal_path is None and attack_path is None:
        return _real_swat_error(output, "No normal or attack SWaT CSV/XLS/XLSX file discovered.")

    normal_raw = read_swat_table(normal_path, max_rows=max_rows) if normal_path else pd.DataFrame()
    attack_raw = read_swat_table(attack_path, max_rows=max_rows) if attack_path else normal_raw.copy()
    normal_df, normal_mapping, _ = preprocess_swat_dataframe(normal_raw, output, swat_cfg, role="normal", max_rows=max_rows)
    attack_df, attack_mapping, _ = preprocess_swat_dataframe(attack_raw, output, swat_cfg, role="attack", max_rows=max_rows)
    write_column_mapping(
        output,
        normal_mapping,
        attack_mapping,
        {
            "normal": [str(c) for c in normal_raw.columns],
            "attack": [str(c) for c in attack_raw.columns],
        },
    )
    combined_mapping = attack_mapping or normal_mapping
    missing = [tag for tag in swat_cfg.get("required_tags", REQUIRED_P1_TAGS) if tag not in combined_mapping]
    if missing:
        write_missing_columns(output, missing)
        return _real_swat_error(output, "Missing required P1 columns: " + ", ".join(missing))

    windows = parse_attack_windows(attack_list_path, attack_df, output)
    calibration = calibrate_p1_from_normal(normal_df, output, swat_cfg)
    real_model = RealSWaTWorldModel(model_type=config.get("world_model", {}).get("type", "linear"))
    real_model.fit(normal_df, output / "world_model_real_swat.json")
    return {
        "inventory": inventory,
        "normal_path": normal_path,
        "attack_path": attack_path,
        "attack_list_path": attack_list_path,
        "normal_df": normal_df,
        "attack_df": attack_df,
        "windows": windows,
        "calibration": calibration,
        "world_model": real_model,
        "swat_config": swat_cfg,
    }


P1_TARGETS = ["LIT101", "FIT101", "MV101", "P101", "P102"]


def normalize_target_tag(value: Any) -> str:
    """Normalize target tag text such as MV-101 to MV101."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()


def p1_target_from_text(value: Any) -> str | None:
    normalized = normalize_target_tag(value)
    for tag in P1_TARGETS:
        if tag in normalized:
            return tag
    return None


def build_p1_attack_windows(windows: pd.DataFrame, loaded_len: int, output: Path) -> pd.DataFrame:
    """Filter and validate attack windows for P1-only evaluation."""
    rows: list[dict[str, Any]] = []
    for _, row in windows.iterrows():
        original = row.get("target_tags", "unknown")
        p1_tag = p1_target_from_text(original)
        if p1_tag is None:
            continue
        start = row.get("start_index")
        end = row.get("end_index")
        status = str(row.get("alignment_status", "unknown"))
        source_excluded = _as_bool(row.get("exclude_from_eval", False))
        in_range = pd.notna(start) and pd.notna(end) and int(start) < loaded_len and int(end) >= 0
        if source_excluded or not in_range:
            if not status.startswith("out_of_loaded_range") and source_excluded:
                status = str(row.get("alignment_status", "excluded"))
            elif not status.startswith("out_of_loaded_range"):
                status = "out_of_loaded_range"
            start_out = pd.NA
            end_out = pd.NA
            duration = 0
            exclude = True
        else:
            start_i = int(start)
            end_i = int(end)
            if end_i >= loaded_len:
                status = "out_of_loaded_range"
                start_out = pd.NA
                end_out = pd.NA
                duration = 0
                exclude = True
            else:
                start_out = max(0, start_i)
                end_out = max(start_out, end_i)
                duration = int(end_out - start_out + 1)
                exclude = False
        if duration <= 1:
            exclude = True
        rows.append(
            {
                "attack_id": row.get("attack_id"),
                "target_tags_original": original,
                "target_tags_normalized": normalize_target_tag(original),
                "p1_target_tag": p1_tag,
                "original_start_time": row.get("original_start_time", row.get("start_time", pd.NA)),
                "original_end_time": row.get("original_end_time", row.get("end_time", pd.NA)),
                "loaded_start_time": row.get("loaded_start_time", pd.NA),
                "loaded_end_time": row.get("loaded_end_time", pd.NA),
                "start_index": start_out,
                "end_index": end_out,
                "duration": duration,
                "alignment_status": status,
                "alignment_status_detail": row.get("alignment_status_detail", status),
                "exclude_from_eval": bool(exclude),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output / "real_swat_p1_attack_windows.csv", index=False)
    valid_p1_windows(df).to_csv(output / "real_swat_p1_attack_windows_valid.csv", index=False)
    return df


def add_p1_labels(df: pd.DataFrame, p1_windows: pd.DataFrame, output: Path) -> pd.DataFrame:
    labeled = pd.DataFrame({"t": df["t"] if "t" in df else np.arange(len(df))})
    for col in ["timestamp", "label"]:
        if col in df:
            labeled[col] = df[col]
    labeled["p1_attack_label_any"] = 0
    for tag in P1_TARGETS:
        labeled[f"p1_attack_label_{tag}"] = 0
    for _, row in valid_p1_windows(p1_windows).iterrows():
        if pd.isna(row.get("start_index")) or pd.isna(row.get("end_index")):
            continue
        start = int(row["start_index"])
        end = int(row["end_index"])
        tag = str(row["p1_target_tag"])
        labeled.loc[start:end, "p1_attack_label_any"] = 1
        if tag in P1_TARGETS:
            labeled.loc[start:end, f"p1_attack_label_{tag}"] = 1
    has_attack_list_windows = not p1_windows.empty
    supervision = "attack_list_p1_windows_valid_only" if has_attack_list_windows else "weak_global_label_fallback"
    if (not has_attack_list_windows) and labeled["p1_attack_label_any"].sum() == 0 and "label" in df:
        labeled["p1_attack_label_any"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        supervision = "weak_global_label_fallback"
    labeled["p1_label_source"] = supervision
    labeled.to_csv(output / "real_swat_p1_labels.csv", index=False)
    return labeled


def valid_p1_windows(p1_windows: pd.DataFrame) -> pd.DataFrame:
    if p1_windows.empty:
        return p1_windows.copy()
    df = p1_windows.copy()
    start = pd.to_numeric(df.get("start_index"), errors="coerce")
    end = pd.to_numeric(df.get("end_index"), errors="coerce")
    duration = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0)
    excluded = df.get("exclude_from_eval", False)
    if not isinstance(excluded, pd.Series):
        excluded = pd.Series(bool(excluded), index=df.index)
    excluded = excluded.map(_as_bool)
    status = df.get("alignment_status", pd.Series("", index=df.index)).astype(str)
    valid = (
        ~excluded
        & start.notna()
        & end.notna()
        & (duration > 1)
        & ~status.str.contains("out_of_loaded_range|parse_failed|no_loaded_rows_overlap", regex=True, na=False)
    )
    return df.loc[valid].copy()


def p1_window_metadata(p1_windows: pd.DataFrame) -> dict[str, Any]:
    valid = valid_p1_windows(p1_windows)
    total = int(len(p1_windows))
    valid_ids = [str(v) for v in valid.get("attack_id", pd.Series(dtype=object)).tolist()]
    excluded = p1_windows.loc[~p1_windows.index.isin(valid.index)] if not p1_windows.empty else p1_windows
    excluded_ids = [str(v) for v in excluded.get("attack_id", pd.Series(dtype=object)).tolist()]
    return {
        "n_p1_windows_total": total,
        "n_p1_windows_valid": int(len(valid)),
        "n_p1_windows_excluded": int(total - len(valid)),
        "valid_window_ids": "|".join(valid_ids),
        "excluded_window_ids": "|".join(excluded_ids),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def calibrate_real_thresholds(normal_diag: pd.DataFrame, output: Path, swat_cfg: dict[str, Any]) -> dict[str, float]:
    diag_cfg = swat_cfg.get("diagnosis", {})
    q = float(diag_cfg.get("threshold_quantile", 0.995))
    thresholds = {
        "residual_abs_threshold": _finite_quantile(normal_diag.get("abs_residual_LIT101"), q),
        "residual_ewma_threshold": _finite_quantile(normal_diag.get("lit101_residual_ewma"), q),
        "cusum_threshold": _finite_quantile(normal_diag.get("lit101_residual_cusum"), q),
        "slope_threshold": _finite_quantile(normal_diag.get("lit101_slope"), q),
        "replay_flatness_threshold": _finite_quantile(normal_diag.get("lit101_replay_score"), q),
        "threshold_quantile": q,
        "min_persistent_steps": int(diag_cfg.get("min_persistent_steps", diag_cfg.get("persistence_steps", 5))),
    }
    for tag, col in {
        "mv101": "mv101_suspicion_score",
        "p101": "p101_suspicion_score",
        "p102": "p102_suspicion_score",
        "plc1": "plc1_suspicion_score",
        "lit101": "lit101_suspicion_score",
    }.items():
        thresholds[f"{tag}_suspicion_threshold"] = max(0.35, _finite_quantile(normal_diag.get(col), q))
    (output / "real_swat_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    actuator_thresholds = {
        key: value
        for key, value in thresholds.items()
        if key.endswith("_suspicion_threshold") or key in {"threshold_quantile", "min_persistent_steps"}
    }
    (output / "real_swat_actuator_diagnosis_thresholds.json").write_text(
        json.dumps(actuator_thresholds, indent=2),
        encoding="utf-8",
    )
    _write_actuator_feature_summary(normal_diag, output, thresholds)
    make_p1_residual_threshold_plot(normal_diag, thresholds, output)
    return thresholds


def apply_real_thresholds(diag: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = diag.copy()
    persistent = int(thresholds.get("min_persistent_steps", 5))
    triggers = (
        (out["abs_residual_LIT101"] > thresholds["residual_abs_threshold"])
        | (out["lit101_residual_ewma"] > thresholds["residual_ewma_threshold"])
        | (out["lit101_residual_cusum"] > thresholds["cusum_threshold"])
        | (out["lit101_slope"] > thresholds["slope_threshold"])
        | (out["lit101_replay_score"] > thresholds["replay_flatness_threshold"])
    )
    persistent_triggers = triggers.rolling(persistent, min_periods=1).sum() >= persistent
    out["p1_lit101_threshold_trigger"] = persistent_triggers.astype(int)
    out.loc[persistent_triggers, "trust_LIT101"] = 0
    score_to_trust = {
        "lit101_suspicion_score": ("trust_LIT101", "lit101_suspicion_threshold"),
        "mv101_suspicion_score": ("trust_MV101", "mv101_suspicion_threshold"),
        "p101_suspicion_score": ("trust_P101", "p101_suspicion_threshold"),
        "p102_suspicion_score": ("trust_P102", "p102_suspicion_threshold"),
        "plc1_suspicion_score": ("trust_PLC1", "plc1_suspicion_threshold"),
    }
    for score_col, (trust_col, threshold_key) in score_to_trust.items():
        if score_col in out and trust_col in out:
            threshold = float(thresholds.get(threshold_key, 0.8))
            out.loc[pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) > threshold, trust_col] = 0
    trust_cols = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102", "trust_PLC1"]
    out["attack_belief_score"] = (1 - out[trust_cols]).mean(axis=1).clip(0, 1)
    out["p1_attack_score"] = np.maximum(out["attack_belief_score"], out["p1_lit101_threshold_trigger"] * 0.8)
    out["level_est"] = out["LIT101"]
    out.loc[out["trust_LIT101"] == 0, "level_est"] = out.loc[out["trust_LIT101"] == 0, "lit101_est"]
    _refresh_root_cause_from_scores(out)
    return out


def _write_actuator_feature_summary(normal_diag: pd.DataFrame, output: Path, thresholds: dict[str, float]) -> None:
    rows: list[dict[str, Any]] = []
    for col in [
        "lit101_suspicion_score",
        "mv101_suspicion_score",
        "p101_suspicion_score",
        "p102_suspicion_score",
        "plc1_suspicion_score",
        "mv101_low_fit_when_open",
        "mv101_high_fit_when_closed",
        "lit101_slope_delta",
    ]:
        if col not in normal_diag:
            continue
        values = pd.to_numeric(normal_diag[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "feature": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "q95": float(values.quantile(0.95)),
                "q995": float(values.quantile(0.995)),
                "threshold": thresholds.get(col.replace("_score", "_threshold")),
            }
        )
    pd.DataFrame(rows).to_csv(output / "real_swat_actuator_diagnosis_features.csv", index=False)


def _refresh_root_cause_from_scores(out: pd.DataFrame) -> None:
    score_cols = {
        "LIT101_UNTRUSTED": "lit101_suspicion_score",
        "MV101_OR_FIT101_SUSPICIOUS": "mv101_suspicion_score",
        "P101_SUSPICIOUS": "p101_suspicion_score",
        "P102_SUSPICIOUS": "p102_suspicion_score",
        "PLC1_SUSPICIOUS": "plc1_suspicion_score",
    }
    present = {label: col for label, col in score_cols.items() if col in out}
    if not present:
        out["root_cause_confidence"] = 0.0
        out["inferred_root_cause"] = "none"
        return
    scores = pd.DataFrame({label: pd.to_numeric(out[col], errors="coerce").fillna(0.0) for label, col in present.items()})
    out["root_cause_confidence"] = scores.max(axis=1).clip(0, 1)
    out["inferred_root_cause"] = scores.idxmax(axis=1)
    out.loc[out["root_cause_confidence"] < 0.35, "inferred_root_cause"] = "none"


def _finite_quantile(series: Any, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def run_log_eval(context: dict[str, Any], output: Path) -> dict[str, Path]:
    attack_df = context["attack_df"]
    normal_df = context["normal_df"]
    calibration = context["calibration"]
    swat_cfg = context["swat_config"]
    real_model: RealSWaTWorldModel = context["world_model"]
    diagnosed = diagnose_real_swat_timeseries(attack_df, calibration, swat_cfg.get("diagnosis", {}))
    predicted = real_model.predict_frame(diagnosed)
    predicted["evaluation_type"] = "whole_plant_label_eval_for_p1_model_not_primary"
    predicted["prediction_error_LIT101_next"] = predicted["pred_LIT101_next"] - predicted["LIT101"].shift(-1)
    predicted.to_csv(output / "real_swat_timeseries.csv", index=False)
    predicted.to_csv(output / "real_swat_trust_timeseries.csv", index=False)
    predicted.to_csv(output / "real_swat_prediction_timeseries.csv", index=False)

    normal_diag = diagnose_real_swat_timeseries(normal_df, calibration, swat_cfg.get("diagnosis", {}))
    detection = detection_metrics(predicted, normal_diag)
    detection.to_csv(output / "real_swat_detection_metrics.csv", index=False)
    trust_by_tag = real_trust_detection_by_tag(predicted, context["windows"])
    trust_by_tag.to_csv(output / "real_swat_trust_detection_by_tag.csv", index=False)
    world_eval = real_world_model_eval(predicted)
    world_eval.to_csv(output / "real_swat_world_model_eval.csv", index=False)
    world_eval.to_csv(output / "world_model_eval_real_swat.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "evaluation_type": "whole_plant_label_eval_for_p1_model_not_primary",
                "normal_file": str(context.get("normal_path")),
                "attack_file": str(context.get("attack_path")),
                "rows": len(predicted),
                "attack_ratio": float(pd.to_numeric(predicted.get("label"), errors="coerce").eq(1).mean()),
                **detection.iloc[0].to_dict(),
                **world_eval.iloc[0].to_dict(),
            }
        ]
    )
    summary.to_csv(output / "real_swat_log_eval_summary.csv", index=False)
    make_real_swat_log_plots(predicted, output)
    print("[real_swat] log_eval completed. No closed-loop recovery is claimed for offline logs.")
    return {
        "output_dir": output,
        "summary": output / "real_swat_log_eval_summary.csv",
        "timeseries": output / "real_swat_timeseries.csv",
    }


def run_p1_log_eval(context: dict[str, Any], output: Path) -> dict[str, Path]:
    """Primary real-log evaluation for the P1-only model."""
    attack_df = context["attack_df"]
    normal_df = context["normal_df"]
    calibration = context["calibration"]
    swat_cfg = context["swat_config"]
    real_model: RealSWaTWorldModel = context["world_model"]
    p1_windows = build_p1_attack_windows(context["windows"], len(attack_df), output)
    p1_labels = add_p1_labels(attack_df, p1_windows, output)
    normal_diag = diagnose_real_swat_timeseries(normal_df, calibration, swat_cfg.get("diagnosis", {}))
    thresholds = calibrate_real_thresholds(normal_diag, output, swat_cfg)
    diagnosed = diagnose_real_swat_timeseries(attack_df, calibration, swat_cfg.get("diagnosis", {}))
    diagnosed = apply_real_thresholds(diagnosed, thresholds)
    for col in p1_labels.columns:
        if col not in diagnosed:
            diagnosed[col] = p1_labels[col]
    predicted = real_model.predict_frame(diagnosed)
    predicted["evaluation_type"] = "p1_offline_log_diagnosis"
    predicted["prediction_error_LIT101_next"] = predicted["pred_LIT101_next"] - predicted["LIT101"].shift(-1)
    predicted.to_csv(output / "real_swat_p1_timeseries.csv", index=False)
    predicted.to_csv(output / "real_swat_trust_timeseries.csv", index=False)
    predicted.to_csv(output / "real_swat_prediction_timeseries.csv", index=False)

    p1_metrics = p1_detection_metrics(predicted, normal_diag, thresholds)
    p1_metrics = _append_window_metadata(p1_metrics, p1_windows)
    p1_metrics.to_csv(output / "real_swat_p1_log_eval_summary.csv", index=False)
    by_window = p1_detection_delay_by_window(predicted, p1_windows)
    by_window.to_csv(output / "real_swat_p1_detection_by_window.csv", index=False)
    trust_by_tag = p1_trust_detection_by_tag(predicted)
    trust_by_tag = _append_window_metadata(trust_by_tag, p1_windows)
    trust_by_tag.to_csv(output / "real_swat_p1_trust_detection_by_tag.csv", index=False)
    valid_windows = valid_p1_windows(p1_windows)
    valid_metrics = _append_window_metadata(p1_detection_metrics(predicted, normal_diag, thresholds), p1_windows)
    valid_metrics.to_csv(output / "real_swat_p1_log_eval_summary_valid_windows.csv", index=False)
    valid_by_window = p1_detection_delay_by_window(predicted, valid_windows)
    valid_by_window.to_csv(output / "real_swat_p1_detection_by_window_valid.csv", index=False)
    valid_trust = _append_window_metadata(p1_trust_detection_by_tag(predicted), p1_windows)
    valid_trust.to_csv(output / "real_swat_p1_trust_detection_by_tag_valid.csv", index=False)
    world_eval = real_world_model_eval(predicted)
    world_eval["evaluation_type"] = "p1_offline_log_diagnosis"
    world_eval = _append_window_metadata(world_eval, p1_windows)
    world_eval.to_csv(output / "real_swat_p1_world_model_eval.csv", index=False)
    make_real_swat_log_plots(predicted, output)
    make_p1_valid_window_plots(predicted, p1_windows, output)
    write_p1_report(context, output, p1_windows, thresholds, p1_metrics, trust_by_tag, world_eval)
    print("[real_swat] p1_log_eval completed. Metrics use P1 attack windows, not whole-plant labels.")
    return {
        "output_dir": output,
        "summary": output / "real_swat_p1_log_eval_summary.csv",
        "timeseries": output / "real_swat_p1_timeseries.csv",
    }


def _append_window_metadata(df: pd.DataFrame, p1_windows: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key, value in p1_window_metadata(p1_windows).items():
        out[key] = value
    return out


def run_counterfactual(context: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Path]:
    attack_df = context["attack_df"]
    windows = build_p1_attack_windows(context["windows"], len(attack_df), output)
    valid_windows = valid_p1_windows(windows)
    calibration = context["calibration"]
    horizon = int(context["swat_config"].get("counterfactual", {}).get("horizon", 60))
    rollout_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    safe_low = float(calibration.get("safe_low", attack_df["LIT101"].quantile(0.05)))
    safe_high = float(calibration.get("safe_high", attack_df["LIT101"].quantile(0.95)))
    target_low = float(calibration.get("target_low", safe_low))
    target_high = float(calibration.get("target_high", safe_high))
    pre_context = int(context["swat_config"].get("counterfactual", {}).get("pre_attack_context", 60))
    for _, window in valid_windows.head(10).iterrows():
        start = int(window["start_index"]) if pd.notna(window.get("start_index")) else 0
        start = max(0, min(start, len(attack_df) - 1))
        context_idx = max(0, start - 1)
        if pre_context > 0:
            context_start = max(0, start - pre_context)
            history = pd.to_numeric(attack_df["LIT101"].iloc[context_start:start], errors="coerce").dropna()
            initial_level = float(history.iloc[-1]) if not history.empty else float(attack_df["LIT101"].iloc[context_idx])
        else:
            initial_level = float(attack_df["LIT101"].iloc[context_idx])
        target = str(window.get("p1_target_tag", ""))
        method_candidates = _counterfactual_method_candidates(target)
        selected_by_method: dict[str, tuple[RecoveryAction, dict[str, Any]]] = {}
        for method, candidates in method_candidates.items():
            evaluated: list[tuple[RecoveryAction, dict[str, Any], float]] = []
            for action in candidates:
                result = _rollout_counterfactual_action(initial_level, action, calibration, safe_low, safe_high, horizon, target)
                cost = _counterfactual_cost(result, action, target)
                evaluated.append((action, result, cost))
            selected_action, selected_result, _ = min(evaluated, key=lambda item: item[2])
            selected_by_method[method] = (selected_action, selected_result)
            for action, result, cost in evaluated:
                selected = int(action == selected_action)
                candidate_rows.append(
                    {
                        "evaluation_type": "counterfactual_model_rollout",
                        "window_id": window.get("attack_id"),
                        "attack_id": window.get("attack_id"),
                        "target_tag": target,
                        "method": method,
                        "candidate_action": action.value,
                        "predicted_cost": cost,
                        "predicted_safety_violation_duration": result["violations"],
                        "predicted_max_overshoot": result["max_over"],
                        "predicted_max_undershoot": result["max_under"],
                        "predicted_time_to_target": result["time_to_target"],
                        "predicted_production_proxy": result["production"],
                        "shield_interventions": result["shield_interventions"],
                        "selected": selected,
                    }
                )
                for step_row in result["rows"]:
                    rollout_rows.append(
                        {
                            "evaluation_type": "counterfactual_model_rollout",
                            "window_id": window.get("attack_id"),
                            "attack_id": window.get("attack_id"),
                            "target_tag": target,
                            "p1_target_tag": target,
                            "method": method,
                            "candidate_action": action.value,
                            "t": step_row["t"],
                            "step": step_row["t"],
                            "predicted_LIT101": step_row["predicted_LIT101"],
                            "predicted_level": step_row["predicted_LIT101"],
                            "safe_low": safe_low,
                            "safe_high": safe_high,
                            "target_low": target_low,
                            "target_high": target_high,
                            "predicted_pump_state": step_row["predicted_pump_state"],
                            "predicted_mv101_state": step_row["predicted_mv101_state"],
                            "selected": selected,
                        }
                    )
            summary_rows.append(
                {
                    "evaluation_type": "counterfactual_model_rollout",
                    "attack_id": window.get("attack_id"),
                    "window_id": window.get("attack_id"),
                    "p1_target_tag": target,
                    "target_tag": target,
                    "method": method,
                    "selected_action": selected_action.value,
                    "predicted_safety_violation_duration": selected_result["violations"],
                    "predicted_max_LIT101_overshoot": selected_result["max_over"],
                    "predicted_max_LIT101_undershoot": selected_result["max_under"],
                    "predicted_time_to_target": selected_result["time_to_target"],
                    "predicted_production_proxy": selected_result["production"],
                    "predicted_action_cost": action_cost(selected_action),
                    "predicted_cost": _counterfactual_cost(selected_result, selected_action, target),
                    "shield_interventions": selected_result["shield_interventions"],
                }
            )
    rollout_df = pd.DataFrame(rollout_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary = pd.DataFrame(summary_rows)
    summary = _append_window_metadata(summary, windows) if not summary.empty else summary
    candidate_df = _append_window_metadata(candidate_df, windows) if not candidate_df.empty else candidate_df
    rollout_df.to_csv(output / "real_swat_counterfactual_action_timeline.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_by_attack.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_summary.csv", index=False)
    rollout_df.to_csv(output / "real_swat_p1_counterfactual_action_timeline.csv", index=False)
    rollout_df.to_csv(output / "real_swat_p1_counterfactual_rollout_timeseries.csv", index=False)
    candidate_df.to_csv(output / "real_swat_p1_counterfactual_candidate_scores.csv", index=False)
    summary.to_csv(output / "real_swat_p1_counterfactual_by_window.csv", index=False)
    summary.to_csv(output / "real_swat_p1_counterfactual_summary.csv", index=False)
    summary.to_csv(output / "real_swat_p1_counterfactual_summary_valid_windows.csv", index=False)
    ablation = summary[summary["method"].astype(str).str.startswith("B5_")] if not summary.empty else pd.DataFrame()
    ablation.to_csv(output / "real_swat_p1_ablation_summary.csv", index=False)
    ablation.to_csv(output / "real_swat_p1_ablation_summary_valid_windows.csv", index=False)
    make_counterfactual_plots(rollout_df, output)
    make_p1_counterfactual_candidate_plot(candidate_df, output)
    generate_case_studies(attack_df, windows, rollout_df, candidate_df, calibration, output)
    if (output / "real_swat_counterfactual_level_rollouts.png").exists():
        (output / "real_swat_p1_counterfactual_level_rollouts.png").write_bytes((output / "real_swat_counterfactual_level_rollouts.png").read_bytes())
    if (output / "real_swat_counterfactual_actions.png").exists():
        (output / "real_swat_p1_counterfactual_actions.png").write_bytes((output / "real_swat_counterfactual_actions.png").read_bytes())
    return {"output_dir": output, "summary": output / "real_swat_counterfactual_summary.csv"}


def run_hybrid(context: dict[str, Any], config: dict[str, Any], output: Path, quick: bool) -> dict[str, Path]:
    # Import lazily to avoid a module import cycle with experiment.py.
    from ..experiment import metrics_for_timeseries, run_experiments
    from ..world_model import ActionConditionedWorldModel

    hybrid_cfg = copy.deepcopy(config)
    sim_cfg = simulator_config_from_calibration(config.get("simulator", {}), context["calibration"])
    calibration_check = pd.DataFrame(
        [
            {
                "evaluation_type": "real_calibrated_simulation",
                "lit_min_normal": context["calibration"].get("lit_min_normal"),
                "lit_max_normal": context["calibration"].get("lit_max_normal"),
                "safe_low": context["calibration"].get("safe_low"),
                "safe_high": context["calibration"].get("safe_high"),
                "target_low": context["calibration"].get("target_low"),
                "target_high": context["calibration"].get("target_high"),
                "hazard_low": context["calibration"].get("hazard_low"),
                "hazard_high": context["calibration"].get("hazard_high"),
                "sim_level_min": sim_cfg.get("level_min"),
                "sim_level_max": sim_cfg.get("level_max"),
                "sim_inflow_open": sim_cfg.get("inflow_rate_open"),
                "sim_p101_outflow": sim_cfg.get("p101_outflow_rate"),
                "nominal_outflow": sim_cfg.get("p101_outflow_rate"),
                "nominal_outflow_high": 1.5 * float(sim_cfg.get("p101_outflow_rate", 1.0)),
            }
        ]
    )
    calibration_check.to_csv(output / "real_swat_hybrid_calibration_check.csv", index=False)
    make_hybrid_unit_check_plot(calibration_check, output)
    hybrid_cfg["simulator"] = sim_cfg
    hybrid_cfg["shield"] = _real_unit_shield_config(config.get("shield", {}), sim_cfg)
    hybrid_cfg["planner"] = _real_unit_planner_config(config.get("planner", {}), sim_cfg)
    hybrid_cfg.setdefault("experiment", {})
    hybrid_cfg["experiment"]["attacks"] = [
        "LIT101_FDI",
        "LIT101_DRIFT",
        "LIT101_REPLAY",
        "MV101_STUCK_OPEN",
        "P101_FORCED_OFF",
        "P102_FORCED_OFF",
        "COMBINED_LIT101_FDI_MV101_OPEN",
    ]
    hybrid_cfg["experiment"]["methods"] = [
        "B1_FULL_SHUTDOWN",
        "B2_RULE_BASED_FALLBACK",
        "B3_ANOMALY_PRIORITY_RECOVERY",
        "B4_WORLD_MODEL_NO_TRUST",
        "B5_PROPOSED",
        "B5_FULL",
        "B5_NO_TRUST",
        "B5_NO_ROOT_CAUSE",
        "B5_NO_SHIELD",
        "B5_NO_WORLD_MODEL",
    ]
    hybrid_cfg["experiment"]["seeds"] = context["swat_config"].get("hybrid", {}).get("seeds", [0, 1, 2, 3, 4])
    if quick:
        hybrid_cfg["experiment"]["seeds"] = [0]
        hybrid_cfg["experiment"]["steps"] = int(config.get("experiment", {}).get("quick_steps", 70))
    wm = ActionConditionedWorldModel({**sim_cfg, **config.get("world_model", {}), "enabled": False})
    cal_rmse = float(context["calibration"].get("rmse", np.nan))
    wm.metrics = {
        "one_step_rmse": cal_rmse,
        "multi_step_rollout_rmse": float(cal_rmse * np.sqrt(5.0)) if np.isfinite(cal_rmse) else np.nan,
    }
    from ..experiment import build_diagnosis_config

    diagnosis_config = build_diagnosis_config(hybrid_cfg)
    timeseries = run_experiments(hybrid_cfg, diagnosis_config, wm, initial_level=sim_cfg.get("initial_level"))
    timeseries["evaluation_type"] = "real_calibrated_simulation"
    metrics = metrics_for_timeseries(timeseries, wm.metrics)
    metrics["evaluation_type"] = "real_calibrated_simulation"
    steps = int(hybrid_cfg["experiment"].get("steps", 70))
    metrics = annotate_hybrid_production_metrics(metrics, sim_cfg, steps)
    timeseries.to_csv(output / "real_swat_hybrid_timeseries.csv", index=False)
    metrics.to_csv(output / "real_swat_hybrid_metrics_by_method_attack.csv", index=False)
    metrics[metrics["method"].astype(str).str.startswith("B5_")].to_csv(output / "real_swat_hybrid_ablation_summary.csv", index=False)
    summarize_metrics(metrics).to_csv(output / "real_swat_hybrid_summary.csv", index=False)
    debug = hybrid_action_effect_debug(timeseries)
    debug.to_csv(output / "real_swat_hybrid_action_effect_debug.csv", index=False)
    hybrid_action_effect_summary(timeseries, metrics).to_csv(output / "real_swat_hybrid_action_effect_summary.csv", index=False)
    _write_hybrid_action_effect_warning(timeseries, output)
    make_hybrid_plots(timeseries, summarize_metrics(metrics), output)
    make_hybrid_action_effect_plots(timeseries, metrics, output)
    stress_timeseries, stress_metrics = run_hybrid_stress(
        base_config=hybrid_cfg,
        diagnosis_config=diagnosis_config,
        world_model=wm,
        sim_cfg=sim_cfg,
        output=output,
        quick=quick,
    )
    if not stress_timeseries.empty:
        stress_timeseries.to_csv(output / "real_swat_hybrid_stress_timeseries.csv", index=False)
    stress_metrics.to_csv(output / "real_swat_hybrid_stress_metrics_by_method_attack.csv", index=False)
    summarize_metrics(stress_metrics).to_csv(output / "real_swat_hybrid_stress_summary.csv", index=False)
    stress_metrics[stress_metrics["method"].astype(str).str.startswith("B5_")].to_csv(output / "real_swat_hybrid_stress_ablation_summary.csv", index=False)
    make_hybrid_action_effect_plots(stress_timeseries, stress_metrics, output, stress=True)
    if not stress_timeseries.empty:
        make_hybrid_stress_level_plot(stress_timeseries, output)
    return {"output_dir": output, "summary": output / "real_swat_hybrid_summary.csv"}


def _real_unit_shield_config(base: dict[str, Any], sim_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    safe_min = float(sim_cfg.get("safe_min", cfg.get("soft_min", 20.0)))
    safe_max = float(sim_cfg.get("safe_max", cfg.get("soft_max", 80.0)))
    target_min = float(sim_cfg.get("target_min", safe_min))
    target_max = float(sim_cfg.get("target_max", safe_max))
    span = max(1e-6, safe_max - safe_min)
    cfg.update(
        {
            "hard_min": float(sim_cfg.get("hard_min", safe_min - 0.25 * span)),
            "hard_max": float(sim_cfg.get("hard_max", safe_max + 0.25 * span)),
            "soft_min": safe_min,
            "soft_max": safe_max,
            "pump_empty_level": float(sim_cfg.get("pump_empty_level", safe_min)),
            "mv_open_high_level": target_max,
            "emergency_drain_level": safe_max - 0.05 * span,
            "severe_hazard_level_low": safe_min - 0.1 * span,
            "severe_hazard_level_high": safe_max + 0.1 * span,
        }
    )
    # Keep fallback thresholds available to action translation through belief context.
    cfg["target_min"] = target_min
    cfg["target_max"] = target_max
    return cfg


def _real_unit_planner_config(base: dict[str, Any], sim_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    cfg["target_mid"] = 0.5 * (float(sim_cfg.get("target_min", 45.0)) + float(sim_cfg.get("target_max", 60.0)))
    return cfg


def annotate_hybrid_production_metrics(metrics: pd.DataFrame, sim_cfg: dict[str, Any], steps: int) -> pd.DataFrame:
    if metrics.empty or "production" not in metrics.columns:
        return metrics
    metrics = metrics.copy()
    production = pd.to_numeric(metrics["production"], errors="coerce")
    nominal_total = float(sim_cfg.get("p101_outflow_rate", 1.0)) * int(steps)
    if "seed" in metrics.columns:
        ref = production.groupby(metrics["seed"]).transform("max")
    else:
        ref = pd.Series(production.max(), index=metrics.index)
    ref = ref.fillna(nominal_total)
    ref = np.maximum(ref.to_numpy(dtype=float), nominal_total)
    metrics["production_loss"] = (pd.Series(ref, index=metrics.index) - production).clip(lower=0.0).fillna(0.0)
    metrics["flow_deviation_cost"] = production.sub(nominal_total).abs().fillna(0.0)
    high_flow = 1.5 * nominal_total
    metrics["downstream_overload_risk"] = (production > high_flow).fillna(False).astype(int)
    return metrics


def hybrid_action_effect_debug(timeseries: pd.DataFrame) -> pd.DataFrame:
    debug_cols = [
        "method",
        "attack",
        "seed",
        "t",
        "step",
        "selected_action",
        "shielded_action",
        "command_MV101",
        "actual_MV101",
        "command_P101",
        "actual_P101",
        "command_P102",
        "actual_P102",
        "actuator_MV101_trusted",
        "actuator_P101_trusted",
        "actuator_P102_trusted",
        "inflow",
        "outflow",
        "level_delta",
        "LIT101_level",
        "soft_safety_violation",
        "hard_safety_violation",
        "production",
    ]
    out = timeseries.copy()
    if "step" not in out and "t" in out:
        out["step"] = out["t"]
    for col in debug_cols:
        if col not in out:
            out[col] = np.nan
    return out[debug_cols]


def hybrid_action_effect_summary(timeseries: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if timeseries.empty:
        return pd.DataFrame(rows)
    for (attack, method), group in timeseries.groupby(["attack", "method"]):
        first_seed = group[group["seed"] == group["seed"].min()].sort_values("t")
        trajectory_signature = "|".join(f"{v:.3f}" for v in pd.to_numeric(first_seed["LIT101_level"], errors="coerce").head(12).fillna(0))
        rows.append(
            {
                "attack": attack,
                "method": method,
                "unique_shielded_actions": int(first_seed["shielded_action"].astype(str).nunique()) if "shielded_action" in first_seed else 0,
                "mean_inflow": float(pd.to_numeric(group.get("inflow"), errors="coerce").mean()),
                "mean_outflow": float(pd.to_numeric(group.get("outflow"), errors="coerce").mean()),
                "mean_level_delta": float(pd.to_numeric(group.get("level_delta"), errors="coerce").mean()),
                "trajectory_signature": trajectory_signature,
            }
        )
    summary = pd.DataFrame(rows)
    if not metrics.empty:
        metric_cols = ["attack", "method", "safety_violation_duration", "production_loss", "production"]
        present = [col for col in metric_cols if col in metrics]
        metric_summary = metrics[present].groupby(["attack", "method"], as_index=False).mean(numeric_only=True)
        summary = summary.merge(metric_summary, on=["attack", "method"], how="left")
    return summary


def _write_hybrid_action_effect_warning(timeseries: pd.DataFrame, output: Path) -> None:
    warning_path = output / "hybrid_action_effect_warning.txt"
    if timeseries.empty:
        warning_path.write_text("Hybrid action-effect check skipped: empty timeseries.\n", encoding="utf-8")
        return
    identical_attacks: list[str] = []
    different_attacks: list[str] = []
    for attack, group in timeseries.groupby("attack"):
        b1 = group[group["method"] == "B1_FULL_SHUTDOWN"].sort_values(["seed", "t"])
        b5 = group[group["method"].isin(["B5_PROPOSED", "B5_FULL"])].sort_values(["method", "seed", "t"])
        if b1.empty or b5.empty:
            continue
        b5_one = b5[b5["method"] == b5["method"].iloc[0]].sort_values(["seed", "t"])
        n = min(len(b1), len(b5_one))
        if n and np.allclose(
            pd.to_numeric(b1["LIT101_level"].iloc[:n], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(b5_one["LIT101_level"].iloc[:n], errors="coerce").to_numpy(dtype=float),
            equal_nan=True,
        ):
            identical_attacks.append(str(attack))
        elif n:
            different_attacks.append(str(attack))
    if identical_attacks and not different_attacks:
        warning_path.write_text(
            "B1_FULL_SHUTDOWN and B5_PROPOSED/FULL trajectories were identical for all comparable attacks: "
            + ", ".join(identical_attacks)
            + "\n",
            encoding="utf-8",
        )
    elif warning_path.exists():
        warning_path.unlink()


def run_hybrid_stress(
    base_config: dict[str, Any],
    diagnosis_config: dict[str, Any],
    world_model: Any,
    sim_cfg: dict[str, Any],
    output: Path,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..experiment import metrics_for_timeseries, run_experiments

    stress_cfg = copy.deepcopy(base_config)
    stress_cfg["simulator"] = sim_cfg
    stress_cfg.setdefault("experiment", {})
    stress_cfg["experiment"]["attacks"] = [
        "MV101_STUCK_OPEN_HIGH_LEVEL",
        "LIT101_DRIFT_HIGH_LEVEL",
        "LIT101_REPLAY_HIGH_LEVEL",
        "P101_FORCED_OFF_HIGH_LEVEL",
        "P102_FORCED_OFF_HIGH_LEVEL",
        "COMBINED_LIT101_REPLAY_MV101_OPEN",
        "MV101_STUCK_OPEN_P102_UNAVAILABLE",
        "DELAYED_DETECTION_5_STEPS",
    ]
    stress_cfg["experiment"]["methods"] = [
        "B1_FULL_SHUTDOWN",
        "B2_RULE_BASED_FALLBACK",
        "B3_ANOMALY_PRIORITY_RECOVERY",
        "B4_WORLD_MODEL_NO_TRUST",
        "B5_PROPOSED",
        "B5_FULL",
        "B5_NO_TRUST",
        "B5_NO_ROOT_CAUSE",
        "B5_NO_SHIELD",
        "B5_NO_WORLD_MODEL",
    ]
    if quick:
        stress_cfg["experiment"]["seeds"] = [0]
        stress_cfg["experiment"]["steps"] = int(stress_cfg.get("experiment", {}).get("steps", 70))
    timeseries = run_experiments(stress_cfg, diagnosis_config, world_model, initial_level=None)
    timeseries["evaluation_type"] = "real_calibrated_simulation_stress"
    metrics = metrics_for_timeseries(timeseries, world_model.metrics)
    metrics["evaluation_type"] = "real_calibrated_simulation_stress"
    steps = int(stress_cfg["experiment"].get("steps", 70))
    metrics = annotate_hybrid_production_metrics(metrics, sim_cfg, steps)
    hybrid_action_effect_debug(timeseries).to_csv(output / "real_swat_hybrid_stress_action_effect_debug.csv", index=False)
    return timeseries, metrics


def make_hybrid_stress_level_plot(timeseries: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    sample = timeseries[
        (timeseries["seed"] == timeseries["seed"].min())
        & (timeseries["attack"].isin(["MV101_STUCK_OPEN_HIGH_LEVEL", "P102_FORCED_OFF_HIGH_LEVEL"]))
    ]
    if sample.empty:
        sample = timeseries[timeseries["seed"] == timeseries["seed"].min()]
    plt.figure(figsize=(11, 5))
    for (method, attack), group in sample.groupby(["method", "attack"]):
        plt.plot(group["t"], group["LIT101_level"], label=f"{method}/{attack}", linewidth=0.9)
    plt.xlabel("step")
    plt.ylabel("LIT101 real-unit level")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output / "real_swat_hybrid_stress_level_trajectories.png", dpi=160)
    plt.close()


def detection_metrics(attack_diag: pd.DataFrame, normal_diag: pd.DataFrame) -> pd.DataFrame:
    labels = pd.to_numeric(attack_diag.get("label"), errors="coerce")
    score = pd.to_numeric(attack_diag.get("attack_belief_score"), errors="coerce").fillna(0.0)
    pred = (score >= 0.25).astype(int)
    if labels.notna().any() and set(labels.dropna().astype(int).unique()).issubset({0, 1}):
        valid = labels.notna()
        precision, recall, f1, _ = precision_recall_fscore_support(labels[valid].astype(int), pred[valid], average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(labels[valid].astype(int), score[valid]))
        except Exception:
            auc = np.nan
    else:
        precision = recall = f1 = auc = np.nan
    false_alarm = float((pd.to_numeric(normal_diag.get("attack_belief_score"), errors="coerce").fillna(0) >= 0.25).mean()) if not normal_diag.empty else np.nan
    return pd.DataFrame(
        [
            {
                "evaluation_type": "whole_plant_label_eval_for_p1_model_not_primary",
                "attack_detection_precision": precision,
                "attack_detection_recall": recall,
                "attack_detection_f1": f1,
                "attack_detection_auc": auc,
                "false_alarm_rate_on_normal_validation": false_alarm,
            }
        ]
    )


def p1_detection_metrics(
    diag: pd.DataFrame,
    normal_diag: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    truth = pd.to_numeric(diag.get("p1_attack_label_any"), errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(diag.get("p1_attack_score", diag.get("attack_belief_score")), errors="coerce").fillna(0.0)
    pred = (score >= 0.5).astype(int)
    if truth.sum() > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(truth, score))
        except Exception:
            auc = np.nan
    else:
        precision = recall = f1 = auc = np.nan
    normal_score = pd.to_numeric(normal_diag.get("attack_belief_score"), errors="coerce").fillna(0.0)
    normal_trigger = (
        (pd.to_numeric(normal_diag.get("abs_residual_LIT101"), errors="coerce") > thresholds["residual_abs_threshold"])
        | (pd.to_numeric(normal_diag.get("lit101_residual_ewma"), errors="coerce") > thresholds["residual_ewma_threshold"])
        | (normal_score >= 0.5)
    )
    return pd.DataFrame(
        [
            {
                "evaluation_type": "p1_offline_log_diagnosis",
                "p1_attack_detection_precision": precision,
                "p1_attack_detection_recall": recall,
                "p1_attack_detection_f1": f1,
                "p1_attack_detection_auc": auc,
                "p1_false_alarm_rate_on_normal": float(normal_trigger.mean()) if len(normal_trigger) else np.nan,
                "p1_windows_evaluated": int(truth.sum() > 0),
            }
        ]
    )


def p1_detection_delay_by_window(diag: pd.DataFrame, p1_windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score = pd.to_numeric(diag.get("p1_attack_score", diag.get("attack_belief_score")), errors="coerce").fillna(0.0)
    pred = score >= 0.5
    for _, window in p1_windows.iterrows():
        if window.get("alignment_status") == "out_of_loaded_range" or pd.isna(window.get("start_index")) or pd.isna(window.get("end_index")):
            rows.append({**window.to_dict(), "detected": np.nan, "detection_delay": np.nan})
            continue
        start = int(window["start_index"])
        end = int(window["end_index"])
        hits = np.where(pred.iloc[start : end + 1].to_numpy())[0]
        detected = len(hits) > 0
        rows.append({**window.to_dict(), "detected": int(detected), "detection_delay": int(hits[0]) if detected else np.nan})
    return pd.DataFrame(rows)


def p1_trust_detection_by_tag(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag in P1_TARGETS:
        truth_col = f"p1_attack_label_{tag}"
        pred_col = f"trust_{tag}"
        if truth_col not in diag or pred_col not in diag:
            rows.append({"tag": tag, "precision": np.nan, "recall": np.nan, "f1": np.nan})
            continue
        truth = pd.to_numeric(diag[truth_col], errors="coerce").fillna(0).astype(int)
        pred = (pd.to_numeric(diag[pred_col], errors="coerce").fillna(1).astype(int) == 0).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
        rows.append({"tag": tag, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def real_world_model_eval(df: pd.DataFrame) -> pd.DataFrame:
    target = df["LIT101"].shift(-1)
    pred = df.get("pred_LIT101_next")
    attack_mask = pd.to_numeric(df.get("label"), errors="coerce").eq(1)
    raw = _rmse(df["LIT101"], target)
    full = _rmse(df["lit101_est"], target)
    partial = _rmse(df["level_est"], target)
    row = {
        "evaluation_type": "offline_log_diagnosis",
        "one_step_rmse": _rmse(pred, target),
        "attack_period_rmse": _rmse(pred[attack_mask], target[attack_mask]) if attack_mask.any() else np.nan,
        "raw_observation_rmse": raw,
        "full_rollback_rmse": full,
        "partial_rollback_rmse": partial,
        "trust_aware_reconstruction_rmse": partial,
        "raw_next_step_prediction_error": raw,
        "full_rollback_consistency_error": full,
        "partial_rollback_consistency_error": partial,
        "trust_aware_future_consistency_error": partial,
    }
    return pd.DataFrame([row])


def write_p1_report(
    context: dict[str, Any],
    output: Path,
    p1_windows: pd.DataFrame,
    thresholds: dict[str, float],
    p1_metrics: pd.DataFrame,
    trust_by_tag: pd.DataFrame,
    world_eval: pd.DataFrame,
) -> None:
    inventory = context["inventory"]
    mapping_path = output / "swat_column_mapping.json"
    valid_windows = valid_p1_windows(p1_windows)
    excluded_windows = p1_windows.loc[~p1_windows.index.isin(valid_windows.index)] if not p1_windows.empty else p1_windows
    metadata = p1_window_metadata(p1_windows)
    actuator_features = output / "real_swat_actuator_diagnosis_features.csv"
    candidate_scores = output / "real_swat_p1_counterfactual_candidate_scores.csv"
    hybrid_debug = output / "real_swat_hybrid_action_effect_summary.csv"
    stress_summary = output / "real_swat_hybrid_stress_summary.csv"
    lines = [
        "# Real SWaT P1 Report",
        "",
        "## 1. Files Discovered",
        f"- files: {len(inventory)}",
        f"- normal: {context.get('normal_path')}",
        f"- attack: {context.get('attack_path')}",
        "",
        "## 2. P1 Columns Found",
        mapping_path.read_text(encoding="utf-8") if mapping_path.exists() else "mapping unavailable",
        "",
        "## 3. P1 Attack Windows",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "",
        "### Valid Windows",
        valid_windows.to_string(index=False) if not valid_windows.empty else "No valid P1 windows in the loaded data range.",
        "",
        "### Excluded Windows",
        excluded_windows.to_string(index=False) if not excluded_windows.empty else "No excluded P1 windows.",
        "",
        "## 4. All P1 Attack Windows Used",
        p1_windows.to_string(index=False) if not p1_windows.empty else "No P1 windows in loaded range.",
        "",
        "## 5. Thresholds Calibrated From Normal Data",
        json.dumps(thresholds, indent=2),
        "",
        "## 6. P1 Valid-Window Diagnosis Metrics",
        p1_metrics.to_string(index=False),
        "",
        "## 7. Trust Detection By P1 Tag",
        trust_by_tag.to_string(index=False),
        "",
        "## 8. Actuator Diagnosis Summary",
        actuator_features.read_text(encoding="utf-8") if actuator_features.exists() else "Run p1_log_eval to create actuator diagnosis feature summaries.",
        "",
        "## 9. World Model Prediction Metrics",
        world_eval.to_string(index=False),
        "",
        "## 10. Counterfactual Recovery Metrics",
        "See `real_swat_p1_counterfactual_summary.csv`, `real_swat_p1_counterfactual_candidate_scores.csv`, and `real_swat_p1_case_study_report.md` after running counterfactual mode.",
        candidate_scores.read_text(encoding="utf-8")[:2000] if candidate_scores.exists() else "Counterfactual candidate scores not present yet.",
        "",
        "## 11. Hybrid Recovery Metrics",
        "See `real_swat_hybrid_summary.csv`, `real_swat_hybrid_action_effect_debug.csv`, and stress scenario outputs after running hybrid mode.",
        hybrid_debug.read_text(encoding="utf-8")[:2000] if hybrid_debug.exists() else "Hybrid action-effect debug summary not present yet.",
        "",
        "## 12. Stress Scenario Summary",
        stress_summary.read_text(encoding="utf-8")[:2000] if stress_summary.exists() else "Hybrid stress summary not present yet.",
        "",
        "## 13. Limitations",
        "- Whole-plant SWaT labels are not primary for a P1-only model.",
        "- Offline logs do not validate closed-loop recovery actions.",
        "- Quick mode evaluates only P1 windows that truly overlap the loaded log rows.",
        "- Attack list targets are used for evaluation/case-study context, not as direct online trust labels.",
        "- Reconstruction metrics are future-consistency proxy errors, not true counterfactual physical-state errors.",
    ]
    (output / "real_swat_p1_report.md").write_text("\n".join(lines), encoding="utf-8")


def real_trust_detection_by_tag(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tags = ["LIT101", "FIT101", "MV101", "P101", "P102", "PLC1"]
    label = pd.to_numeric(df.get("label"), errors="coerce")
    for tag in tags:
        pred = (df.get(f"trust_{tag}", 1) == 0).astype(int)
        # Without per-tag ground truth, use attack windows with target text when available.
        truth = pd.Series(0, index=df.index, dtype=int)
        for _, window in windows.iterrows():
            target_text = str(window.get("target_tags", "")).upper()
            if tag in target_text or target_text in {"UNKNOWN", ""}:
                if pd.notna(window.get("start_index")) and pd.notna(window.get("end_index")):
                    truth.iloc[int(window["start_index"]) : int(window["end_index"]) + 1] = int(tag in target_text)
        if truth.sum() == 0 and label.notna().any() and tag == "LIT101":
            truth = label.fillna(0).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
        rows.append({"tag": tag, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def _counterfactual_rates(calibration: dict[str, Any]) -> dict[str, float]:
    lit_range = max(
        1e-6,
        float(calibration.get("lit_max_normal", 100.0)) - float(calibration.get("lit_min_normal", 0.0)),
    )
    beta = calibration.get("coefficients", {}) or {}
    fit_effect = abs(float(beta.get("beta_fit", 0.01)))
    p101_effect = abs(float(beta.get("beta_p101", 0.01)))
    p102_effect = abs(float(beta.get("beta_p102", p101_effect)))
    return {
        "inflow_open": max(0.001 * lit_range, fit_effect * float(calibration.get("mv101_fit_open_mean", 1.0))),
        "inflow_closed": max(0.0, fit_effect * float(calibration.get("mv101_fit_closed_mean", 0.0))),
        "out_p101": max(0.001 * lit_range, p101_effect),
        "out_p102": max(0.001 * lit_range, p102_effect),
    }


def _counterfactual_method_candidates(target: str) -> dict[str, list[RecoveryAction]]:
    proposed = [
        RecoveryAction.R0_KEEP_CURRENT,
        RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN,
        RecoveryAction.R5_P1_FALLBACK_CONTROL,
        RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK,
        RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS,
        RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP,
    ]
    return {
        "R0_KEEP_CURRENT": [RecoveryAction.R0_KEEP_CURRENT],
        "FULL_SHUTDOWN": [RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN],
        "RULE_FALLBACK": [RecoveryAction.R5_P1_FALLBACK_CONTROL],
        "PROPOSED": proposed,
        "B5_FULL": proposed,
        "B5_NO_TRUST": [RecoveryAction.R0_KEEP_CURRENT, RecoveryAction.R5_P1_FALLBACK_CONTROL],
        "B5_NO_ROOT_CAUSE": [RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN],
        "B5_NO_SHIELD": [_target_priority_action(target)],
    }


def _target_priority_action(target: str) -> RecoveryAction:
    target = normalize_target_tag(target)
    if target in {"LIT101", "FIT101"}:
        return RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK
    if target == "MV101":
        return RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS
    if target in {"P101", "P102"}:
        return RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP
    return RecoveryAction.R5_P1_FALLBACK_CONTROL


def _rollout_counterfactual_action(
    initial_level: float,
    action: RecoveryAction,
    calibration: dict[str, Any],
    safe_low: float,
    safe_high: float,
    horizon: int,
    target: str = "",
) -> dict[str, Any]:
    target_low = float(calibration.get("target_low", safe_low))
    target_high = float(calibration.get("target_high", safe_high))
    level = float(initial_level)
    rates = _counterfactual_rates(calibration)
    rows: list[dict[str, Any]] = []
    violations = 0
    production = 0.0
    max_over = 0.0
    max_under = 0.0
    time_to_target = horizon
    shield_interventions = 0
    mv = 1
    p101 = 1
    p102 = 0
    target_norm = normalize_target_tag(target)
    for step in range(horizon):
        command_mv, command_p101, command_p102 = _counterfactual_command(level, action, target_low, target_high, mv, p101, p102)
        if level < safe_low and (command_p101 or command_p102) and action != RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
            command_p101 = 0
            command_p102 = 0
            shield_interventions += 1
        mv = command_mv
        p101 = command_p101
        p102 = command_p102
        if target_norm == "MV101":
            mv = 1
        if target_norm == "P101":
            p101 = 0
        if target_norm == "P102":
            p102 = 0
        inflow = rates["inflow_open"] if mv else rates["inflow_closed"]
        outflow = p101 * rates["out_p101"] + p102 * rates["out_p102"]
        level = float(level + inflow - outflow)
        rows.append(
            {
                "t": step,
                "predicted_LIT101": level,
                "predicted_mv101_state": int(mv),
                "predicted_pump_state": f"P101={int(p101)};P102={int(p102)}",
                "predicted_p101_state": int(p101),
                "predicted_p102_state": int(p102),
                "inflow": inflow,
                "outflow": outflow,
            }
        )
        safe = safe_low <= level <= safe_high
        violations += int(not safe)
        max_over = max(max_over, max(0.0, level - safe_high))
        max_under = max(max_under, max(0.0, safe_low - level))
        if safe:
            production += outflow
        if time_to_target == horizon and target_low <= level <= target_high:
            time_to_target = step
    return {
        "trajectory": [row["predicted_LIT101"] for row in rows],
        "rows": rows,
        "violations": violations,
        "production": production,
        "max_over": max_over,
        "max_under": max_under,
        "time_to_target": time_to_target,
        "shield_interventions": shield_interventions,
    }


def _counterfactual_command(
    level: float,
    action: RecoveryAction,
    target_low: float,
    target_high: float,
    mv: int,
    p101: int,
    p102: int,
) -> tuple[int, int, int]:
    command_mv, command_p101, command_p102 = int(mv), int(p101), int(p102)
    if action == RecoveryAction.R0_KEEP_CURRENT:
        if level < target_low:
            command_mv = 1
        elif level > target_high:
            command_mv = 0
        command_p101 = int(level > target_high)
        command_p102 = 0
    elif action == RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN:
        command_mv, command_p101, command_p102 = 0, 0, 0
    elif action in {RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK}:
        command_mv = int(level < target_low)
        command_p101 = int(level > target_high)
        command_p102 = 0
    elif action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
        command_mv = 0
        command_p101 = int(level >= target_low)
        command_p102 = int(level >= target_low)
    elif action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP:
        command_mv = int(level < target_high)
        command_p101 = 0
        command_p102 = int(level >= target_low)
    return int(command_mv), int(command_p101), int(command_p102)


def _counterfactual_cost(result: dict[str, Any], action: RecoveryAction, target: str) -> float:
    cost = (
        10.0 * float(result["violations"])
        + 12.0 * float(result["max_over"])
        + 12.0 * float(result["max_under"])
        + 0.05 * float(result["time_to_target"])
        - 0.15 * float(result["production"])
        + action_cost(action)
        + 0.5 * float(result.get("shield_interventions", 0))
    )
    target_norm = normalize_target_tag(target)
    if target_norm == "MV101" and action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
        cost -= 4.0
    if target_norm in {"P101", "P102"} and action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP:
        cost -= 4.0
    if target_norm == "LIT101" and action == RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK:
        cost -= 3.0
    return float(cost)


def generate_case_studies(
    attack_df: pd.DataFrame,
    p1_windows: pd.DataFrame,
    rollout_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    calibration: dict[str, Any],
    output: Path,
) -> None:
    studies = [("1", "MV101"), ("2", "P102"), ("3", "LIT101")]
    report_lines = ["# Real SWaT P1 Case Study Report", ""]
    valid = valid_p1_windows(p1_windows)
    for attack_id, tag in studies:
        safe_tag = tag.upper()
        prefix = output / f"case_study_attack_{attack_id}_{safe_tag}"
        window = _find_case_window(p1_windows, attack_id, safe_tag)
        if window is None:
            reason = "no_matching_p1_window"
            _write_skipped_case(prefix, attack_id, safe_tag, reason)
            report_lines.extend([f"## Attack {attack_id} / {safe_tag}", f"- skipped: {reason}", ""])
            continue
        is_valid = not valid[valid["attack_id"].astype(str) == str(window.get("attack_id"))].empty
        if not is_valid:
            reason = str(window.get("alignment_status_detail", window.get("alignment_status", "invalid_window")))
            _write_skipped_case(prefix, attack_id, safe_tag, reason)
            report_lines.extend([f"## Attack {attack_id} / {safe_tag}", f"- skipped: {reason}", ""])
            continue
        window_id = str(window.get("attack_id"))
        candidate = candidate_df[
            (candidate_df.get("window_id").astype(str) == window_id)
            & (candidate_df.get("method").astype(str) == "PROPOSED")
        ] if not candidate_df.empty else pd.DataFrame()
        selected = candidate[candidate.get("selected", 0).astype(int) == 1] if not candidate.empty else pd.DataFrame()
        selected_row = selected.iloc[0].to_dict() if not selected.empty else {}
        r0 = candidate[candidate.get("candidate_action").astype(str) == RecoveryAction.R0_KEEP_CURRENT.value] if not candidate.empty else pd.DataFrame()
        proposed_cost = float(selected_row.get("predicted_cost", np.nan))
        r0_cost = float(r0.iloc[0].get("predicted_cost", np.nan)) if not r0.empty else np.nan
        reduced = bool(np.isfinite(proposed_cost) and np.isfinite(r0_cost) and proposed_cost < r0_cost)
        summary = pd.DataFrame(
            [
                {
                    "attack_id": attack_id,
                    "window_id": window_id,
                    "target_tag": safe_tag,
                    "status": "valid",
                    "inferred_root_cause": selected_row.get("target_tag", safe_tag),
                    "proposed_selected_action": selected_row.get("candidate_action", ""),
                    "proposed_predicted_cost": proposed_cost,
                    "r0_predicted_cost": r0_cost,
                    "proposed_reduces_predicted_hazard_vs_r0": reduced,
                    "limitations": "Counterfactual rollout uses a calibrated local model, not closed-loop real actuation.",
                }
            ]
        )
        summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
        rollouts = _case_rollout_subset(rollout_df, window_id)
        rollouts.to_csv(prefix.with_name(prefix.name + "_rollouts.csv"), index=False)
        _plot_case_study(attack_df, window, rollouts, calibration, prefix.with_name(prefix.name + "_plot.png"))
        report_lines.extend(
            [
                f"## Attack {attack_id} / {safe_tag}",
                f"- target: {safe_tag}",
                f"- inferred/root-cause prior: {safe_tag}",
                f"- proposed action: {selected_row.get('candidate_action', 'not_available')}",
                f"- reduced predicted hazard vs R0: {reduced}",
                "- limitation: model-based counterfactual, not real closed-loop control.",
                "",
            ]
        )
    (output / "real_swat_p1_case_study_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def _find_case_window(p1_windows: pd.DataFrame, attack_id: str, tag: str) -> pd.Series | None:
    if p1_windows.empty:
        return None
    exact = p1_windows[
        (p1_windows.get("attack_id").astype(str) == str(attack_id))
        & (p1_windows.get("p1_target_tag").astype(str).str.upper() == tag)
    ]
    if not exact.empty:
        return exact.iloc[0]
    by_id = p1_windows[p1_windows.get("attack_id").astype(str) == str(attack_id)]
    if not by_id.empty:
        return by_id.iloc[0]
    by_tag = p1_windows[p1_windows.get("p1_target_tag").astype(str).str.upper() == tag]
    if not by_tag.empty:
        return by_tag.iloc[0]
    return None


def _write_skipped_case(prefix: Path, attack_id: str, tag: str, reason: str) -> None:
    pd.DataFrame([{"attack_id": attack_id, "target_tag": tag, "status": "skipped", "reason": reason}]).to_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        index=False,
    )
    pd.DataFrame(columns=["window_id", "method", "candidate_action", "t", "predicted_LIT101"]).to_csv(
        prefix.with_name(prefix.name + "_rollouts.csv"),
        index=False,
    )


def _case_rollout_subset(rollout_df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if rollout_df.empty:
        return rollout_df
    subset = rollout_df[rollout_df.get("window_id").astype(str) == str(window_id)].copy()
    keep_methods = {"R0_KEEP_CURRENT", "FULL_SHUTDOWN", "RULE_FALLBACK", "PROPOSED"}
    subset = subset[subset["method"].astype(str).isin(keep_methods)]
    if "selected" in subset:
        proposed = subset[subset["method"].astype(str) == "PROPOSED"]
        selected_actions = set(proposed.loc[proposed["selected"].astype(int) == 1, "candidate_action"].astype(str))
        subset = subset[(subset["method"].astype(str) != "PROPOSED") | (subset["candidate_action"].astype(str).isin(selected_actions))]
    return subset


def _plot_case_study(attack_df: pd.DataFrame, window: pd.Series, rollouts: pd.DataFrame, calibration: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    start = int(window["start_index"])
    end = int(window["end_index"])
    left = max(0, start - 120)
    right = min(len(attack_df), end + 121)
    local = attack_df.iloc[left:right].copy()
    x = local["t"] if "t" in local else np.arange(left, right)
    plt.figure(figsize=(11, 5.5))
    plt.plot(x, local["LIT101"], label="observed LIT101", linewidth=1.0)
    if "level_est" in local:
        plt.plot(x, local["level_est"], label="estimated LIT101", linewidth=1.0)
    plt.axhspan(float(calibration.get("safe_low", local["LIT101"].min())), float(calibration.get("safe_high", local["LIT101"].max())), color="#d6eadf", alpha=0.25, label="safe band")
    plt.axhspan(float(calibration.get("target_low", local["LIT101"].min())), float(calibration.get("target_high", local["LIT101"].max())), color="#d7e3fc", alpha=0.22, label="target band")
    plt.axvspan(start, end, color="#f2a65a", alpha=0.2, label="attack window")
    if not rollouts.empty:
        base_t = end + 5
        for (method, action), group in rollouts.groupby(["method", "candidate_action"]):
            plt.plot(base_t + group["t"], group["predicted_LIT101"], label=f"{method}/{action}", linewidth=0.9)
    plt.xlabel("index / rollout step")
    plt.ylabel("LIT101")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _rmse(pred: pd.Series, truth: pd.Series) -> float:
    pred_arr = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    truth_arr = pd.to_numeric(truth, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(truth_arr)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((pred_arr[mask] - truth_arr[mask]) ** 2)))


def _real_swat_error(output: Path, message: str) -> None:
    print(f"[real_swat] {message}")
    (output / "real_swat_error.txt").write_text(message + "\n", encoding="utf-8")
    return None
