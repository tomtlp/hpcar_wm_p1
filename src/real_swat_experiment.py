"""Real SWaT offline evaluation tasks.

Real SWaT logs are not interactive. The log-eval task therefore evaluates
diagnosis, reconstruction, and prediction only. Recovery metrics are produced
only in counterfactual model rollouts or real-calibrated hybrid simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .attacks import AttackConfig
from .causal_logic import diagnose_real_swat_timeseries
from .metrics import summarize_metrics
from .plotting import (
    make_counterfactual_plots,
    make_hybrid_plots,
    make_real_swat_log_plots,
    make_p1_residual_threshold_plot,
    make_hybrid_unit_check_plot,
)
from .recovery_actions import RecoveryAction, action_cost, all_recovery_actions
from .swat_attack_windows import parse_attack_windows
from .swat_calibration import calibrate_p1_from_normal, simulator_config_from_calibration
from .swat_loader import (
    REQUIRED_P1_TAGS,
    choose_role_file,
    discover_swat_files,
    read_swat_table,
    write_column_mapping,
    write_missing_columns,
)
from .swat_preprocess import preprocess_swat_dataframe
from .utils import ensure_dir
from .world_model import RealSWaTWorldModel


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
        in_range = pd.notna(start) and pd.notna(end) and int(start) < loaded_len and int(end) >= 0
        if not in_range:
            status = "out_of_loaded_range"
            start_out = pd.NA
            end_out = pd.NA
            duration = 0
        else:
            start_i = int(start)
            end_i = int(end)
            if end_i >= loaded_len:
                status = "out_of_loaded_range"
                start_out = pd.NA
                end_out = pd.NA
                duration = 0
            else:
                start_out = max(0, start_i)
                end_out = max(start_out, end_i)
                duration = int(end_out - start_out + 1)
        rows.append(
            {
                "attack_id": row.get("attack_id"),
                "target_tags_original": original,
                "target_tags_normalized": normalize_target_tag(original),
                "p1_target_tag": p1_tag,
                "start_index": start_out,
                "end_index": end_out,
                "duration": duration,
                "alignment_status": status,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output / "real_swat_p1_attack_windows.csv", index=False)
    return df


def add_p1_labels(df: pd.DataFrame, p1_windows: pd.DataFrame, output: Path) -> pd.DataFrame:
    labeled = pd.DataFrame({"t": df["t"] if "t" in df else np.arange(len(df))})
    for col in ["timestamp", "label"]:
        if col in df:
            labeled[col] = df[col]
    labeled["p1_attack_label_any"] = 0
    for tag in P1_TARGETS:
        labeled[f"p1_attack_label_{tag}"] = 0
    for _, row in p1_windows.iterrows():
        if row.get("alignment_status") == "out_of_loaded_range":
            continue
        if pd.isna(row.get("start_index")) or pd.isna(row.get("end_index")):
            continue
        start = int(row["start_index"])
        end = int(row["end_index"])
        tag = str(row["p1_target_tag"])
        labeled.loc[start:end, "p1_attack_label_any"] = 1
        if tag in P1_TARGETS:
            labeled.loc[start:end, f"p1_attack_label_{tag}"] = 1
    supervision = "attack_list_p1_windows" if not p1_windows.empty else "weak_global_label_fallback"
    if labeled["p1_attack_label_any"].sum() == 0 and "label" in df:
        labeled["p1_attack_label_any"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        supervision = "weak_global_label_fallback"
    labeled["p1_label_source"] = supervision
    labeled.to_csv(output / "real_swat_p1_labels.csv", index=False)
    return labeled


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
    (output / "real_swat_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
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
    trust_cols = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102", "trust_PLC1"]
    out["attack_belief_score"] = (1 - out[trust_cols]).mean(axis=1).clip(0, 1)
    out["p1_attack_score"] = np.maximum(out["attack_belief_score"], out["p1_lit101_threshold_trigger"] * 0.8)
    out["level_est"] = out["LIT101"]
    out.loc[out["trust_LIT101"] == 0, "level_est"] = out.loc[out["trust_LIT101"] == 0, "lit101_est"]
    return out


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
    p1_metrics.to_csv(output / "real_swat_p1_log_eval_summary.csv", index=False)
    by_window = p1_detection_delay_by_window(predicted, p1_windows)
    by_window.to_csv(output / "real_swat_p1_detection_by_window.csv", index=False)
    trust_by_tag = p1_trust_detection_by_tag(predicted)
    trust_by_tag.to_csv(output / "real_swat_p1_trust_detection_by_tag.csv", index=False)
    world_eval = real_world_model_eval(predicted)
    world_eval["evaluation_type"] = "p1_offline_log_diagnosis"
    world_eval.to_csv(output / "real_swat_p1_world_model_eval.csv", index=False)
    make_real_swat_log_plots(predicted, output)
    write_p1_report(context, output, p1_windows, thresholds, p1_metrics, trust_by_tag, world_eval)
    print("[real_swat] p1_log_eval completed. Metrics use P1 attack windows, not whole-plant labels.")
    return {
        "output_dir": output,
        "summary": output / "real_swat_p1_log_eval_summary.csv",
        "timeseries": output / "real_swat_p1_timeseries.csv",
    }


def run_counterfactual(context: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Path]:
    attack_df = context["attack_df"]
    windows = build_p1_attack_windows(context["windows"], len(attack_df), output)
    calibration = context["calibration"]
    horizon = int(context["swat_config"].get("counterfactual", {}).get("horizon", 60))
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    safe_low = float(calibration.get("safe_low", attack_df["LIT101"].quantile(0.05)))
    safe_high = float(calibration.get("safe_high", attack_df["LIT101"].quantile(0.95)))
    for _, window in windows[windows["alignment_status"] != "out_of_loaded_range"].head(10).iterrows():
        start = int(window["start_index"]) if pd.notna(window.get("start_index")) else 0
        start = max(0, min(start, len(attack_df) - 1))
        initial_level = float(attack_df["LIT101"].iloc[start])
        target = str(window.get("p1_target_tag", ""))
        methods = {
            "R0_KEEP_CURRENT": RecoveryAction.R0_KEEP_CURRENT,
            "FULL_SHUTDOWN": RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN,
            "RULE_FALLBACK": RecoveryAction.R5_P1_FALLBACK_CONTROL,
            "PROPOSED": _target_priority_action(target),
        }
        ablations = {
            "B5_FULL": methods["PROPOSED"],
            "B5_NO_TRUST": RecoveryAction.R0_KEEP_CURRENT,
            "B5_NO_ROOT_CAUSE": RecoveryAction.R5_P1_FALLBACK_CONTROL,
            "B5_NO_SHIELD": methods["PROPOSED"],
        }
        method_results = []
        for method, action in {**methods, **ablations}.items():
            result = _rollout_counterfactual_action(initial_level, action, calibration, safe_low, safe_high, horizon)
            method_results.append((method, action, result))
            for step, level in enumerate(result["trajectory"]):
                rows.append(
                    {
                        "evaluation_type": "counterfactual_model_rollout",
                        "attack_id": window.get("attack_id"),
                        "p1_target_tag": target,
                        "method": method,
                        "candidate_action": action.value,
                        "step": step,
                        "predicted_level": level,
                        "selected": int(method in {"PROPOSED", "B5_FULL"}),
                    }
                )
            if method in methods:
                summary_rows.append(
                    {
                        "evaluation_type": "counterfactual_model_rollout",
                        "attack_id": window.get("attack_id"),
                        "p1_target_tag": target,
                        "method": method,
                        "selected_action": action.value,
                        "predicted_safety_violation_duration": result["violations"],
                        "predicted_max_LIT101_overshoot": result["max_over"],
                        "predicted_max_LIT101_undershoot": result["max_under"],
                        "predicted_time_to_target": result["time_to_target"],
                        "predicted_production_proxy": result["production"],
                        "predicted_action_cost": action_cost(action),
                        "shield_interventions": int(action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS),
                    }
                )
        for method, action, result in method_results:
            if method.startswith("B5_"):
                summary_rows.append(
                    {
                        "evaluation_type": "counterfactual_model_rollout",
                        "attack_id": window.get("attack_id"),
                        "p1_target_tag": target,
                        "method": method,
                        "selected_action": action.value,
                        "predicted_safety_violation_duration": result["violations"],
                        "predicted_max_LIT101_overshoot": result["max_over"],
                        "predicted_max_LIT101_undershoot": result["max_under"],
                        "predicted_time_to_target": result["time_to_target"],
                        "predicted_production_proxy": result["production"],
                        "predicted_action_cost": action_cost(action),
                        "shield_interventions": int(action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS),
                    }
                )
    rollout_df = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    rollout_df.to_csv(output / "real_swat_counterfactual_action_timeline.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_by_attack.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_summary.csv", index=False)
    rollout_df.to_csv(output / "real_swat_p1_counterfactual_action_timeline.csv", index=False)
    summary.to_csv(output / "real_swat_p1_counterfactual_by_window.csv", index=False)
    summary.to_csv(output / "real_swat_p1_counterfactual_summary.csv", index=False)
    summary[summary["method"].astype(str).str.startswith("B5_")].to_csv(output / "real_swat_p1_ablation_summary.csv", index=False)
    make_counterfactual_plots(rollout_df, output)
    if (output / "real_swat_counterfactual_level_rollouts.png").exists():
        (output / "real_swat_p1_counterfactual_level_rollouts.png").write_bytes((output / "real_swat_counterfactual_level_rollouts.png").read_bytes())
    if (output / "real_swat_counterfactual_actions.png").exists():
        (output / "real_swat_p1_counterfactual_actions.png").write_bytes((output / "real_swat_counterfactual_actions.png").read_bytes())
    return {"output_dir": output, "summary": output / "real_swat_counterfactual_summary.csv"}


def run_hybrid(context: dict[str, Any], config: dict[str, Any], output: Path, quick: bool) -> dict[str, Path]:
    # Import lazily to avoid a module import cycle with experiment.py.
    from .experiment import metrics_for_timeseries, run_experiments
    from .world_model import ActionConditionedWorldModel

    hybrid_cfg = dict(config)
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
    hybrid_cfg.setdefault("experiment", {})
    hybrid_cfg["experiment"]["attacks"] = [
        "LIT101_FDI",
        "LIT101_DRIFT",
        "LIT101_REPLAY",
        "MV101_STUCK_OPEN",
        "P101_FORCED_OFF",
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
    from .experiment import build_diagnosis_config

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
    make_hybrid_plots(timeseries, summarize_metrics(metrics), output)
    return {"output_dir": output, "summary": output / "real_swat_hybrid_summary.csv"}


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
        "## 3. P1 Attack Windows Used",
        p1_windows.to_string(index=False) if not p1_windows.empty else "No P1 windows in loaded range.",
        "",
        "## 4. Thresholds Calibrated From Normal Data",
        json.dumps(thresholds, indent=2),
        "",
        "## 5. P1 Diagnosis Metrics",
        p1_metrics.to_string(index=False),
        "",
        "## 6. Trust Detection By P1 Tag",
        trust_by_tag.to_string(index=False),
        "",
        "## 7. World Model Prediction Metrics",
        world_eval.to_string(index=False),
        "",
        "## 8. Counterfactual Recovery Metrics",
        "See `real_swat_p1_counterfactual_summary.csv` after running counterfactual mode.",
        "",
        "## 9. Hybrid Recovery Metrics",
        "See `real_swat_hybrid_summary.csv` after running hybrid mode.",
        "",
        "## 10. Limitations",
        "- Whole-plant SWaT labels are not primary for a P1-only model.",
        "- Offline logs do not validate closed-loop recovery actions.",
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


def _counterfactual_step(level: float, action: RecoveryAction, calibration: dict[str, Any]) -> float:
    open_fit = float(calibration.get("mv101_fit_open_mean", 1.0))
    closed_fit = float(calibration.get("mv101_fit_closed_mean", 0.0))
    drain = 0.0
    inflow = open_fit * 0.05
    if action in {RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN, RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS}:
        inflow = closed_fit * 0.05
    if action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
        drain = max(0.1, open_fit * 0.08)
    elif action in {RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK}:
        drain = max(0.05, open_fit * 0.03) if level > float(calibration.get("target_high", level + 1)) else 0.0
    return float(level + inflow - drain)


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
) -> dict[str, Any]:
    target_low = float(calibration.get("target_low", safe_low))
    target_high = float(calibration.get("target_high", safe_high))
    level = float(initial_level)
    trajectory: list[float] = []
    violations = 0
    production = 0.0
    max_over = 0.0
    max_under = 0.0
    time_to_target = horizon
    for step in range(horizon):
        level = _counterfactual_step(level, action, calibration)
        trajectory.append(level)
        safe = safe_low <= level <= safe_high
        violations += int(not safe)
        max_over = max(max_over, max(0.0, level - safe_high))
        max_under = max(max_under, max(0.0, safe_low - level))
        if safe and action != RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN:
            production += 1.0
        if time_to_target == horizon and target_low <= level <= target_high:
            time_to_target = step
    return {
        "trajectory": trajectory,
        "violations": violations,
        "production": production,
        "max_over": max_over,
        "max_under": max_under,
        "time_to_target": time_to_target,
    }


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
