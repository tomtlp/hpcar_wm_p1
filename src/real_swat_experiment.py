"""Real SWaT offline evaluation tasks.

Real SWaT logs are not interactive. The log-eval task therefore evaluates
diagnosis, reconstruction, and prediction only. Recovery metrics are produced
only in counterfactual model rollouts or real-calibrated hybrid simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .attacks import AttackConfig
from .causal_logic import diagnose_real_swat_timeseries
from .metrics import summarize_metrics
from .plotting import make_counterfactual_plots, make_hybrid_plots, make_real_swat_log_plots
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


def run_log_eval(context: dict[str, Any], output: Path) -> dict[str, Path]:
    attack_df = context["attack_df"]
    normal_df = context["normal_df"]
    calibration = context["calibration"]
    swat_cfg = context["swat_config"]
    real_model: RealSWaTWorldModel = context["world_model"]
    diagnosed = diagnose_real_swat_timeseries(attack_df, calibration, swat_cfg.get("diagnosis", {}))
    predicted = real_model.predict_frame(diagnosed)
    predicted["evaluation_type"] = "offline_log_diagnosis"
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
                "evaluation_type": "offline_log_diagnosis",
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


def run_counterfactual(context: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Path]:
    attack_df = context["attack_df"]
    windows = context["windows"]
    calibration = context["calibration"]
    horizon = int(context["swat_config"].get("counterfactual", {}).get("horizon", 60))
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    safe_low = float(calibration.get("safe_low", attack_df["LIT101"].quantile(0.05)))
    safe_high = float(calibration.get("safe_high", attack_df["LIT101"].quantile(0.95)))
    for _, window in windows.head(10).iterrows():
        start = int(window["start_index"]) if pd.notna(window.get("start_index")) else 0
        start = max(0, min(start, len(attack_df) - 1))
        initial_level = float(attack_df["LIT101"].iloc[start])
        candidates = [RecoveryAction.R0_KEEP_CURRENT, RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS, RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK]
        action_metrics = []
        for action in candidates:
            level = initial_level
            production = 0.0
            violations = 0
            max_over = 0.0
            for step in range(horizon):
                level = _counterfactual_step(level, action, calibration)
                production += max(0.0, float(calibration.get("mv101_fit_open_mean", 1.0))) if safe_low <= level <= safe_high else 0.0
                violations += int(level < safe_low or level > safe_high)
                max_over = max(max_over, max(0.0, level - safe_high))
                rows.append(
                    {
                        "evaluation_type": "counterfactual_model_rollout",
                        "attack_id": window.get("attack_id"),
                        "candidate_action": action.value,
                        "step": step,
                        "predicted_level": level,
                        "selected": 0,
                    }
                )
            action_metrics.append((violations + 0.1 * action_cost(action) + max_over, action, production, violations, max_over))
        best = min(action_metrics, key=lambda item: item[0])
        for row in rows:
            if row["attack_id"] == window.get("attack_id") and row["candidate_action"] == best[1].value:
                row["selected"] = 1
        summary_rows.append(
            {
                "evaluation_type": "counterfactual_model_rollout",
                "attack_id": window.get("attack_id"),
                "selected_action": best[1].value,
                "predicted_safety_violation_duration": best[3],
                "predicted_max_LIT101_overshoot": best[4],
                "predicted_production_proxy": best[2],
                "predicted_action_cost": action_cost(best[1]),
                "shield_interventions": int(best[1] == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS),
            }
        )
    rollout_df = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    rollout_df.to_csv(output / "real_swat_counterfactual_action_timeline.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_by_attack.csv", index=False)
    summary.to_csv(output / "real_swat_counterfactual_summary.csv", index=False)
    make_counterfactual_plots(rollout_df, output)
    return {"output_dir": output, "summary": output / "real_swat_counterfactual_summary.csv"}


def run_hybrid(context: dict[str, Any], config: dict[str, Any], output: Path, quick: bool) -> dict[str, Path]:
    # Import lazily to avoid a module import cycle with experiment.py.
    from .experiment import metrics_for_timeseries, run_experiments
    from .world_model import ActionConditionedWorldModel

    hybrid_cfg = dict(config)
    sim_cfg = simulator_config_from_calibration(config.get("simulator", {}), context["calibration"])
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
    hybrid_cfg["experiment"]["seeds"] = context["swat_config"].get("hybrid", {}).get("seeds", [0, 1, 2, 3, 4])
    if quick:
        hybrid_cfg["experiment"]["seeds"] = [0]
        hybrid_cfg["experiment"]["steps"] = int(config.get("experiment", {}).get("quick_steps", 70))
    wm = ActionConditionedWorldModel({**sim_cfg, **config.get("world_model", {}), "enabled": False})
    from .experiment import build_diagnosis_config

    diagnosis_config = build_diagnosis_config(hybrid_cfg)
    timeseries = run_experiments(hybrid_cfg, diagnosis_config, wm, initial_level=sim_cfg.get("initial_level"))
    timeseries["evaluation_type"] = "real_calibrated_simulation"
    metrics = metrics_for_timeseries(timeseries, wm.metrics)
    metrics["evaluation_type"] = "real_calibrated_simulation"
    timeseries.to_csv(output / "real_swat_hybrid_timeseries.csv", index=False)
    metrics.to_csv(output / "real_swat_hybrid_metrics_by_method_attack.csv", index=False)
    summarize_metrics(metrics).to_csv(output / "real_swat_hybrid_summary.csv", index=False)
    make_hybrid_plots(timeseries, summarize_metrics(metrics), output)
    return {"output_dir": output, "summary": output / "real_swat_hybrid_summary.csv"}


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
                "evaluation_type": "offline_log_diagnosis",
                "attack_detection_precision": precision,
                "attack_detection_recall": recall,
                "attack_detection_f1": f1,
                "attack_detection_auc": auc,
                "false_alarm_rate_on_normal_validation": false_alarm,
            }
        ]
    )


def real_world_model_eval(df: pd.DataFrame) -> pd.DataFrame:
    target = df["LIT101"].shift(-1)
    pred = df.get("pred_LIT101_next")
    attack_mask = pd.to_numeric(df.get("label"), errors="coerce").eq(1)
    return pd.DataFrame(
        [
            {
                "evaluation_type": "offline_log_diagnosis",
                "one_step_rmse": _rmse(pred, target),
                "attack_period_rmse": _rmse(pred[attack_mask], target[attack_mask]) if attack_mask.any() else np.nan,
                "raw_observation_rmse": _rmse(df["LIT101"], target),
                "full_rollback_rmse": _rmse(df["lit101_est"], target),
                "partial_rollback_rmse": _rmse(df["level_est"], target),
                "trust_aware_reconstruction_rmse": _rmse(df["level_est"], target),
            }
        ]
    )


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
