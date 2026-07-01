"""Experiment metrics for P1 recovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .recovery_actions import RecoveryAction, action_cost


def compute_run_metrics(
    run_df: pd.DataFrame,
    normal_production: float | None = None,
    one_step_rmse: float | None = None,
    multi_step_rmse: float | None = None,
    extra_metrics: dict[str, float] | None = None,
    consecutive_safe_steps: int = 5,
) -> dict[str, Any]:
    if run_df.empty:
        return {}
    attack_start = int(run_df["attack_start"].iloc[0])
    after_attack = run_df[run_df["t"] >= attack_start]
    safe_mask = (after_attack["level_true"] >= 20.0) & (after_attack["level_true"] <= 80.0)
    target_mask = (after_attack["level_true"] >= 45.0) & (after_attack["level_true"] <= 60.0)
    if safe_mask.any():
        time_to_safe = int(after_attack.loc[safe_mask, "t"].iloc[0] - attack_start)
    else:
        time_to_safe = int(run_df["t"].max() - attack_start + 1)
    if target_mask.any():
        time_to_target = int(after_attack.loc[target_mask, "t"].iloc[0] - attack_start)
    else:
        time_to_target = int(run_df["t"].max() - attack_start + 1)

    recover_time, recovered = time_to_recover_after_first_violation(after_attack, consecutive_safe_steps)

    safety_violation_duration = int(((after_attack["level_true"] < 20.0) | (after_attack["level_true"] > 80.0)).sum())
    hard_safety_duration = int(((after_attack["level_true"] < 10.0) | (after_attack["level_true"] > 90.0)).sum())
    production = float(run_df["production"].sum())
    normal_reference = production if normal_production is None else normal_production
    production_loss = max(0.0, float(normal_reference - production))
    trust_acc = trust_mask_accuracy(run_df)
    trust_scores = trust_detection_scores(run_df)

    result = {
        "method": run_df["method"].iloc[0],
        "attack": run_df["attack"].iloc[0],
        "seed": int(run_df["seed"].iloc[0]),
        "time_to_safe_set": time_to_safe,
        "time_to_safe_after_attack": time_to_safe,
        "time_to_target_after_attack": time_to_target,
        "time_to_recover_after_first_violation": recover_time,
        "recovery_success": int(recovered),
        "safety_violation_duration": safety_violation_duration,
        "hard_safety_violation_duration": hard_safety_duration,
        "max_level_overshoot": max(0.0, float(run_df["level_true"].max() - 80.0)),
        "max_level_undershoot": max(0.0, float(20.0 - run_df["level_true"].min())),
        "pump_empty_run_count": int(run_df["pump_empty_run"].sum()),
        "production": production,
        "production_loss": production_loss,
        "action_cost": float(run_df["action"].map(_safe_action_cost).sum()),
        "false_recovery_count": int(((run_df["attack"] == "normal") & (run_df["action"] != RecoveryAction.R0_KEEP_CURRENT.value)).sum()),
        "shield_intervention_count": int(run_df["shield_intervened"].sum()),
        "trust_mask_accuracy": trust_acc,
        **trust_scores,
        "one_step_rmse": float(one_step_rmse) if one_step_rmse is not None else np.nan,
        "multi_step_rollout_rmse": float(multi_step_rmse) if multi_step_rmse is not None else np.nan,
    }
    if extra_metrics:
        result.update(extra_metrics)
    return result


def time_to_recover_after_first_violation(after_attack: pd.DataFrame, consecutive_safe_steps: int) -> tuple[int, bool]:
    """Measure recovery from the first post-attack safe-band violation."""
    if after_attack.empty:
        return 0, True
    violation = (after_attack["level_true"] < 20.0) | (after_attack["level_true"] > 80.0)
    if not violation.any():
        return 0, True
    first_violation_pos = int(np.where(violation.to_numpy())[0][0])
    subset = after_attack.iloc[first_violation_pos:].reset_index(drop=True)
    safe = ((subset["level_true"] >= 20.0) & (subset["level_true"] <= 80.0)).to_numpy()
    needed = max(1, int(consecutive_safe_steps))
    for idx in range(0, len(safe) - needed + 1):
        if bool(np.all(safe[idx : idx + needed])):
            start_t = int(subset["t"].iloc[0])
            recover_t = int(subset["t"].iloc[idx])
            return recover_t - start_t, True
    return int(subset["t"].iloc[-1] - subset["t"].iloc[0] + 1), False


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [col for col in ["seed"] if col in numeric_cols]
    numeric_cols = [col for col in numeric_cols if col not in drop_cols]
    return (
        metrics_df.groupby(["method", "attack"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["attack", "method"])
    )


def trust_mask_accuracy(run_df: pd.DataFrame) -> float:
    pairs = [
        ("trust_LIT101", "gt_trust_LIT101"),
        ("trust_FIT101", "gt_trust_FIT101"),
        ("trust_MV101", "gt_trust_MV101"),
        ("trust_P101", "gt_trust_P101"),
        ("trust_P102", "gt_trust_P102"),
    ]
    scores = []
    for pred, truth in pairs:
        if pred in run_df.columns and truth in run_df.columns:
            scores.append((run_df[pred].round().astype(int) == run_df[truth].round().astype(int)).mean())
    return float(np.mean(scores)) if scores else np.nan


def trust_detection_scores(run_df: pd.DataFrame) -> dict[str, float]:
    """Micro precision/recall/F1 for detecting compromised tags."""
    tags = ["LIT101", "FIT101", "MV101", "P101", "P102", "PLC1"]
    y_true: list[int] = []
    y_pred: list[int] = []
    attack_start = int(run_df["attack_start"].iloc[0]) if "attack_start" in run_df else 0
    eval_df = run_df[run_df["t"] >= attack_start]
    for tag in tags:
        pred_col = f"trust_{tag}"
        truth_col = f"gt_trust_{tag}"
        if pred_col not in eval_df or truth_col not in eval_df:
            continue
        y_true.extend((eval_df[truth_col].round().astype(int) == 0).astype(int).tolist())
        y_pred.extend((eval_df[pred_col].round().astype(int) == 0).astype(int).tolist())
    if not y_true:
        return {"trust_detection_precision": np.nan, "trust_detection_recall": np.nan, "trust_detection_f1": np.nan}
    true = np.array(y_true, dtype=int)
    pred = np.array(y_pred, dtype=int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else np.nan
    return {
        "trust_detection_precision": float(precision) if precision == precision else np.nan,
        "trust_detection_recall": float(recall) if recall == recall else np.nan,
        "trust_detection_f1": float(f1) if f1 == f1 else np.nan,
    }


def _safe_action_cost(name: str) -> float:
    try:
        return action_cost(RecoveryAction(name))
    except Exception:
        return 0.0
