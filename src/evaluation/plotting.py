"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def make_plots(timeseries: pd.DataFrame, metrics: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if timeseries.empty or metrics.empty:
        return
    _level_trajectories(timeseries, output / "level_trajectories.png")
    _action_timeline(timeseries, output / "action_timeline.png")
    _bar(metrics, "safety_violation_duration", output / "safety_violations_bar.png")
    _bar(metrics, "production_loss", output / "production_loss_bar.png")
    _bar(metrics, "shield_intervention_count", output / "shield_interventions_bar.png")
    _trust_example(timeseries, output / "trust_mask_example.png")


def _level_trajectories(df: pd.DataFrame, path: Path) -> None:
    sample = df[(df["seed"] == df["seed"].min()) & (df["attack"].isin(["normal", "COMBINED_LIT101_FDI_MV101_OPEN"]))]
    plt.figure(figsize=(10, 5))
    for (method, attack), group in sample.groupby(["method", "attack"]):
        plt.plot(group["t"], group["level_true"], label=f"{method}/{attack}", linewidth=1.2)
    plt.axhspan(20, 80, color="#d6eadf", alpha=0.35, label="safe band")
    plt.axhline(10, color="#b23a48", linestyle="--", linewidth=0.8)
    plt.axhline(90, color="#b23a48", linestyle="--", linewidth=0.8)
    plt.xlabel("time step")
    plt.ylabel("T101 true level")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _action_timeline(df: pd.DataFrame, path: Path) -> None:
    sample = df[(df["seed"] == df["seed"].min()) & (df["attack"] == "COMBINED_LIT101_FDI_MV101_OPEN")]
    if sample.empty:
        sample = df[df["seed"] == df["seed"].min()]
    actions = {name: idx for idx, name in enumerate(sorted(sample["action"].unique()))}
    plt.figure(figsize=(10, 5))
    for method, group in sample.groupby("method"):
        y = [actions[action] for action in group["action"]]
        plt.step(group["t"], y, where="post", label=method, linewidth=1.1)
    plt.yticks(list(actions.values()), list(actions.keys()), fontsize=7)
    plt.xlabel("time step")
    plt.ylabel("recovery action")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bar(metrics: pd.DataFrame, column: str, path: Path) -> None:
    pivot = metrics.groupby("method", as_index=False)[column].mean(numeric_only=True)
    plt.figure(figsize=(8, 4))
    plt.bar(pivot["method"], pivot[column], color="#4f7cac")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _trust_example(df: pd.DataFrame, path: Path) -> None:
    sample = df[(df["method"] == "B5_PROPOSED") & (df["seed"] == df["seed"].min())]
    if sample.empty:
        sample = df[df["seed"] == df["seed"].min()]
    cols = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102"]
    sample = sample[sample["attack"] == "COMBINED_LIT101_FDI_MV101_OPEN"]
    if sample.empty:
        sample = df[df["seed"] == df["seed"].min()]
    plt.figure(figsize=(10, 4))
    for col in cols:
        if col in sample.columns:
            plt.step(sample["t"], sample[col], where="post", label=col)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("time step")
    plt.ylabel("trust")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_real_swat_log_plots(timeseries: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if timeseries.empty:
        return
    x = timeseries["t"] if "t" in timeseries else range(len(timeseries))
    plt.figure(figsize=(11, 5))
    plt.plot(x, timeseries["LIT101"], label="LIT101 observed", linewidth=1.0)
    if "lit101_est" in timeseries:
        plt.plot(x, timeseries["lit101_est"], label="estimated", linewidth=1.0)
    if "level_est" in timeseries:
        plt.plot(x, timeseries["level_est"], label="reconstructed", linewidth=1.0)
    if "label" in timeseries:
        _shade_attacks(x, timeseries["label"])
    plt.legend(fontsize=8)
    plt.xlabel("index")
    plt.ylabel("LIT101")
    plt.tight_layout()
    plt.savefig(output / "real_swat_level_prediction.png", dpi=160)
    plt.close()

    trust_cols = [c for c in ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102", "trust_PLC1"] if c in timeseries]
    if trust_cols:
        plt.figure(figsize=(11, 3.8))
        data = timeseries[trust_cols].to_numpy(dtype=float).T
        plt.imshow(data, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        plt.yticks(range(len(trust_cols)), trust_cols)
        plt.xlabel("index")
        plt.colorbar(label="trust")
        plt.tight_layout()
        plt.savefig(output / "real_swat_trust_mask.png", dpi=160)
        plt.close()

    residual_cols = [c for c in ["abs_residual_LIT101", "lit101_residual_ewma", "lit101_residual_cusum"] if c in timeseries]
    if residual_cols:
        plt.figure(figsize=(11, 4.5))
        for col in residual_cols:
            plt.plot(x, timeseries[col], label=col, linewidth=1.0)
        if "label" in timeseries:
            _shade_attacks(x, timeseries["label"])
        plt.legend(fontsize=8)
        plt.xlabel("index")
        plt.ylabel("residual")
        plt.tight_layout()
        plt.savefig(output / "real_swat_residuals.png", dpi=160)
        plt.close()

    plt.figure(figsize=(11, 3.8))
    if "label" in timeseries:
        plt.plot(x, timeseries["label"].fillna(0), label="attack label", linewidth=1.0)
    if "attack_belief_score" in timeseries:
        plt.plot(x, timeseries["attack_belief_score"], label="attack belief", linewidth=1.0)
    plt.legend(fontsize=8)
    plt.xlabel("index")
    plt.tight_layout()
    plt.savefig(output / "real_swat_attack_windows_overlay.png", dpi=160)
    plt.close()


def make_counterfactual_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    for (attack_id, action), group in df.groupby(["attack_id", "candidate_action"]):
        plt.plot(group["step"], group["predicted_level"], label=f"{attack_id}/{action}", linewidth=1.0)
    plt.legend(fontsize=6, ncol=2)
    plt.xlabel("rollout step")
    plt.ylabel("predicted LIT101")
    plt.tight_layout()
    plt.savefig(output / "real_swat_counterfactual_level_rollouts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    selected = df[df.get("selected", 0) == 1] if "selected" in df else df
    labels = selected["candidate_action"].value_counts()
    plt.bar(labels.index.astype(str), labels.values, color="#4f7cac")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.ylabel("selected count")
    plt.tight_layout()
    plt.savefig(output / "real_swat_counterfactual_actions.png", dpi=160)
    plt.close()


def make_hybrid_plots(timeseries: pd.DataFrame, metrics: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if timeseries.empty:
        return
    _level_trajectories(timeseries.rename(columns={"level_true": "level_true"}), output / "real_swat_hybrid_level_trajectories.png")
    if not metrics.empty:
        _bar(metrics, "production_loss", output / "real_swat_hybrid_production_loss_bar.png")
        _bar(metrics, "safety_violation_duration", output / "real_swat_hybrid_safety_violations_bar.png")


def make_p1_residual_threshold_plot(df: pd.DataFrame, thresholds: dict[str, float], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if df.empty or "abs_residual_LIT101" not in df:
        return
    x = df["t"] if "t" in df else range(len(df))
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, df["abs_residual_LIT101"], label="abs residual", linewidth=1.0)
    if "lit101_residual_ewma" in df:
        plt.plot(x, df["lit101_residual_ewma"], label="EWMA", linewidth=1.0)
    plt.axhline(thresholds.get("residual_abs_threshold", 0.0), color="#b23a48", linestyle="--", label="abs threshold")
    plt.axhline(thresholds.get("residual_ewma_threshold", 0.0), color="#4f7cac", linestyle="--", label="EWMA threshold")
    plt.xlabel("index")
    plt.ylabel("LIT101 residual")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "real_swat_p1_residual_thresholds.png", dpi=160)
    plt.close()


def make_hybrid_unit_check_plot(calibration_check: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if calibration_check.empty:
        return
    keys = ["lit_min_normal", "safe_low", "target_low", "target_high", "safe_high", "lit_max_normal"]
    values = [float(calibration_check.iloc[0].get(key, 0.0)) for key in keys]
    plt.figure(figsize=(8, 4))
    plt.plot(keys, values, marker="o", color="#4f7cac")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("LIT101 real units")
    plt.tight_layout()
    plt.savefig(output / "real_swat_hybrid_unit_check.png", dpi=160)
    plt.close()


def make_p1_valid_window_plots(timeseries: pd.DataFrame, p1_windows: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if timeseries.empty:
        return
    x = timeseries["t"] if "t" in timeseries else pd.Series(np.arange(len(timeseries)))
    valid = p1_windows.copy()
    if "exclude_from_eval" in valid:
        valid = valid[~valid["exclude_from_eval"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if "duration" in valid:
        valid = valid[pd.to_numeric(valid["duration"], errors="coerce").fillna(0) > 1]

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, timeseries["LIT101"], label="LIT101 observed", linewidth=1.0)
    if "level_est" in timeseries:
        plt.plot(x, timeseries["level_est"], label="reconstructed", linewidth=1.0)
    for _, row in valid.iterrows():
        if pd.notna(row.get("start_index")) and pd.notna(row.get("end_index")):
            plt.axvspan(int(row["start_index"]), int(row["end_index"]), color="#f2a65a", alpha=0.18)
    plt.xlabel("index")
    plt.ylabel("LIT101")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "real_swat_p1_valid_windows_overlay.png", dpi=160)
    plt.close()

    trust_cols = [c for c in ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102", "trust_PLC1"] if c in timeseries]
    if trust_cols:
        plt.figure(figsize=(11, 3.8))
        plt.imshow(timeseries[trust_cols].to_numpy(dtype=float).T, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        plt.yticks(range(len(trust_cols)), trust_cols)
        plt.xlabel("index")
        plt.colorbar(label="trust")
        plt.tight_layout()
        plt.savefig(output / "real_swat_p1_trust_by_tag_heatmap.png", dpi=160)
        plt.close()

    score_cols = [c for c in ["lit101_suspicion_score", "mv101_suspicion_score", "p101_suspicion_score", "p102_suspicion_score", "plc1_suspicion_score"] if c in timeseries]
    if score_cols:
        plt.figure(figsize=(11, 4.5))
        for col in score_cols:
            plt.plot(x, timeseries[col], label=col, linewidth=0.9)
        plt.xlabel("index")
        plt.ylabel("suspicion score")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(output / "real_swat_p1_root_cause_scores.png", dpi=160)
        plt.close()


def make_p1_counterfactual_candidate_plot(candidate_scores: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if candidate_scores.empty or "predicted_cost" not in candidate_scores:
        return
    selected = candidate_scores[candidate_scores.get("selected", 0).astype(int) == 1].copy()
    if selected.empty:
        selected = candidate_scores.copy()
    selected["label"] = selected["method"].astype(str) + "/" + selected["candidate_action"].astype(str)
    sample = selected.sort_values("predicted_cost").head(20)
    plt.figure(figsize=(11, 5))
    plt.bar(sample["label"], sample["predicted_cost"], color="#4f7cac")
    plt.xticks(rotation=35, ha="right", fontsize=7)
    plt.ylabel("predicted cost")
    plt.tight_layout()
    plt.savefig(output / "real_swat_p1_counterfactual_candidate_scores.png", dpi=160)
    plt.close()


def make_hybrid_action_effect_plots(timeseries: pd.DataFrame, metrics: pd.DataFrame, output_dir: str | Path, stress: bool = False) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not timeseries.empty:
        attack = "MV101_STUCK_OPEN"
        if stress:
            attack = "MV101_STUCK_OPEN_HIGH_LEVEL"
        sample = timeseries[(timeseries["seed"] == timeseries["seed"].min()) & (timeseries["attack"] == attack)]
        if sample.empty:
            sample = timeseries[timeseries["seed"] == timeseries["seed"].min()]
        action_col = "shielded_action" if "shielded_action" in sample else "action"
        actions = {name: idx for idx, name in enumerate(sorted(sample[action_col].astype(str).unique()))}
        plt.figure(figsize=(11, 4.8))
        for method, group in sample.groupby("method"):
            y = [actions[str(action)] for action in group[action_col]]
            plt.step(group["t"], y, where="post", label=method, linewidth=1.0)
        plt.yticks(list(actions.values()), list(actions.keys()), fontsize=7)
        plt.xlabel("step")
        plt.ylabel("action")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        action_name = "real_swat_hybrid_stress_action_timeline_by_method.png" if stress else "real_swat_hybrid_action_timeline_by_method.png"
        plt.savefig(output / action_name, dpi=160)
        plt.close()

    if metrics.empty:
        return
    ablation = metrics[metrics["method"].astype(str).str.startswith("B5_")]
    if not ablation.empty and "safety_violation_duration" in ablation:
        pivot = ablation.groupby("method", as_index=False)["safety_violation_duration"].mean(numeric_only=True)
        plt.figure(figsize=(8, 4))
        plt.bar(pivot["method"], pivot["safety_violation_duration"], color="#4f7cac")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.ylabel("safety violation duration")
        plt.tight_layout()
        ablation_name = "real_swat_hybrid_stress_ablation_bar.png" if stress else "real_swat_hybrid_ablation_bar.png"
        plt.savefig(output / ablation_name, dpi=160)
        plt.close()

    if stress and "production_loss" in metrics:
        pivot = metrics.groupby("method", as_index=False)["production_loss"].mean(numeric_only=True)
        plt.figure(figsize=(9, 4.2))
        plt.bar(pivot["method"], pivot["production_loss"], color="#7a9e7e")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.ylabel("production loss")
        plt.tight_layout()
        plt.savefig(output / "real_swat_hybrid_stress_bar.png", dpi=160)
        plt.close()


def _shade_attacks(x: pd.Series | range, labels: pd.Series) -> None:
    arr = labels.fillna(0).to_numpy()
    xs = np.asarray(list(x))
    if len(arr) != len(xs):
        return
    active = arr == 1
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        if start is not None and (not flag or idx == len(active) - 1):
            end = idx if not flag else idx + 1
            plt.axvspan(xs[start], xs[min(end - 1, len(xs) - 1)], color="#f2a65a", alpha=0.2)
            start = None
