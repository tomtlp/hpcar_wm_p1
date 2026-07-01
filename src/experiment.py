"""Command-line experiment runner for the SWaT P1 recovery MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attacks import ATTACK_SCENARIOS, AttackConfig, create_attack
from .baselines import METHODS, choose_action
from .causal_logic import BELIEF_COLUMNS, diagnose, reconstruct_belief_state
from .data_loader import load_swat_csv, swat_initial_level
from .metrics import compute_run_metrics, summarize_metrics
from .p1_simulator import P1Simulator
from .planner import HazardPrioritizedPlanner
from .plotting import make_plots
from .recovery_actions import RecoveryAction, all_recovery_actions
from .safety_shield import SafetyShield
from .utils import clamp, load_config, output_path, set_seed
from .world_model import ActionConditionedWorldModel, make_model_input


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    out_dir = output_path(config, args.output_dir)
    set_seed(int(config["experiment"]["seeds"][0]))

    if args.mode == "real_swat":
        from .real_swat_experiment import run_real_swat_task

        swat_dir = args.swat_dir or config.get("swat", {}).get("data_dir", "dataset/SWat")
        return run_real_swat_task(config, swat_dir, args.real_swat_task, out_dir, args.quick)

    initial_level = maybe_load_csv(args, config)
    diagnosis_config = build_diagnosis_config(config)
    world_model = ActionConditionedWorldModel(build_world_model_config(config))
    train_x, train_y = generate_world_model_training_data(config, diagnosis_config, args.quick)
    wm_metrics = world_model.train(train_x, train_y, out_dir / "world_model.pt")

    timeseries = run_experiments(config, diagnosis_config, world_model, initial_level)
    world_eval = evaluate_world_model_on_timeseries(timeseries, world_model)
    per_run_metrics = metrics_for_timeseries(
        timeseries,
        wm_metrics,
        world_eval,
        int(config.get("experiment", {}).get("consecutive_safe_steps", 5)),
    )
    summary = summarize_metrics(per_run_metrics)
    attack_only = per_run_metrics[per_run_metrics["attack"] != "normal"]
    attack_only_summary = (
        attack_only.groupby("method", as_index=False)
        .mean(numeric_only=True)
        .sort_values("safety_violation_duration")
        if not attack_only.empty
        else pd.DataFrame()
    )
    method_numeric = [
        col
        for col in per_run_metrics.select_dtypes(include=[np.number]).columns
        if col != "seed"
    ]
    method_summary = per_run_metrics.groupby("method", as_index=False)[method_numeric].mean(numeric_only=True)

    timeseries_path = out_dir / "per_run_timeseries.csv"
    metrics_path = out_dir / "metrics_by_method_attack.csv"
    summary_path = out_dir / "results_summary.csv"
    summary_all_path = out_dir / "results_summary_all.csv"
    attack_summary_path = out_dir / "results_summary_attack_only.csv"
    best_safety_path = out_dir / "per_attack_best_method_by_safety.csv"
    best_production_path = out_dir / "per_attack_best_method_by_production.csv"
    trust_by_tag_path = out_dir / "trust_detection_by_tag.csv"
    world_eval_path = out_dir / "world_model_eval.csv"
    timeseries.to_csv(timeseries_path, index=False)
    summary.to_csv(metrics_path, index=False)
    attack_only_summary.to_csv(summary_path, index=False)
    method_summary.to_csv(summary_all_path, index=False)
    attack_only_summary.to_csv(attack_summary_path, index=False)
    best_methods(summary, "safety_violation_duration").to_csv(best_safety_path, index=False)
    best_methods(summary, "production_loss").to_csv(best_production_path, index=False)
    trust_detection_by_tag(timeseries).to_csv(trust_by_tag_path, index=False)
    world_eval.to_csv(world_eval_path, index=False)
    make_plots(timeseries, summary, out_dir)

    print(f"[experiment] Wrote {timeseries_path}")
    print(f"[experiment] Wrote {metrics_path}")
    print(f"[experiment] Wrote {summary_path}")
    print(f"[experiment] Wrote {attack_summary_path}")
    print(f"[experiment] Wrote {world_eval_path}")
    print(f"[experiment] Plots written under {out_dir}")
    return {
        "timeseries": timeseries_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "summary_all": summary_all_path,
        "summary_attack_only": attack_summary_path,
        "world_eval": world_eval_path,
        "output_dir": out_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPCAR-WM P1 recovery experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["synthetic", "swat_csv", "real_swat"], default="synthetic")
    parser.add_argument("--csv_path", default=None, help="Optional SWaT CSV path")
    parser.add_argument("--swat_dir", default=None, help="Directory containing real SWaT files")
    parser.add_argument(
        "--real_swat_task",
        choices=["log_eval", "p1_log_eval", "counterfactual", "hybrid"],
        default="log_eval",
        help="Real SWaT offline evaluation task",
    )
    parser.add_argument("--quick", action="store_true", help="Run a short CPU-friendly experiment")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("experiment", {})
    config.setdefault("world_model", {})
    if args.quick:
        config["experiment"]["seeds"] = [0]
        config["experiment"]["steps"] = int(config["experiment"].get("quick_steps", 70))
        config["world_model"]["train_rollouts"] = min(int(config["world_model"].get("train_rollouts", 60)), 10)
        config["world_model"]["train_steps"] = min(int(config["world_model"].get("train_steps", 70)), 35)
        config["world_model"]["epochs"] = min(int(config["world_model"].get("epochs", 8)), 3)
        config["world_model"]["hidden_dim"] = min(int(config["world_model"].get("hidden_dim", 48)), 32)
    return config


def maybe_load_csv(args: argparse.Namespace, config: dict[str, Any]) -> float | None:
    if args.mode != "swat_csv":
        return None
    df, mapping = load_swat_csv(args.csv_path)
    level = swat_initial_level(df, mapping)
    if level is not None and 0.0 <= level <= 100.0:
        print(f"[experiment] Using CSV-derived initial level: {level:.2f}")
        return level
    if level is not None:
        print(
            "[experiment] CSV LIT101 scale is outside the 0-100 MVP range; "
            "using configured synthetic initial level."
        )
    return None


def build_diagnosis_config(config: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ["simulator", "diagnosis", "hazard"]:
        merged.update(config.get(key, {}))
    merged["hazard_weights"] = config.get("hazard", {})
    return merged


def build_world_model_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("simulator", {}))
    merged.update(config.get("world_model", {}))
    return merged


def generate_world_model_training_data(
    config: dict[str, Any],
    diagnosis_config: dict[str, Any],
    quick: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sim_cfg = config.get("simulator", {})
    wm_cfg = config.get("world_model", {})
    attack_cfg = AttackConfig.from_dict(config.get("attacks", {}))
    rollouts = int(wm_cfg.get("train_rollouts", 10 if quick else 60))
    steps = int(wm_cfg.get("train_steps", 35 if quick else 70))
    rng = np.random.default_rng(123)
    actions = all_recovery_actions()
    attack_pool = ["normal", "LIT101_FDI", "LIT101_DRIFT", "MV101_STUCK_OPEN", "P101_FORCED_OFF"]
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for rollout in range(rollouts):
        sim = P1Simulator(sim_cfg, seed=1000 + rollout)
        attack_name = attack_pool[int(rng.integers(0, len(attack_pool)))]
        attack = create_attack(attack_name, attack_cfg)
        history: list[dict[str, Any]] = []
        state = sim.reset(seed=1000 + rollout, initial_level=float(rng.uniform(35.0, 68.0)))
        for _ in range(steps):
            current = state.to_dict()
            diagnostics = diagnose(current, history, diagnosis_config)
            belief_vec, belief = reconstruct_belief_state(current, history, diagnostics, diagnosis_config)
            action = actions[int(rng.integers(0, len(actions)))]
            next_state, info = sim.step(action, attack, level_est=belief["level_est"], recovery_context=belief)
            risk = safety_risk(next_state.level_true, next_state.p101_state, next_state.p102_state)
            x_rows.append(make_model_input(belief_vec, action))
            y_rows.append(np.array([next_state.level_true, next_state.fit_true, risk, info["production"]], dtype=np.float32))
            history_row = dict(current)
            history_row["level_est"] = belief["level_est"]
            history_row["mass_balance_level"] = diagnostics.mass_balance_level
            for key, value in diagnostics.residuals.items():
                history_row[f"residual_{key}"] = value
            history.append(history_row)
            state = next_state

    x = np.vstack(x_rows).astype(np.float32)
    y = np.vstack(y_rows).astype(np.float32)
    print(f"[experiment] Generated {len(x)} world-model training samples.")
    return x, y


def run_experiments(
    config: dict[str, Any],
    diagnosis_config: dict[str, Any],
    world_model: ActionConditionedWorldModel,
    initial_level: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sim_cfg = config.get("simulator", {})
    attack_cfg = AttackConfig.from_dict(config.get("attacks", {}))
    methods = config.get("experiment", {}).get("methods", METHODS)
    attacks = config.get("experiment", {}).get("attacks", ATTACK_SCENARIOS)
    seeds = config.get("experiment", {}).get("seeds", [0])
    steps = int(config.get("experiment", {}).get("steps", 70))

    total_runs = len(methods) * len(attacks) * len(seeds)
    run_idx = 0
    for method in methods:
        for attack_name in attacks:
            for seed in seeds:
                run_idx += 1
                print(f"[experiment] Run {run_idx}/{total_runs}: {method} on {attack_name}, seed={seed}")
                rows.extend(
                    run_single(
                        method=method,
                        attack_name=attack_name,
                        seed=int(seed),
                        steps=steps,
                        sim_cfg=sim_cfg,
                        attack_cfg=attack_cfg,
                        diagnosis_config=diagnosis_config,
                        planner_cfg=config.get("planner", {}),
                        shield_cfg=config.get("shield", {}),
                        world_model=world_model,
                        initial_level=initial_level,
                    )
                )
    return pd.DataFrame(rows)


def run_single(
    method: str,
    attack_name: str,
    seed: int,
    steps: int,
    sim_cfg: dict[str, Any],
    attack_cfg: AttackConfig,
    diagnosis_config: dict[str, Any],
    planner_cfg: dict[str, Any],
    shield_cfg: dict[str, Any],
    world_model: ActionConditionedWorldModel,
    initial_level: float | None,
) -> list[dict[str, Any]]:
    set_seed(seed)
    sim = P1Simulator(sim_cfg, seed=seed)
    state = sim.reset(seed=seed, initial_level=initial_level)
    attack = create_attack(attack_name, attack_cfg)
    proposed_planner = HazardPrioritizedPlanner(
        world_model=world_model,
        shield=SafetyShield(shield_cfg),
        planner_config=planner_cfg,
        diagnosis_config=diagnosis_config,
        use_trust=True,
    )
    no_trust_planner = HazardPrioritizedPlanner(
        world_model=world_model,
        shield=SafetyShield(shield_cfg),
        planner_config=planner_cfg,
        diagnosis_config=diagnosis_config,
        use_trust=False,
    )
    baseline_shield = SafetyShield(shield_cfg)
    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for _ in range(steps):
        current = state.to_dict()
        attack_detected = attack_name != "normal" and int(current["t"]) >= int(attack_cfg.start_step)
        decision = choose_action(
            method=method,
            current_state=current,
            history=history,
            proposed_planner=proposed_planner,
            no_trust_planner=no_trust_planner,
            shield=baseline_shield,
            diagnosis_config=diagnosis_config,
            attack_detected=attack_detected,
        )
        action = decision.action
        belief = decision.belief
        diagnostics = decision.diagnostics
        next_state, info = sim.step(action, attack, level_est=belief.get("level_est"), recovery_context=belief)
        gt_trust = attack.ground_truth_trust(next_state.t)
        rows.append(
            make_row(
                method=method,
                attack_name=attack_name,
                seed=seed,
                attack_start=int(attack_cfg.start_step),
                action=action,
                decision=decision,
                state=next_state.to_dict(),
                info=info,
                belief=belief,
                diagnostics=diagnostics,
                gt_trust=gt_trust,
            )
        )
        history_row = dict(current)
        history_row["level_est"] = belief.get("level_est", current.get("lit101_obs", 50.0))
        history_row["mass_balance_level"] = diagnostics.mass_balance_level
        for key, value in diagnostics.residuals.items():
            history_row[f"residual_{key}"] = value
        history.append(history_row)
        state = next_state
    return rows


def make_row(
    method: str,
    attack_name: str,
    seed: int,
    attack_start: int,
    action: RecoveryAction,
    decision: Any,
    state: dict[str, Any],
    info: dict[str, Any],
    belief: dict[str, float],
    diagnostics: Any,
    gt_trust: dict[str, int],
) -> dict[str, Any]:
    shield_intervened = bool(getattr(decision, "shield_intervened", False))
    shield_reason = str(getattr(decision, "shield_reason", ""))
    if hasattr(decision, "shield"):
        shield_intervened = bool(decision.shield.intervened)
        shield_reason = decision.shield.reason
    row = {
        "method": method,
        "attack": attack_name,
        "seed": seed,
        "attack_start": attack_start,
        "action": action.value,
        "requested_action": getattr(decision, "requested_action", action).value,
        "shield_intervened": int(shield_intervened),
        "shield_reason": shield_reason,
        **state,
        **info,
        "level_est": belief.get("level_est", np.nan),
        "fit_est": belief.get("fit_est", np.nan),
        "hazard_priority": belief.get("hazard_priority", np.nan),
        "attack_belief_score": belief.get("attack_belief_score", np.nan),
        "causal_score": getattr(diagnostics, "causal_score", np.nan),
        "logic_violation_score": getattr(diagnostics, "logic_violation_score", np.nan),
        "mass_balance_level": getattr(diagnostics, "mass_balance_level", np.nan),
        "root_causes": "|".join(getattr(diagnostics, "root_causes", [])),
    }
    for key in ["LIT101", "FIT101", "MV101", "P101", "P102", "PLC1"]:
        row[f"trust_{key}"] = int(diagnostics.trust_mask.get(key, 1))
        row[f"gt_trust_{key}"] = int(gt_trust.get(key, 1))
    for key, value in diagnostics.residuals.items():
        row[f"residual_{key}"] = value
    return row


def metrics_for_timeseries(
    timeseries: pd.DataFrame,
    wm_metrics: dict[str, float],
    world_eval: pd.DataFrame | None = None,
    consecutive_safe_steps: int = 5,
) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame()
    normal_refs = (
        timeseries[(timeseries["attack"] == "normal") & (timeseries["method"] == "B5_PROPOSED")]
        .groupby("seed")["production"]
        .sum()
        .to_dict()
    )
    eval_lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    if world_eval is not None and not world_eval.empty:
        for _, row in world_eval.iterrows():
            key = (str(row["method"]), str(row["attack"]), int(row["seed"]))
            eval_lookup[key] = {
                col: float(row[col])
                for col in world_eval.columns
                if col not in {"method", "attack", "seed"}
            }
    rows = []
    for (method, attack, seed), group in timeseries.groupby(["method", "attack", "seed"]):
        extra = eval_lookup.get((method, attack, int(seed)), {})
        one_step = wm_metrics.get("one_step_rmse")
        multi_step = wm_metrics.get("multi_step_rollout_rmse")
        if one_step is None or pd.isna(one_step):
            one_step = extra.get("one_step_rmse")
        if multi_step is None or pd.isna(multi_step):
            multi_step = extra.get("multi_step_rollout_rmse")
        rows.append(
            compute_run_metrics(
                group,
                normal_production=normal_refs.get(seed),
                one_step_rmse=one_step,
                multi_step_rmse=multi_step,
                extra_metrics=extra,
                consecutive_safe_steps=consecutive_safe_steps,
            )
        )
    return pd.DataFrame(rows)


def evaluate_world_model_on_timeseries(
    timeseries: pd.DataFrame,
    world_model: ActionConditionedWorldModel,
    rollout_horizon: int = 5,
) -> pd.DataFrame:
    """Evaluate model prediction and reconstruction quality from saved traces."""
    if timeseries.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (method, attack, seed), group in timeseries.groupby(["method", "attack", "seed"]):
        group = group.sort_values("t").reset_index(drop=True)
        pred_level: list[float] = []
        pred_fit: list[float] = []
        true_level: list[float] = []
        true_fit: list[float] = []
        pred_attack_level: list[float] = []
        true_attack_level: list[float] = []
        attack_start = int(group["attack_start"].iloc[0])
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            cur = group.iloc[idx]
            pred = world_model.predict(_belief_from_row(prev), RecoveryAction(str(prev["action"])))
            pred_level.append(pred.level_est_next)
            pred_fit.append(pred.fit_next)
            true_level.append(float(cur["level_true"]))
            true_fit.append(float(cur["fit_true"]))
            if int(cur["t"]) >= attack_start:
                pred_attack_level.append(pred.level_est_next)
                true_attack_level.append(float(cur["level_true"]))

        multi_pred: list[float] = []
        multi_true: list[float] = []
        horizon = max(1, int(rollout_horizon))
        for idx in range(0, max(0, len(group) - horizon), horizon):
            belief = _belief_from_row(group.iloc[idx])
            for step in range(1, horizon + 1):
                action = RecoveryAction(str(group.iloc[idx + step - 1]["action"]))
                pred = world_model.predict(belief, action)
                belief["level_est"] = pred.level_est_next
                belief["fit_est"] = pred.fit_next
                if idx + step < len(group):
                    multi_pred.append(pred.level_est_next)
                    multi_true.append(float(group.iloc[idx + step]["level_true"]))

        after_attack = group[group["t"] >= attack_start]
        row = {
            "method": method,
            "attack": attack,
            "seed": int(seed),
            "one_step_rmse": _rmse_pair(pred_level, true_level),
            "multi_step_rollout_rmse": _rmse_pair(multi_pred, multi_true),
            "attack_period_rmse": _rmse_pair(pred_attack_level, true_attack_level),
            "trust_aware_reconstruction_rmse": _rmse_series(after_attack["level_est"], after_attack["level_true"]),
            "raw_observation_rmse": _rmse_series(after_attack["lit101_obs"], after_attack["level_true"]),
            "full_rollback_rmse": _rmse_series(after_attack["mass_balance_level"], after_attack["level_true"]),
            "partial_rollback_rmse": _rmse_series(after_attack["level_est"], after_attack["level_true"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _belief_from_row(row: pd.Series) -> dict[str, float]:
    values = {
        "level_est": row.get("level_est", row.get("lit101_obs", 50.0)),
        "fit_est": row.get("fit_est", row.get("fit101_obs", 0.0)),
        "mv101_state": row.get("mv101_state", 0.0),
        "p101_state": row.get("p101_state", 0.0),
        "p102_state": row.get("p102_state", 0.0),
        "trust_LIT101": row.get("trust_LIT101", 1.0),
        "trust_FIT101": row.get("trust_FIT101", 1.0),
        "trust_MV101": row.get("trust_MV101", 1.0),
        "trust_P101": row.get("trust_P101", 1.0),
        "trust_P102": row.get("trust_P102", 1.0),
        "hazard_priority": row.get("hazard_priority", 0.0),
        "attack_belief_score": row.get("attack_belief_score", 0.0),
    }
    return {key: float(values[key]) for key in BELIEF_COLUMNS}


def trust_detection_by_tag(timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tags = ["LIT101", "FIT101", "MV101", "P101", "P102", "PLC1"]
    for method, attack, tag, group in _tag_groups(timeseries, tags):
        pred = (group[f"trust_{tag}"].round().astype(int) == 0).astype(int)
        truth = (group[f"gt_trust_{tag}"].round().astype(int) == 0).astype(int)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else np.nan
        rows.append(
            {
                "method": method,
                "attack": attack,
                "tag": tag,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


def _tag_groups(timeseries: pd.DataFrame, tags: list[str]):
    for (method, attack), group in timeseries.groupby(["method", "attack"]):
        attack_start = int(group["attack_start"].iloc[0])
        eval_group = group[group["t"] >= attack_start]
        for tag in tags:
            if f"trust_{tag}" in eval_group and f"gt_trust_{tag}" in eval_group:
                yield method, attack, tag, eval_group


def best_methods(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    attack_summary = summary[summary["attack"] != "normal"].copy()
    if attack_summary.empty:
        return pd.DataFrame()
    idx = attack_summary.groupby("attack")[metric].idxmin()
    return attack_summary.loc[idx, ["attack", "method", metric]].sort_values("attack")


def _rmse_pair(pred: list[float], truth: list[float]) -> float:
    if not pred or not truth:
        return np.nan
    pred_arr = np.array(pred, dtype=float)
    truth_arr = np.array(truth, dtype=float)
    return float(np.sqrt(np.mean((pred_arr - truth_arr) ** 2)))


def _rmse_series(pred: pd.Series, truth: pd.Series) -> float:
    pred_arr = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    truth_arr = pd.to_numeric(truth, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(truth_arr)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((pred_arr[mask] - truth_arr[mask]) ** 2)))


def safety_risk(level: float, p101: int, p102: int) -> float:
    hard = level < 10.0 or level > 90.0
    empty = (p101 or p102) and level < 15.0
    soft = max(0.0, 20.0 - level, level - 80.0) / 20.0
    return clamp(float(hard or empty) + soft, 0.0, 2.0)


if __name__ == "__main__":
    main()
