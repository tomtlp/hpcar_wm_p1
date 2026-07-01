"""Lightweight causal and rule logic for P1 diagnosis.

This module is not a causal discovery engine. It encodes the P1 causal graph
and PLC rules directly, then uses residuals to build a trust mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import clamp


TRUST_KEYS = ["LIT101", "FIT101", "MV101", "P101", "P102", "PLC1"]
BELIEF_COLUMNS = [
    "level_est",
    "fit_est",
    "mv101_state",
    "p101_state",
    "p102_state",
    "trust_LIT101",
    "trust_FIT101",
    "trust_MV101",
    "trust_P101",
    "trust_P102",
    "hazard_priority",
    "attack_belief_score",
]


@dataclass
class CausalDiagnostics:
    trust_mask: dict[str, int]
    residuals: dict[str, float]
    rule_violations: dict[str, int]
    causal_score: float
    logic_violation_score: float
    attack_belief_score: float
    hazard_priority: float
    mass_balance_level: float
    root_causes: list[str]


def get_value(state: Any, key: str, default: Any = 0.0) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def estimate_level_from_mass_balance(
    prev_level_est: float,
    fit101_obs: float,
    p101_state: int,
    p102_state: int,
    dt: float,
    tank_area: float = 1.0,
    p101_outflow_rate: float = 0.72,
    p102_outflow_rate: float = 0.66,
    level_min: float = 0.0,
    level_max: float = 100.0,
) -> float:
    """Estimate next level from inflow observation and pump states."""
    outflow = int(p101_state) * p101_outflow_rate + int(p102_state) * p102_outflow_rate
    level = float(prev_level_est) + float(dt) * (float(fit101_obs) - outflow) / float(tank_area)
    return clamp(level, level_min, level_max)


def predicted_inflow_from_mv101(mv101_state: int, config: dict[str, Any] | None = None) -> float:
    cfg = config or {}
    return float(cfg.get("inflow_rate_open", 1.2) if int(mv101_state) else cfg.get("inflow_rate_closed", 0.02))


def _mass_level_for_current(
    current_state: Any,
    history: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
) -> float:
    cfg = config or {}
    dt = float(cfg.get("dt", 1.0))
    if history:
        prev = history[-1]
        prev_level = float(prev.get("level_est", prev.get("lit101_obs", get_value(current_state, "lit101_obs", 0.0))))
        return estimate_level_from_mass_balance(
            prev_level,
            float(prev.get("fit101_obs", get_value(current_state, "fit101_obs", 0.0))),
            int(prev.get("p101_state", get_value(current_state, "p101_state", 0))),
            int(prev.get("p102_state", get_value(current_state, "p102_state", 0))),
            dt=dt,
            tank_area=float(cfg.get("tank_area", 1.0)),
            p101_outflow_rate=float(cfg.get("p101_outflow_rate", 0.72)),
            p102_outflow_rate=float(cfg.get("p102_outflow_rate", 0.66)),
            level_min=float(cfg.get("level_min", 0.0)),
            level_max=float(cfg.get("level_max", 100.0)),
        )
    # In tests we may have true state available. Real experiment logic uses history.
    if get_value(current_state, "level_true", None) is not None:
        return float(get_value(current_state, "level_true", 0.0))
    return float(get_value(current_state, "lit101_obs", 0.0))


def compute_causal_residuals(
    window: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute simple residuals over a recent state window."""
    if not window:
        return {"LIT101": 0.0, "FIT101": 0.0, "MV101": 0.0, "P101": 0.0, "P102": 0.0}
    current = window[-1]
    history = window[:-1]
    cfg = config or {}
    mass_level = _mass_level_for_current(current, history, cfg)
    lit_residual = abs(float(get_value(current, "lit101_obs", 0.0)) - mass_level)
    fit_expected = predicted_inflow_from_mv101(int(get_value(current, "mv101_state", 0)), cfg)
    fit_residual = abs(float(get_value(current, "fit101_obs", 0.0)) - fit_expected)
    temporal = compute_lit101_temporal_features(window, cfg)
    return {
        "LIT101": float(lit_residual),
        "FIT101": float(fit_residual),
        "MV101": float(abs(int(get_value(current, "mv101_command", 0)) - int(get_value(current, "mv101_state", 0)))),
        "P101": float(abs(int(get_value(current, "p101_command", 0)) - int(get_value(current, "p101_state", 0)))),
        "P102": float(abs(int(get_value(current, "p102_command", 0)) - int(get_value(current, "p102_state", 0)))),
        **temporal,
    }


def compute_lit101_temporal_features(
    window: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Temporal LIT101 residual features for slow drift and replay detection."""
    cfg = config or {}
    if len(window) < 3:
        return {
            "LIT101_residual_mean": 0.0,
            "LIT101_residual_ewma": 0.0,
            "LIT101_residual_cusum": 0.0,
            "LIT101_residual_slope": 0.0,
            "LIT101_multistep_residual": 0.0,
            "LIT101_replay_score": 0.0,
            "LIT101_consecutive_high_residual": 0.0,
        }

    estimates = _mass_balance_series(window, cfg)
    obs = np.array([float(get_value(row, "lit101_obs", estimates[i])) for i, row in enumerate(window)], dtype=float)
    residuals = np.abs(obs - estimates)
    threshold = float(cfg.get("lit101_temporal_residual_threshold", 2.0))
    alpha = float(cfg.get("lit101_ewma_alpha", 0.35))
    ewma = 0.0
    for value in residuals:
        ewma = alpha * float(value) + (1.0 - alpha) * ewma
    cusum = float(np.maximum(residuals - 0.5 * threshold, 0.0).sum())
    if len(residuals) >= 3:
        x = np.arange(len(residuals), dtype=float)
        slope = float(np.polyfit(x, residuals, 1)[0])
    else:
        slope = 0.0
    high = residuals > threshold
    consecutive = 0
    for value in high[::-1]:
        if value:
            consecutive += 1
        else:
            break

    obs_delta = np.diff(obs)
    est_delta = np.diff(estimates)
    obs_delta_std = float(np.std(obs_delta)) if len(obs_delta) else 0.0
    obs_change = float(abs(obs[-1] - obs[0]))
    expected_change = float(abs(estimates[-1] - estimates[0]))
    rounded_unique_ratio = len(set(np.round(obs, 2))) / max(1, len(obs))
    flat_or_repeated = (
        obs_delta_std < float(cfg.get("lit101_replay_flat_std_threshold", 0.22))
        or rounded_unique_ratio < 0.45
    )
    replay_score = 1.0 if (
        flat_or_repeated
        and expected_change > float(cfg.get("lit101_replay_expected_change_threshold", 1.2))
        and obs_change < 0.6 * expected_change
    ) else 0.0
    # A replayed old window can be non-flat; multi-step mass balance divergence
    # still exposes it once the process leaves the replayed trajectory.
    if residuals[-1] > float(cfg.get("lit101_multistep_threshold", 3.0)) and expected_change > 0.8:
        replay_score = max(replay_score, 0.5)

    return {
        "LIT101_residual_mean": float(residuals.mean()),
        "LIT101_residual_ewma": float(ewma),
        "LIT101_residual_cusum": cusum,
        "LIT101_residual_slope": max(0.0, slope),
        "LIT101_multistep_residual": float(residuals[-1]),
        "LIT101_replay_score": float(replay_score),
        "LIT101_consecutive_high_residual": float(consecutive),
    }


def _mass_balance_series(window: list[dict[str, Any]], config: dict[str, Any]) -> np.ndarray:
    cfg = config or {}
    estimates = [float(get_value(window[0], "level_est", get_value(window[0], "lit101_obs", 0.0)))]
    for idx in range(1, len(window)):
        prev = window[idx - 1]
        estimates.append(
            estimate_level_from_mass_balance(
                estimates[-1],
                float(get_value(prev, "fit101_obs", 0.0)),
                int(get_value(prev, "p101_state", 0)),
                int(get_value(prev, "p102_state", 0)),
                dt=float(cfg.get("dt", 1.0)),
                tank_area=float(cfg.get("tank_area", 1.0)),
                p101_outflow_rate=float(cfg.get("p101_outflow_rate", 0.72)),
                p102_outflow_rate=float(cfg.get("p102_outflow_rate", 0.66)),
                level_min=float(cfg.get("level_min", 0.0)),
                level_max=float(cfg.get("level_max", 100.0)),
            )
        )
    return np.array(estimates, dtype=float)


def compute_rule_violations(
    current_state: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Check PLC and physics consistency rules."""
    cfg = config or {}
    level_obs = float(get_value(current_state, "lit101_obs", 0.0))
    level_est = float(get_value(current_state, "level_est", level_obs))
    fit_obs = float(get_value(current_state, "fit101_obs", 0.0))
    mv_state = int(get_value(current_state, "mv101_state", 0))
    mv_command = int(get_value(current_state, "mv101_command", mv_state))
    p101_state = int(get_value(current_state, "p101_state", 0))
    p101_command = int(get_value(current_state, "p101_command", p101_state))
    p102_state = int(get_value(current_state, "p102_state", 0))
    p102_command = int(get_value(current_state, "p102_command", p102_state))
    fit_high = float(cfg.get("fit101_high_threshold", 0.55))

    violations = {
        "mv_closed_fit_high": int(mv_state == 0 and fit_obs > fit_high),
        "mv_command_mismatch": int(mv_command != mv_state),
        "p101_command_mismatch": int(p101_command != p101_state),
        "p102_command_mismatch": int(p102_command != p102_state),
        "pump_empty_risk": int((p101_state or p102_state) and level_est < 15.0),
        "plc_should_open_mv": int(level_obs < 45.0 and mv_command == 0),
        "plc_should_close_mv": int(level_obs > 60.0 and mv_command == 1),
        "plc_should_stop_pump": int(level_obs < 25.0 and (p101_command or p102_command)),
    }
    return violations


def infer_trust_mask(
    current_state: Any,
    history: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Infer binary trust for P1 sensors, actuators, and PLC1."""
    cfg = config or {}
    history = history or []
    window = history[-int(cfg.get("history_window", 8)) :] + [
        current_state if isinstance(current_state, dict) else current_state.to_dict()
    ]
    residuals = compute_causal_residuals(window, cfg)
    violations = compute_rule_violations(current_state, cfg)
    trust = {key: 1 for key in TRUST_KEYS}

    if residuals["LIT101"] > float(cfg.get("lit101_residual_threshold", 5.0)):
        trust["LIT101"] = 0
    if _temporal_lit101_untrusted(residuals, cfg):
        trust["LIT101"] = 0
    if residuals["FIT101"] > float(cfg.get("fit101_residual_threshold", 0.45)):
        trust["FIT101"] = 0
    if violations["mv_command_mismatch"] or violations["mv_closed_fit_high"]:
        trust["MV101"] = 0
    if violations["p101_command_mismatch"]:
        trust["P101"] = 0
    if violations["p102_command_mismatch"]:
        trust["P102"] = 0
    if _recent_command_mismatch(history, "p101", cfg):
        trust["P101"] = 0
    if _recent_command_mismatch(history, "p102", cfg):
        trust["P102"] = 0
    if _recent_command_mismatch(history, "mv101", cfg) or any(
        int(get_value(row, "mv101_state", 0)) == 1
        and int(get_value(row, "mv101_command", 1)) == 0
        and float(get_value(row, "fit101_obs", 0.0)) > float(cfg.get("fit101_high_threshold", 0.55))
        for row in history[-int(cfg.get("actuator_memory_window", 10)) :]
    ):
        trust["MV101"] = 0
    if (
        violations["plc_should_open_mv"]
        or violations["plc_should_close_mv"]
        or violations["plc_should_stop_pump"]
    ):
        trust["PLC1"] = 0
    return trust


def _recent_command_mismatch(history: list[dict[str, Any]], prefix: str, config: dict[str, Any]) -> bool:
    window = int(config.get("actuator_memory_window", 10))
    for row in history[-window:]:
        command_key = f"{prefix}_command"
        state_key = f"{prefix}_state"
        if command_key in row and state_key in row:
            if int(get_value(row, command_key, 0)) != int(get_value(row, state_key, 0)):
                return True
    return False


def _temporal_lit101_untrusted(residuals: dict[str, float], config: dict[str, Any]) -> bool:
    consecutive_needed = int(config.get("lit101_temporal_consecutive_steps", 3))
    return bool(
        residuals.get("LIT101_consecutive_high_residual", 0.0) >= consecutive_needed
        or residuals.get("LIT101_residual_ewma", 0.0) > float(config.get("lit101_ewma_threshold", 2.2))
        or residuals.get("LIT101_residual_cusum", 0.0) > float(config.get("lit101_cusum_threshold", 8.0))
        or (
            residuals.get("LIT101_residual_slope", 0.0) > float(config.get("lit101_slope_threshold", 0.18))
            and residuals.get("LIT101_multistep_residual", 0.0) > float(config.get("lit101_multistep_threshold", 3.0))
        )
        or residuals.get("LIT101_replay_score", 0.0) >= 1.0
    )


def infer_root_causes(
    current_state: Any,
    trust_mask: dict[str, int],
    residuals: dict[str, float],
    violations: dict[str, int],
) -> list[str]:
    """Map trust and rule evidence to coarse root-cause labels."""
    causes: list[str] = []
    if trust_mask.get("LIT101", 1) == 0:
        if residuals.get("LIT101_replay_score", 0.0) >= 1.0:
            causes.append("LIT101_REPLAY_OR_STALE")
        elif residuals.get("LIT101_residual_slope", 0.0) > 0.0:
            causes.append("LIT101_DRIFT_OR_FDI")
        else:
            causes.append("LIT101_UNTRUSTED")
    mv_command = int(get_value(current_state, "mv101_command", get_value(current_state, "mv101_state", 0)))
    mv_state = int(get_value(current_state, "mv101_state", mv_command))
    if trust_mask.get("MV101", 1) == 0:
        if mv_command == 0 and mv_state == 1:
            causes.append("MV101_STUCK_OPEN")
        elif mv_command == 1 and mv_state == 0:
            causes.append("MV101_STUCK_CLOSED")
        else:
            causes.append("MV101_UNTRUSTED")
    if trust_mask.get("P101", 1) == 0 or violations.get("p101_command_mismatch", 0):
        causes.append("P101_UNTRUSTED_OR_FORCED_OFF")
    if trust_mask.get("P102", 1) == 0:
        causes.append("P102_UNTRUSTED")
    if trust_mask.get("PLC1", 1) == 0:
        causes.append("PLC1_UNTRUSTED")
    return causes


def compute_hazard_priority(
    level_est: float,
    trust_mask: dict[str, int],
    residuals: dict[str, float],
    violations: dict[str, int],
    config: dict[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    """Return hazard priority plus component scores."""
    cfg = config or {}
    safe_min = float(cfg.get("safe_min", 20.0))
    safe_max = float(cfg.get("safe_max", 80.0))
    lit_threshold = float(cfg.get("lit101_residual_threshold", 5.0))
    fit_threshold = float(cfg.get("fit101_residual_threshold", 0.45))

    low_margin = max(0.0, (safe_min - level_est) / max(safe_min, 1.0))
    high_margin = max(0.0, (level_est - safe_max) / max(100.0 - safe_max, 1.0))
    near_low = max(0.0, (safe_min + 8.0 - level_est) / 8.0)
    near_high = max(0.0, (level_est - (safe_max - 8.0)) / 8.0)
    hazard_margin_score = clamp(max(low_margin, high_margin, near_low, near_high), 0.0, 1.5)

    causal_score = clamp(
        0.5 * residuals.get("LIT101", 0.0) / max(lit_threshold, 1e-6)
        + 0.25 * residuals.get("FIT101", 0.0) / max(fit_threshold, 1e-6)
        + 0.25 * max(residuals.get("MV101", 0.0), residuals.get("P101", 0.0)),
        0.0,
        2.0,
    )
    logic_score = clamp(sum(violations.values()) / 4.0, 0.0, 2.0)
    untrusted = sum(1 for value in trust_mask.values() if value == 0)
    attack_score = clamp(untrusted / len(trust_mask) + 0.25 * causal_score, 0.0, 2.0)
    criticality_score = 0.0
    for key, weight in {"LIT101": 0.35, "MV101": 0.35, "P101": 0.2, "FIT101": 0.1}.items():
        criticality_score += weight * (1 - trust_mask.get(key, 1))

    weights = cfg.get("hazard_weights", {})
    priority = (
        float(weights.get("w_causal", cfg.get("w_causal", 0.22))) * causal_score
        + float(weights.get("w_logic", cfg.get("w_logic", 0.20))) * logic_score
        + float(weights.get("w_margin", cfg.get("w_margin", 0.28))) * hazard_margin_score
        + float(weights.get("w_criticality", cfg.get("w_criticality", 0.12))) * criticality_score
        + float(weights.get("w_attack", cfg.get("w_attack", 0.18))) * attack_score
    )
    return float(priority), float(causal_score), float(logic_score), float(attack_score)


def diagnose(
    current_state: Any,
    history: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> CausalDiagnostics:
    """Run residual, rule, trust, and hazard-priority diagnosis."""
    cfg = config or {}
    history = history or []
    state_dict = current_state if isinstance(current_state, dict) else current_state.to_dict()
    mass_level = _mass_level_for_current(state_dict, history, cfg)
    trust_mask = infer_trust_mask(state_dict, history, cfg)
    window = history[-int(cfg.get("history_window", 8)) :] + [state_dict]
    residuals = compute_causal_residuals(window, cfg)
    rule_state = dict(state_dict)
    rule_state["level_est"] = mass_level if trust_mask["LIT101"] == 0 else float(get_value(state_dict, "lit101_obs", mass_level))
    violations = compute_rule_violations(rule_state, cfg)
    level_est = rule_state["level_est"]
    root_causes = infer_root_causes(rule_state, trust_mask, residuals, violations)
    priority, causal_score, logic_score, attack_score = compute_hazard_priority(
        level_est,
        trust_mask,
        residuals,
        violations,
        cfg,
    )
    return CausalDiagnostics(
        trust_mask=trust_mask,
        residuals=residuals,
        rule_violations=violations,
        causal_score=causal_score,
        logic_violation_score=logic_score,
        attack_belief_score=attack_score,
        hazard_priority=priority,
        mass_balance_level=mass_level,
        root_causes=root_causes,
    )


def reconstruct_belief_state(
    current_state: Any,
    history: list[dict[str, Any]] | None,
    diagnostics: CausalDiagnostics,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply trust-aware partial rollback and return the compact belief vector."""
    cfg = config or {}
    state_dict = current_state if isinstance(current_state, dict) else current_state.to_dict()
    trust = diagnostics.trust_mask
    level_est = float(get_value(state_dict, "lit101_obs", diagnostics.mass_balance_level))
    if trust["LIT101"] == 0:
        level_est = diagnostics.mass_balance_level
    fit_est = float(get_value(state_dict, "fit101_obs", 0.0))
    if trust["FIT101"] == 0:
        fit_est = predicted_inflow_from_mv101(int(get_value(state_dict, "mv101_state", 0)), cfg)
    level_est = clamp(level_est, float(cfg.get("level_min", 0.0)), float(cfg.get("level_max", 100.0)))

    belief_dict = {
        "level_est": level_est,
        "fit_est": fit_est,
        "mv101_state": float(get_value(state_dict, "mv101_state", 0)),
        "p101_state": float(get_value(state_dict, "p101_state", 0)),
        "p102_state": float(get_value(state_dict, "p102_state", 0)),
        "trust_LIT101": float(trust["LIT101"]),
        "trust_FIT101": float(trust["FIT101"]),
        "trust_MV101": float(trust["MV101"]),
        "trust_P101": float(trust["P101"]),
        "trust_P102": float(trust["P102"]),
        "hazard_priority": float(diagnostics.hazard_priority),
        "attack_belief_score": float(diagnostics.attack_belief_score),
    }
    vector = np.array([belief_dict[col] for col in BELIEF_COLUMNS], dtype=np.float32)
    return vector, belief_dict


def diagnose_real_swat_timeseries(
    df: Any,
    calibration: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Run offline causal/rule diagnosis on real SWaT P1 logs.

    Real logs do not expose a hidden true level, so this estimates the latent
    expected LIT101 from a calibrated one-step process model and scores
    consistency residuals over time.
    """
    import pandas as pd

    cfg = config or {}
    cal = calibration or {}
    out = df.copy()
    if out.empty:
        return out
    b = cal.get("coefficients", {})
    if cal.get("ok") and b:
        prev = out.shift(1)
        out["lit101_est"] = (
            float(cal.get("intercept", 0.0))
            + float(b.get("beta_lit", 1.0)) * prev["LIT101"].fillna(out["LIT101"])
            + float(b.get("beta_fit", 0.0)) * prev["FIT101"].fillna(out["FIT101"])
            + float(b.get("beta_p101", 0.0)) * prev.get("P101_on_binary", 0).fillna(0)
            + float(b.get("beta_p102", 0.0)) * prev.get("P102_on_binary", 0).fillna(0)
        )
        out["lit101_est"] = out["lit101_est"].fillna(out["LIT101"])
    else:
        out["lit101_est"] = out["LIT101"].shift(1).fillna(out["LIT101"])
    out["residual_LIT101"] = out["LIT101"] - out["lit101_est"]
    out["abs_residual_LIT101"] = out["residual_LIT101"].abs()
    alpha = float(cfg.get("ewma_alpha", cfg.get("lit101_ewma_alpha", 0.1)))
    out["lit101_residual_ewma"] = out["abs_residual_LIT101"].ewm(alpha=alpha, adjust=False).mean()
    window = int(cfg.get("residual_window", cfg.get("history_window", 20)))
    out["lit101_residual_mean"] = out["abs_residual_LIT101"].rolling(window, min_periods=1).mean()
    threshold = float(cfg.get("cusum_threshold", max(5.0, 3.0 * float(cal.get("rmse", 1.0)))))
    out["lit101_residual_cusum"] = (out["abs_residual_LIT101"] - 0.5 * threshold).clip(lower=0).rolling(window, min_periods=1).sum()
    out["lit101_slope"] = out["abs_residual_LIT101"].diff().rolling(window, min_periods=1).mean().clip(lower=0)
    expected_change = out["lit101_est"].diff().abs().rolling(window, min_periods=1).sum()
    observed_change = out["LIT101"].diff().abs().rolling(window, min_periods=1).sum()
    flat = out["LIT101"].diff().abs().rolling(window, min_periods=1).std().fillna(0) < float(cfg.get("replay_flat_std", 0.05))
    out["lit101_replay_score"] = ((flat) & (expected_change > 1.0) & (observed_change < 0.5 * expected_change)).astype(float)
    persistent = int(cfg.get("persistence_steps", 3))
    high_resid = out["abs_residual_LIT101"] > max(float(cal.get("rmse", 1.0)) * 3.0, 1.0)
    out["trust_LIT101"] = 1
    out.loc[
        high_resid.rolling(persistent, min_periods=1).sum() >= persistent,
        "trust_LIT101",
    ] = 0
    out.loc[out["lit101_residual_cusum"] > threshold, "trust_LIT101"] = 0
    out.loc[out["lit101_replay_score"] >= 1.0, "trust_LIT101"] = 0

    out = _add_real_actuator_suspicion(out, cal, cfg)
    actuator_threshold = float(cfg.get("actuator_suspicion_trust_threshold", 0.8))
    out["trust_FIT101"] = 1
    out["trust_MV101"] = 1
    out.loc[out["mv101_suspicion_score"] >= actuator_threshold, ["trust_MV101", "trust_FIT101"]] = 0
    out["trust_P101"] = 1
    out["trust_P102"] = 1
    out.loc[out["p101_suspicion_score"] >= actuator_threshold, "trust_P101"] = 0
    out.loc[out["p102_suspicion_score"] >= actuator_threshold, "trust_P102"] = 0
    out["trust_PLC1"] = 1
    # PLC proxy: normal hysteresis behavior is rough; mark only persistent strong contradictions.
    out.loc[(out["LIT101"] > out["LIT101"].quantile(0.95)) & (out.get("MV101_open_binary", 0) == 1), "trust_PLC1"] = 0
    out.loc[out["plc1_suspicion_score"] >= actuator_threshold, "trust_PLC1"] = 0
    out["causal_score"] = (out["abs_residual_LIT101"] / max(float(cal.get("rmse", 1.0)), 1e-6)).clip(0, 10)
    out["logic_violation_score"] = (
        (1 - out["trust_MV101"]) + (1 - out["trust_FIT101"]) + (1 - out["trust_P101"]) + (1 - out["trust_P102"]) + (1 - out["trust_PLC1"])
    ) / 5.0
    trust_cols = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102", "trust_PLC1"]
    out["attack_belief_score"] = (1 - out[trust_cols]).mean(axis=1).clip(0, 1)
    out["level_est"] = out["LIT101"]
    out.loc[out["trust_LIT101"] == 0, "level_est"] = out.loc[out["trust_LIT101"] == 0, "lit101_est"]
    return out


def _add_real_actuator_suspicion(out: pd.DataFrame, cal: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    out = out.copy()
    persistent = int(cfg.get("actuator_persistence_steps", cfg.get("persistence_steps", 3)))
    window = max(1, int(cfg.get("residual_window", 20)))
    lit = pd.to_numeric(out.get("LIT101"), errors="coerce").ffill().bfill()
    fit = pd.to_numeric(out.get("FIT101"), errors="coerce").ffill().bfill()
    lit_delta = lit.diff().fillna(0.0)
    rmse = max(float(cal.get("rmse", 1.0)), 1e-6)
    lit_range = max(
        1e-6,
        float(cal.get("lit_max_normal", lit.max())) - float(cal.get("lit_min_normal", lit.min())),
    )
    slope_tol = max(float(cfg.get("actuator_slope_tolerance", 0.02 * rmse)), 0.0005 * lit_range)

    open_mean = float(cal.get("mv101_fit_open_mean", fit.quantile(0.75)))
    closed_mean = float(cal.get("mv101_fit_closed_mean", fit.quantile(0.25)))
    separator = closed_mean + 0.5 * max(1e-6, open_mean - closed_mean)
    mv = _numeric_column(out, "MV101_open_binary", default=0).round().clip(0, 1)
    p101 = _numeric_column(out, "P101_on_binary", default=0).round().clip(0, 1)
    p102 = _numeric_column(out, "P102_on_binary", default=0).round().clip(0, 1)

    mv_open_low_fit_bool = (mv == 1) & (fit < separator)
    mv_closed_high_fit_bool = (mv == 0) & (fit > separator)
    mv_open_low_fit = mv_open_low_fit_bool.astype(float)
    mv_closed_high_fit = mv_closed_high_fit_bool.astype(float)
    mv_change = mv.diff().abs().fillna(0)
    mv_run = mv_change.cumsum()
    mv_run_length = mv.groupby(mv_run).cumcount() + 1
    long_constant = (mv_run_length > max(window, persistent * 4)).astype(float)
    fit_mismatch = (mv_open_low_fit_bool | mv_closed_high_fit_bool).astype(float)
    out["mv101_fit_separator"] = separator
    out["mv101_low_fit_when_open"] = mv_open_low_fit
    out["mv101_high_fit_when_closed"] = mv_closed_high_fit
    out["mv101_suspicion_score"] = (
        fit_mismatch.rolling(persistent, min_periods=1).mean()
        + 0.25 * long_constant * fit_mismatch.rolling(window, min_periods=1).max()
    ).clip(0, 1)

    no_pump = (p101 == 0) & (p102 == 0)
    no_pump_median = _finite_median(lit_delta[no_pump], default=float(lit_delta.median()))
    p101_on_median = _finite_median(lit_delta[(p101 == 1) & (p102 == 0)], default=no_pump_median - slope_tol)
    p102_on_median = _finite_median(lit_delta[(p102 == 1) & (p101 == 0)], default=no_pump_median - slope_tol)
    p101_no_drain = ((p101 == 1) & (lit_delta >= min(no_pump_median, p101_on_median) - slope_tol)).astype(float)
    p102_no_drain = ((p102 == 1) & (lit_delta >= min(no_pump_median, p102_on_median) - slope_tol)).astype(float)
    p101_off_drain_like = ((p101 == 0) & (p102 == 0) & (lit_delta < p101_on_median - slope_tol)).astype(float)
    p102_off_drain_like = ((p102 == 0) & (p101 == 0) & (lit_delta < p102_on_median - slope_tol)).astype(float)
    out["lit101_slope_delta"] = lit_delta
    out["p101_expected_slope_on"] = p101_on_median
    out["p102_expected_slope_on"] = p102_on_median
    out["p101_suspicion_score"] = (
        p101_no_drain.rolling(persistent, min_periods=1).mean()
        + 0.5 * p101_off_drain_like.rolling(persistent, min_periods=1).mean()
    ).clip(0, 1)
    out["p102_suspicion_score"] = (
        p102_no_drain.rolling(persistent, min_periods=1).mean()
        + 0.5 * p102_off_drain_like.rolling(persistent, min_periods=1).mean()
    ).clip(0, 1)

    safe_low = float(cal.get("safe_low", lit.quantile(0.05)))
    safe_high = float(cal.get("safe_high", lit.quantile(0.95)))
    target_low = float(cal.get("target_low", lit.quantile(0.35)))
    target_high = float(cal.get("target_high", lit.quantile(0.65)))
    plc_unlikely = (
        ((lit < target_low) & (mv == 0))
        | ((lit > target_high) & (mv == 1))
        | ((lit < safe_low) & ((p101 == 1) | (p102 == 1)))
        | ((lit > safe_high) & ((p101 == 0) & (p102 == 0)))
    ).astype(float)
    out["plc1_suspicion_score"] = plc_unlikely.rolling(persistent, min_periods=1).mean().clip(0, 1)
    lit_scale = max(3.0 * rmse, 1e-6)
    out["lit101_suspicion_score"] = (
        pd.to_numeric(out.get("abs_residual_LIT101"), errors="coerce").fillna(0) / lit_scale
        + pd.to_numeric(out.get("lit101_replay_score"), errors="coerce").fillna(0)
    ).clip(0, 1)

    score_cols = {
        "LIT101_UNTRUSTED": "lit101_suspicion_score",
        "MV101_OR_FIT101_SUSPICIOUS": "mv101_suspicion_score",
        "P101_SUSPICIOUS": "p101_suspicion_score",
        "P102_SUSPICIOUS": "p102_suspicion_score",
        "PLC1_SUSPICIOUS": "plc1_suspicion_score",
    }
    scores = pd.DataFrame({label: pd.to_numeric(out[col], errors="coerce").fillna(0.0) for label, col in score_cols.items()})
    out["root_cause_confidence"] = scores.max(axis=1).clip(0, 1)
    out["inferred_root_cause"] = scores.idxmax(axis=1)
    out.loc[out["root_cause_confidence"] < float(cfg.get("root_cause_min_confidence", 0.35)), "inferred_root_cause"] = "none"
    return out


def _numeric_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _finite_median(series: pd.Series, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float(default)
    return float(values.median())
