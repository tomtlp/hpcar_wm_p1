"""Baseline recovery policies for comparison with the proposed method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .causal_logic import diagnose, reconstruct_belief_state
from .planner import HazardPrioritizedPlanner, PlannerDecision
from .recovery_actions import RecoveryAction
from .safety_shield import SafetyShield, ShieldDecision


METHODS = [
    "B1_FULL_SHUTDOWN",
    "B2_RULE_BASED_FALLBACK",
    "B3_ANOMALY_PRIORITY_RECOVERY",
    "B4_WORLD_MODEL_NO_TRUST",
    "B5_PROPOSED",
]


@dataclass
class BaselineDecision:
    action: RecoveryAction
    requested_action: RecoveryAction
    diagnostics: Any
    belief: dict[str, float]
    shield_intervened: bool
    shield_reason: str = ""


def choose_action(
    method: str,
    current_state: Any,
    history: list[dict[str, Any]],
    proposed_planner: HazardPrioritizedPlanner,
    no_trust_planner: HazardPrioritizedPlanner,
    shield: SafetyShield,
    diagnosis_config: dict[str, Any],
    attack_detected: bool,
) -> BaselineDecision | PlannerDecision:
    """Return a recovery action for one method."""
    if method in {"B5_PROPOSED", "B5_FULL"}:
        return proposed_planner.select_action(current_state, history)
    if method in {"B4_WORLD_MODEL_NO_TRUST", "B5_NO_TRUST"}:
        return no_trust_planner.select_action(current_state, history)
    if method == "B5_NO_SHIELD":
        decision = proposed_planner.select_action(current_state, history)
        return PlannerDecision(
            action=decision.requested_action,
            requested_action=decision.requested_action,
            diagnostics=decision.diagnostics,
            belief=decision.belief,
            action_costs=decision.action_costs,
            shield=ShieldDecision(decision.requested_action, decision.requested_action, False, "disabled_for_ablation"),
        )

    diagnostics = diagnose(current_state, history, diagnosis_config)
    _, belief = reconstruct_belief_state(current_state, history, diagnostics, diagnosis_config)

    if method == "B1_FULL_SHUTDOWN":
        requested = RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN if attack_detected else RecoveryAction.R0_KEEP_CURRENT
    elif method == "B2_RULE_BASED_FALLBACK":
        requested = RecoveryAction.R5_P1_FALLBACK_CONTROL if attack_detected else RecoveryAction.R0_KEEP_CURRENT
    elif method == "B3_ANOMALY_PRIORITY_RECOVERY":
        requested = anomaly_priority_action(diagnostics, attack_detected)
    elif method == "B5_NO_ROOT_CAUSE":
        requested = RecoveryAction.R5_P1_FALLBACK_CONTROL if attack_detected else RecoveryAction.R0_KEEP_CURRENT
    else:
        raise ValueError(f"Unknown method: {method}")

    shield_decision = shield.filter_action(requested, belief, current_state)
    return BaselineDecision(
        action=shield_decision.action,
        requested_action=requested,
        diagnostics=diagnostics,
        belief=belief,
        shield_intervened=shield_decision.intervened,
        shield_reason=shield_decision.reason,
    )


def anomaly_priority_action(diagnostics: Any, attack_detected: bool) -> RecoveryAction:
    """Policy that reacts to the largest residual, ignoring physical hazard priority."""
    if not attack_detected:
        return RecoveryAction.R0_KEEP_CURRENT
    residuals = {key: diagnostics.residuals.get(key, 0.0) for key in ["LIT101", "FIT101", "MV101", "P101", "P102"]}
    largest = max(residuals, key=lambda key: residuals[key])
    if largest == "LIT101":
        return RecoveryAction.R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL
    if largest in {"MV101", "FIT101"}:
        return RecoveryAction.R2_FREEZE_MV101_SAFE
    if largest in {"P101", "P102"}:
        return RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP
    return RecoveryAction.R5_P1_FALLBACK_CONTROL
