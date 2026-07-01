"""Hazard-prioritized model rollout planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .causal_logic import CausalDiagnostics, diagnose, reconstruct_belief_state
from .recovery_actions import RecoveryAction, action_cost, all_recovery_actions
from .safety_shield import SafetyShield, ShieldDecision
from .world_model import ActionConditionedWorldModel, belief_dict, next_belief_from_prediction


@dataclass
class PlannerConfig:
    horizon: int = 5
    lambda_hazard: float = 9.0
    lambda_safety: float = 18.0
    lambda_production: float = 1.6
    lambda_action: float = 0.35
    lambda_uncertainty: float = 2.0
    lambda_recovery: float = 2.8
    target_mid: float = 52.5
    attack_belief_threshold: float = 0.45
    min_persistent_steps: int = 3
    recovery_exit_steps: int = 5
    warmup_steps: int = 8
    severe_hazard_margin: float = 0.9

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlannerConfig":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class PlannerDecision:
    action: RecoveryAction
    requested_action: RecoveryAction
    diagnostics: CausalDiagnostics
    belief: dict[str, float]
    action_costs: dict[str, float]
    shield: ShieldDecision


class HazardPrioritizedPlanner:
    """Rollout planner over high-level recovery actions."""

    def __init__(
        self,
        world_model: ActionConditionedWorldModel,
        shield: SafetyShield | None = None,
        planner_config: PlannerConfig | dict[str, Any] | None = None,
        diagnosis_config: dict[str, Any] | None = None,
        use_trust: bool = True,
    ):
        self.world_model = world_model
        self.shield = shield or SafetyShield()
        self.config = planner_config if isinstance(planner_config, PlannerConfig) else PlannerConfig.from_dict(planner_config)
        self.diagnosis_config = diagnosis_config or {}
        self.use_trust = use_trust
        self.suspicion_count = 0
        self.clear_count = 0
        self.recovery_active = False

    def select_action(
        self,
        current_state: Any,
        history: list[dict[str, Any]] | None = None,
    ) -> PlannerDecision:
        history = history or []
        diagnostics = diagnose(current_state, history, self.diagnosis_config)
        belief_vector, belief = reconstruct_belief_state(current_state, history, diagnostics, self.diagnosis_config)
        if not self.use_trust:
            belief = self._observed_belief(current_state, diagnostics)
            belief_vector = np.array(list(belief.values()), dtype=np.float32)

        if self.use_trust and not self._should_enter_or_stay_recovery(current_state, belief, diagnostics):
            shield_decision = self.shield.filter_action(RecoveryAction.R0_KEEP_CURRENT, belief, current_state)
            return PlannerDecision(
                action=shield_decision.action,
                requested_action=RecoveryAction.R0_KEEP_CURRENT,
                diagnostics=diagnostics,
                belief=belief,
                action_costs={action.value: (0.0 if action == RecoveryAction.R0_KEEP_CURRENT else 1e6) for action in all_recovery_actions()},
                shield=shield_decision,
            )

        costs = {
            action.value: self._rollout_cost(belief_vector, action)
            + (self._root_cause_adjustment(action, belief, diagnostics) if self.use_trust else 0.0)
            for action in all_recovery_actions()
        }
        requested = min(all_recovery_actions(), key=lambda action: costs[action.value])
        shield_decision = self.shield.filter_action(requested, belief, current_state)
        return PlannerDecision(
            action=shield_decision.action,
            requested_action=requested,
            diagnostics=diagnostics,
            belief=belief,
            action_costs=costs,
            shield=shield_decision,
        )

    def _should_enter_or_stay_recovery(
        self,
        current_state: Any,
        belief: dict[str, float],
        diagnostics: CausalDiagnostics,
    ) -> bool:
        t = int(current_state.get("t", 0) if isinstance(current_state, dict) else getattr(current_state, "t", 0))
        level = float(belief.get("level_est", 50.0))
        severe_hazard = level < 20.0 or level > 80.0
        suspicious = (
            diagnostics.attack_belief_score >= self.config.attack_belief_threshold
            or any(value == 0 for value in diagnostics.trust_mask.values())
            or diagnostics.logic_violation_score > 0.25
            or diagnostics.hazard_priority >= self.config.severe_hazard_margin
        )
        if t < self.config.warmup_steps and not severe_hazard:
            suspicious = False

        if suspicious or severe_hazard:
            self.suspicion_count += 1
            self.clear_count = 0
        else:
            self.suspicion_count = 0
            self.clear_count += 1

        if severe_hazard or self.suspicion_count >= self.config.min_persistent_steps:
            self.recovery_active = True
        elif self.recovery_active and self.clear_count >= self.config.recovery_exit_steps:
            self.recovery_active = False
        return self.recovery_active

    def _root_cause_adjustment(
        self,
        action: RecoveryAction,
        belief: dict[str, float],
        diagnostics: CausalDiagnostics,
    ) -> float:
        """Negative values prioritize root-cause-appropriate actions."""
        causes = set(getattr(diagnostics, "root_causes", []))
        level = float(belief.get("level_est", 50.0))
        trust_mv = float(belief.get("trust_MV101", 1.0))
        trust_p101 = float(belief.get("trust_P101", 1.0))
        trust_p102 = float(belief.get("trust_P102", 1.0))
        adjustment = 0.0

        if action == RecoveryAction.R0_KEEP_CURRENT and causes:
            adjustment += 6.0
        if any(cause.startswith("LIT101") for cause in causes):
            if action in {
                RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK,
                RecoveryAction.R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL,
            }:
                adjustment -= 5.0
            elif action == RecoveryAction.R5_P1_FALLBACK_CONTROL:
                adjustment -= 2.0
        if "MV101_STUCK_OPEN" in causes or (trust_mv < 0.5 and level > 70.0):
            if action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
                adjustment -= 9.0
            if action in {RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN, RecoveryAction.R2_FREEZE_MV101_SAFE}:
                adjustment += 4.0
        if "MV101_STUCK_CLOSED" in causes:
            if action in {RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R6_BLOCK_SCADA_REMOTE_WRITE}:
                adjustment -= 2.0
            if action == RecoveryAction.R2_FREEZE_MV101_SAFE:
                adjustment += 3.0
        if "P101_UNTRUSTED_OR_FORCED_OFF" in causes or trust_p101 < 0.5:
            if action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP:
                adjustment -= 7.0
            if action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS and level > 70.0 and trust_p102 >= 0.5:
                adjustment -= 5.0
        if "PLC1_UNTRUSTED" in causes:
            if action in {RecoveryAction.R5_P1_FALLBACK_CONTROL, RecoveryAction.R6_BLOCK_SCADA_REMOTE_WRITE}:
                adjustment -= 3.0

        if trust_mv < 0.5 and action == RecoveryAction.R2_FREEZE_MV101_SAFE:
            adjustment += 4.0
        if trust_p101 < 0.5 and action not in {
            RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP,
            RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS,
            RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN,
        }:
            adjustment += 1.5
        return float(adjustment)

    def _rollout_cost(self, belief: np.ndarray | dict[str, float], action: RecoveryAction) -> float:
        cfg = self.config
        b = belief_dict(belief)
        total = cfg.lambda_action * action_cost(action)
        for step in range(max(1, int(cfg.horizon))):
            pred = self.world_model.predict(b, action)
            level = pred.level_est_next
            hazard_cost = self._hazard_margin_cost(level) + pred.safety_risk_next
            safety_cost = 1.0 if level < 10.0 or level > 90.0 else 0.0
            production_loss_cost = max(0.0, 0.72 - pred.production_next)
            uncertainty_cost = self._uncertainty_cost(b)
            recovery_cost = abs(level - cfg.target_mid) / 50.0
            discount = 1.0 + 0.08 * step
            total += discount * (
                cfg.lambda_hazard * hazard_cost
                + cfg.lambda_safety * safety_cost
                + cfg.lambda_production * production_loss_cost
                + cfg.lambda_uncertainty * uncertainty_cost
                + cfg.lambda_recovery * recovery_cost
            )
            b = next_belief_from_prediction(b, pred)
        return float(total)

    @staticmethod
    def _hazard_margin_cost(level: float) -> float:
        if level < 20.0:
            return (20.0 - level) / 20.0
        if level > 80.0:
            return (level - 80.0) / 20.0
        if level < 30.0:
            return (30.0 - level) / 60.0
        if level > 70.0:
            return (level - 70.0) / 60.0
        return 0.0

    @staticmethod
    def _uncertainty_cost(belief: dict[str, float]) -> float:
        keys = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102"]
        return sum(1.0 - float(belief.get(key, 1.0)) for key in keys) / len(keys)

    def _observed_belief(self, current_state: Any, diagnostics: CausalDiagnostics) -> dict[str, float]:
        if isinstance(current_state, dict):
            state = current_state
        else:
            state = current_state.to_dict()
        level = float(state.get("lit101_obs", diagnostics.mass_balance_level))
        fit = float(state.get("fit101_obs", 0.0))
        priority = diagnostics.hazard_priority
        return {
            "level_est": level,
            "fit_est": fit,
            "mv101_state": float(state.get("mv101_state", 0)),
            "p101_state": float(state.get("p101_state", 0)),
            "p102_state": float(state.get("p102_state", 0)),
            "trust_LIT101": 1.0,
            "trust_FIT101": 1.0,
            "trust_MV101": 1.0,
            "trust_P101": 1.0,
            "trust_P102": 1.0,
            "hazard_priority": float(priority),
            "attack_belief_score": 0.0,
        }
