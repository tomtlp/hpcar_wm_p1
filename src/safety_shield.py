"""Predictive safety shield for high-level recovery actions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .recovery_actions import ControlCommand, RecoveryAction, apply_recovery_action


@dataclass
class SafetyShieldConfig:
    hard_min: float = 10.0
    hard_max: float = 90.0
    soft_min: float = 20.0
    soft_max: float = 80.0
    pump_empty_level: float = 15.0
    mv_open_high_level: float = 85.0
    emergency_drain_level: float = 78.0
    severe_hazard_level_low: float = 8.0
    severe_hazard_level_high: float = 92.0
    severe_untrusted_count: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SafetyShieldConfig":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class ShieldDecision:
    requested_action: RecoveryAction
    action: RecoveryAction
    intervened: bool
    reason: str = ""


class SafetyShield:
    """Action filter that prevents obvious secondary hazards."""

    def __init__(self, config: SafetyShieldConfig | dict[str, Any] | None = None):
        self.config = config if isinstance(config, SafetyShieldConfig) else SafetyShieldConfig.from_dict(config)
        self.intervention_count = 0

    def reset(self) -> None:
        self.intervention_count = 0

    def filter_action(
        self,
        action: RecoveryAction,
        belief: dict[str, float],
        current_state: Any | None = None,
    ) -> ShieldDecision:
        """Replace unsafe actions with a fallback or local safe shutdown."""
        cfg = self.config
        level_est = float(belief.get("level_est", 50.0))
        candidate_command = self._command_for(action, belief, current_state)
        replacement = action
        reason = ""

        mv_untrusted = float(belief.get("trust_MV101", 1.0)) < 0.5
        p101_trusted = float(belief.get("trust_P101", 1.0)) >= 0.5
        p102_trusted = float(belief.get("trust_P102", 1.0)) >= 0.5

        if level_est <= cfg.severe_hazard_level_low:
            replacement = RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN
            reason = "severe_hazard_margin"
        elif level_est >= cfg.severe_hazard_level_high:
            replacement = (
                RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS
                if p101_trusted or p102_trusted
                else RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN
            )
            reason = "severe_high_level_requires_drain"
        elif mv_untrusted and level_est >= cfg.emergency_drain_level and action != RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
            replacement = (
                RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS
                if p101_trusted or p102_trusted
                else RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN
            )
            reason = "mv101_untrusted_high_level_drain"
        elif self._untrusted_count(belief) >= cfg.severe_untrusted_count:
            replacement = (
                RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS
                if level_est >= cfg.emergency_drain_level and (p101_trusted or p102_trusted)
                else RecoveryAction.R5_P1_FALLBACK_CONTROL
            )
            reason = "severe_trust_failure"
        elif level_est < cfg.pump_empty_level and (
            action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP
            or candidate_command.p101_command
            or candidate_command.p102_command
        ):
            replacement = RecoveryAction.R5_P1_FALLBACK_CONTROL
            reason = "pump_empty_run_prevention"
        elif level_est > cfg.mv_open_high_level and candidate_command.mv101_command:
            replacement = RecoveryAction.R5_P1_FALLBACK_CONTROL
            reason = "overflow_inlet_prevention"

        # Fallback can still command pumps near empty through stale baseline state, so harden it.
        if replacement == RecoveryAction.R5_P1_FALLBACK_CONTROL and level_est < cfg.pump_empty_level:
            safe_command = self._command_for(replacement, belief, current_state)
            if safe_command.p101_command or safe_command.p102_command:
                replacement = RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN
                reason = "fallback_not_safe_for_empty_tank"

        intervened = replacement != action
        if intervened:
            self.intervention_count += 1
        return ShieldDecision(action, replacement, intervened, reason)

    def is_action_safe(
        self,
        action: RecoveryAction,
        belief: dict[str, float],
        current_state: Any | None = None,
    ) -> bool:
        return not self.filter_action(action, belief, current_state).intervened

    def _command_for(
        self,
        action: RecoveryAction,
        belief: dict[str, float],
        current_state: Any | None,
    ) -> ControlCommand:
        state = self._state_obj(current_state, belief)
        baseline = ControlCommand(
            mv101_command=int(getattr(state, "mv101_command", getattr(state, "mv101_state", 1))),
            p101_command=int(getattr(state, "p101_command", getattr(state, "p101_state", 0))),
            p102_command=int(getattr(state, "p102_command", getattr(state, "p102_state", 0))),
            scada_write_enabled=bool(getattr(state, "scada_write_enabled", True)),
            plc_mode=str(getattr(state, "plc_mode", "AUTO")),
        )
        return apply_recovery_action(action, state, baseline, float(belief.get("level_est", 50.0)))

    @staticmethod
    def _state_obj(current_state: Any | None, belief: dict[str, float] | None = None) -> Any:
        belief = belief or {}
        if current_state is None:
            return SimpleNamespace(
                mv101_state=1,
                p101_state=0,
                p102_state=0,
                mv101_command=1,
                p101_command=0,
                p102_command=0,
                scada_write_enabled=True,
                plc_mode="AUTO",
                trust_P101=belief.get("trust_P101", 1.0),
                trust_P102=belief.get("trust_P102", 1.0),
                trust_MV101=belief.get("trust_MV101", 1.0),
            )
        if isinstance(current_state, dict):
            merged = dict(current_state)
            merged.update({k: v for k, v in belief.items() if k.startswith("trust_")})
            return SimpleNamespace(**merged)
        for key, value in belief.items():
            if key.startswith("trust_"):
                setattr(current_state, key, value)
        return current_state

    @staticmethod
    def _untrusted_count(belief: dict[str, float]) -> int:
        keys = ["trust_LIT101", "trust_FIT101", "trust_MV101", "trust_P101", "trust_P102"]
        return sum(1 for key in keys if float(belief.get(key, 1.0)) < 0.5)
