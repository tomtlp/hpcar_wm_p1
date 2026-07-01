"""High-level recovery actions for SWaT P1.

The planner chooses these actions rather than raw arbitrary actuator writes.
Each action is translated into conservative valve and pump commands here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RecoveryAction(str, Enum):
    R0_KEEP_CURRENT = "R0_KEEP_CURRENT"
    R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL = "R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL"
    R2_FREEZE_MV101_SAFE = "R2_FREEZE_MV101_SAFE"
    R3_SWITCH_TO_BACKUP_PUMP = "R3_SWITCH_TO_BACKUP_PUMP"
    R4_LIMIT_PUMP_SWITCHING = "R4_LIMIT_PUMP_SWITCHING"
    R5_P1_FALLBACK_CONTROL = "R5_P1_FALLBACK_CONTROL"
    R6_BLOCK_SCADA_REMOTE_WRITE = "R6_BLOCK_SCADA_REMOTE_WRITE"
    R7_LOCAL_SAFE_SHUTDOWN = "R7_LOCAL_SAFE_SHUTDOWN"
    R8_GRADUAL_RERAMP = "R8_GRADUAL_RERAMP"
    R9_EMERGENCY_DRAIN_BOTH_PUMPS = "R9_EMERGENCY_DRAIN_BOTH_PUMPS"
    R10_SENSOR_ISOLATION_AND_FALLBACK = "R10_SENSOR_ISOLATION_AND_FALLBACK"


@dataclass
class ControlCommand:
    mv101_command: int
    p101_command: int
    p102_command: int
    scada_write_enabled: bool = True
    plc_mode: str = "AUTO"

    def copy(self) -> "ControlCommand":
        return ControlCommand(
            mv101_command=int(self.mv101_command),
            p101_command=int(self.p101_command),
            p102_command=int(self.p102_command),
            scada_write_enabled=bool(self.scada_write_enabled),
            plc_mode=str(self.plc_mode),
        )


ACTION_COSTS: dict[RecoveryAction, float] = {
    RecoveryAction.R0_KEEP_CURRENT: 0.0,
    RecoveryAction.R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL: 0.8,
    RecoveryAction.R2_FREEZE_MV101_SAFE: 0.7,
    RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP: 1.3,
    RecoveryAction.R4_LIMIT_PUMP_SWITCHING: 0.4,
    RecoveryAction.R5_P1_FALLBACK_CONTROL: 1.0,
    RecoveryAction.R6_BLOCK_SCADA_REMOTE_WRITE: 0.5,
    RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN: 2.5,
    RecoveryAction.R8_GRADUAL_RERAMP: 0.9,
    RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS: 2.0,
    RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK: 0.9,
}


def all_recovery_actions() -> list[RecoveryAction]:
    return list(RecoveryAction)


def action_one_hot(action: RecoveryAction) -> list[float]:
    actions = all_recovery_actions()
    return [1.0 if action == item else 0.0 for item in actions]


def action_cost(action: RecoveryAction) -> float:
    return ACTION_COSTS[action]


def fallback_control(
    level_est: float,
    current: ControlCommand | None = None,
    low_open: float = 40.0,
    high_close: float = 60.0,
    pump_off: float = 25.0,
    pump_on: float = 35.0,
) -> ControlCommand:
    """Local level-based fallback rules using reconstructed level."""
    if current is None:
        current = ControlCommand(1, 1, 0, True, "FALLBACK")
    command = current.copy()
    command.plc_mode = "FALLBACK"

    if level_est < low_open:
        command.mv101_command = 1
    elif level_est > high_close:
        command.mv101_command = 0

    if level_est < pump_off:
        command.p101_command = 0
        command.p102_command = 0
    elif level_est > pump_on:
        command.p101_command = 1
        command.p102_command = 0
    return command


def apply_recovery_action(
    action: RecoveryAction,
    current_state: Any,
    baseline_command: ControlCommand,
    level_est: float,
) -> ControlCommand:
    """Translate a high-level recovery action into actuator commands."""
    command = baseline_command.copy()
    trust_p101 = float(_state_value(current_state, "trust_P101", 1.0))
    trust_p102 = float(_state_value(current_state, "trust_P102", 1.0))
    low_open = float(_state_value(current_state, "fallback_low_open", _state_value(current_state, "target_low", 40.0)))
    high_close = float(_state_value(current_state, "fallback_high_close", _state_value(current_state, "target_high", 60.0)))
    pump_off = float(_state_value(current_state, "fallback_pump_off", _state_value(current_state, "safe_low", 25.0)))
    pump_on = float(_state_value(current_state, "fallback_pump_on", _state_value(current_state, "target_high", 35.0)))
    pump_min_level = float(_state_value(current_state, "pump_empty_level", _state_value(current_state, "safe_low", 20.0)))
    emergency_drain_level = float(_state_value(current_state, "emergency_drain_level", high_close + max(1.0, 0.25 * abs(high_close - low_open))))

    if action == RecoveryAction.R0_KEEP_CURRENT:
        return command

    if action in {
        RecoveryAction.R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL,
        RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK,
    }:
        return fallback_control(level_est, command, low_open, high_close, pump_off, pump_on)

    if action == RecoveryAction.R2_FREEZE_MV101_SAFE:
        if level_est > 60.0:
            command.mv101_command = 0
        elif level_est < 40.0:
            command.mv101_command = 1
        command.plc_mode = "RECOVERY_FREEZE_MV101"
        return command

    if action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP:
        # In a pump-root-cause case, preserve safe production through P102.
        # When the level is already high, do not unnecessarily disable a healthy
        # P101: emergency drain can use both pumps.
        if level_est > emergency_drain_level and trust_p101 >= 0.5:
            command.p101_command = 1
            command.p102_command = 1 if trust_p102 >= 0.5 else 0
        else:
            command.p101_command = 0
            command.p102_command = 1 if level_est >= pump_min_level and trust_p102 >= 0.5 else 0
        command.plc_mode = "BACKUP_PUMP"
        return command

    if action == RecoveryAction.R4_LIMIT_PUMP_SWITCHING:
        command.p101_command = int(getattr(current_state, "p101_state", command.p101_command))
        command.p102_command = int(getattr(current_state, "p102_state", command.p102_command))
        command.plc_mode = "ANTI_CHATTER"
        return command

    if action == RecoveryAction.R5_P1_FALLBACK_CONTROL:
        return fallback_control(level_est, command, low_open, high_close, pump_off, pump_on)

    if action == RecoveryAction.R6_BLOCK_SCADA_REMOTE_WRITE:
        command.scada_write_enabled = False
        command.plc_mode = "REMOTE_WRITE_BLOCKED"
        return command

    if action == RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN:
        return ControlCommand(
            mv101_command=0,
            p101_command=0,
            p102_command=0,
            scada_write_enabled=False,
            plc_mode="LOCAL_SAFE_SHUTDOWN",
        )

    if action == RecoveryAction.R8_GRADUAL_RERAMP:
        command = fallback_control(level_est, command, low_open, high_close, pump_off, pump_on)
        command.plc_mode = "GRADUAL_RERAMP"
        if low_open <= level_est <= high_close:
            command.mv101_command = int(getattr(current_state, "mv101_state", command.mv101_command))
            command.p101_command = int(getattr(current_state, "p101_state", command.p101_command))
            command.p102_command = 0
        return command

    if action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
        command.mv101_command = 0
        command.p101_command = 1 if trust_p101 >= 0.5 and level_est >= pump_min_level else 0
        command.p102_command = 1 if trust_p102 >= 0.5 and level_est >= pump_min_level else 0
        command.scada_write_enabled = False
        command.plc_mode = "EMERGENCY_DRAIN"
        if command.p101_command == 0 and command.p102_command == 0:
            command.plc_mode = "UNRECOVERABLE_BY_CONTROL"
        return command

    raise ValueError(f"Unknown recovery action: {action}")


def _state_value(current_state: Any, key: str, default: Any) -> Any:
    if isinstance(current_state, dict):
        return current_state.get(key, default)
    return getattr(current_state, key, default)
