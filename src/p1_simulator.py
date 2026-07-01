"""Simplified SWaT P1 tank simulator.

The simulator deliberately separates true physical state from observations.
Attacks can corrupt LIT101_obs while level_true remains governed by physics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .recovery_actions import (
    ControlCommand,
    RecoveryAction,
    apply_recovery_action,
)
from .utils import clamp


@dataclass
class P1SimConfig:
    dt: float = 1.0
    tank_area: float = 1.0
    level_min: float = 0.0
    level_max: float = 100.0
    initial_level: float = 52.0
    inflow_rate_open: float = 1.2
    inflow_rate_closed: float = 0.02
    p101_outflow_rate: float = 0.72
    p102_outflow_rate: float = 0.66
    process_noise_std: float = 0.035
    sensor_noise_std: float = 0.12
    safe_min: float = 20.0
    safe_max: float = 80.0
    hard_min: float = 10.0
    hard_max: float = 90.0
    target_min: float = 45.0
    target_max: float = 60.0
    pump_empty_level: float = 15.0
    baseline_low_open: float = 45.0
    baseline_high_close: float = 60.0
    pump_on_level: float = 35.0
    pump_off_level: float = 25.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "P1SimConfig":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class P1State:
    t: int
    level_true: float
    fit_true: float
    mv101_state: int
    p101_state: int
    p102_state: int
    lit101_obs: float
    fit101_obs: float
    scada_write_enabled: bool
    plc_mode: str
    mv101_command: int
    p101_command: int
    p102_command: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "P1State":
        return P1State(**self.to_dict())


class P1Simulator:
    """Stateful simulator for the P1 tank, inlet valve, and pump pair."""

    def __init__(self, config: P1SimConfig | dict[str, Any] | None = None, seed: int = 0):
        self.config = config if isinstance(config, P1SimConfig) else P1SimConfig.from_dict(config)
        self.rng = np.random.default_rng(seed)
        self.state = self.reset(seed=seed)

    def reset(self, seed: int | None = None, initial_level: float | None = None) -> P1State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        level = float(self.config.initial_level if initial_level is None else initial_level)
        mv = 1
        p101 = 1 if level > self.config.pump_on_level else 0
        p102 = 0
        fit = self._inflow(mv)
        lit_obs = self._sensor(level)
        fit_obs = self._sensor(fit)
        self.state = P1State(
            t=0,
            level_true=level,
            fit_true=fit,
            mv101_state=mv,
            p101_state=p101,
            p102_state=p102,
            lit101_obs=lit_obs,
            fit101_obs=fit_obs,
            scada_write_enabled=True,
            plc_mode="AUTO",
            mv101_command=mv,
            p101_command=p101,
            p102_command=p102,
        )
        return self.state.copy()

    def baseline_control(self, level_obs: float) -> ControlCommand:
        """PLC baseline logic with simple hysteresis."""
        mv = int(self.state.mv101_state)
        p101 = int(self.state.p101_state)
        if level_obs < self.config.baseline_low_open:
            mv = 1
        elif level_obs > self.config.baseline_high_close:
            mv = 0

        if level_obs > self.config.pump_on_level:
            p101 = 1
        elif level_obs < self.config.pump_off_level:
            p101 = 0

        return ControlCommand(
            mv101_command=mv,
            p101_command=p101,
            p102_command=0,
            scada_write_enabled=self.state.scada_write_enabled,
            plc_mode="AUTO",
        )

    def step(
        self,
        action: RecoveryAction = RecoveryAction.R0_KEEP_CURRENT,
        attack: Any | None = None,
        level_est: float | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> tuple[P1State, dict[str, Any]]:
        """Advance one closed-loop step under a recovery action and optional attack."""
        level_for_action = float(self.state.lit101_obs if level_est is None else level_est)
        baseline = self.baseline_control(self.state.lit101_obs)
        action_state: Any = self.state
        if recovery_context:
            action_state = self.state.to_dict()
            action_state.update(recovery_context)
        command = apply_recovery_action(action, action_state, baseline, level_for_action)

        actual_mv = int(command.mv101_command)
        actual_p101 = int(command.p101_command)
        actual_p102 = int(command.p102_command)
        if attack is not None:
            actual_mv, actual_p101, actual_p102 = attack.apply_actuator_attack(
                self.state.t,
                actual_mv,
                actual_p101,
                actual_p102,
            )

        next_state, info = self._advance(
            actual_mv=actual_mv,
            actual_p101=actual_p101,
            actual_p102=actual_p102,
            command=command,
        )
        if attack is not None:
            next_state = attack.apply_observation_attack(next_state)
        self.state = next_state
        return next_state.copy(), info

    def step_physics_only(self, mv101_state: int, p101_state: int, p102_state: int) -> P1State:
        """Testing helper for direct physics checks without PLC or attacks."""
        command = ControlCommand(mv101_state, p101_state, p102_state)
        next_state, _ = self._advance(mv101_state, p101_state, p102_state, command)
        self.state = next_state
        return next_state.copy()

    def _advance(
        self,
        actual_mv: int,
        actual_p101: int,
        actual_p102: int,
        command: ControlCommand,
    ) -> tuple[P1State, dict[str, Any]]:
        cfg = self.config
        inflow = self._inflow(actual_mv)
        outflow = self._outflow(actual_p101, actual_p102)
        noise = float(self.rng.normal(0.0, cfg.process_noise_std))
        level_next = self.state.level_true + cfg.dt * (inflow - outflow) / cfg.tank_area + noise
        level_next = clamp(level_next, cfg.level_min, cfg.level_max)
        fit_true = inflow
        lit_obs = self._sensor(level_next)
        fit_obs = self._sensor(fit_true)

        pump_empty = int((actual_p101 or actual_p102) and level_next < cfg.pump_empty_level)
        hard_violation = int(level_next < cfg.hard_min or level_next > cfg.hard_max)
        soft_violation = int(level_next < cfg.safe_min or level_next > cfg.safe_max)
        production = outflow if not soft_violation else 0.0

        next_state = P1State(
            t=self.state.t + 1,
            level_true=level_next,
            fit_true=fit_true,
            mv101_state=int(actual_mv),
            p101_state=int(actual_p101),
            p102_state=int(actual_p102),
            lit101_obs=lit_obs,
            fit101_obs=fit_obs,
            scada_write_enabled=bool(command.scada_write_enabled),
            plc_mode=command.plc_mode,
            mv101_command=int(command.mv101_command),
            p101_command=int(command.p101_command),
            p102_command=int(command.p102_command),
        )
        info = {
            "inflow": inflow,
            "outflow": outflow,
            "level_delta": level_next - self.state.level_true,
            "production": production,
            "pump_empty_run": pump_empty,
            "hard_safety_violation": hard_violation,
            "soft_safety_violation": soft_violation,
            "unrecoverable_by_control": int(command.plc_mode == "UNRECOVERABLE_BY_CONTROL"),
        }
        return next_state, info

    def _inflow(self, mv101_state: int) -> float:
        return self.config.inflow_rate_open if int(mv101_state) else self.config.inflow_rate_closed

    def _outflow(self, p101_state: int, p102_state: int) -> float:
        return (
            int(p101_state) * self.config.p101_outflow_rate
            + int(p102_state) * self.config.p102_outflow_rate
        )

    def _sensor(self, value: float) -> float:
        return float(value + self.rng.normal(0.0, self.config.sensor_noise_std))
