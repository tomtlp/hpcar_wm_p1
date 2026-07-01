"""Attack models for the P1 MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .p1_simulator import P1State


ATTACK_SCENARIOS = [
    "normal",
    "LIT101_FDI",
    "LIT101_DRIFT",
    "LIT101_REPLAY",
    "MV101_STUCK_OPEN",
    "MV101_STUCK_CLOSED",
    "P101_FORCED_OFF",
    "COMBINED_LIT101_FDI_MV101_OPEN",
]


@dataclass
class AttackConfig:
    start_step: int = 25
    end_step: int | None = None
    lit101_fdi_offset: float = 24.0
    lit101_drift_rate: float = 0.35
    replay_window: int = 20
    p101_forced_state: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AttackConfig":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


class P1Attack:
    """Single attack scenario with sensor and actuator hooks."""

    def __init__(self, name: str, config: AttackConfig | dict[str, Any] | None = None):
        if name not in ATTACK_SCENARIOS:
            raise ValueError(f"Unknown attack scenario: {name}")
        self.name = name
        self.config = config if isinstance(config, AttackConfig) else AttackConfig.from_dict(config)
        self.replay_buffer: list[float] = []

    def reset(self) -> None:
        self.replay_buffer.clear()

    def is_active(self, t: int) -> bool:
        if self.name == "normal":
            return False
        if t < self.config.start_step:
            return False
        return self.config.end_step is None or t <= self.config.end_step

    def apply_actuator_attack(
        self,
        t: int,
        mv101_state: int,
        p101_state: int,
        p102_state: int,
    ) -> tuple[int, int, int]:
        if not self.is_active(t):
            return int(mv101_state), int(p101_state), int(p102_state)

        if self.name == "MV101_STUCK_OPEN":
            mv101_state = 1
        elif self.name == "MV101_STUCK_CLOSED":
            mv101_state = 0
        elif self.name == "P101_FORCED_OFF":
            p101_state = int(self.config.p101_forced_state)
        elif self.name == "COMBINED_LIT101_FDI_MV101_OPEN":
            mv101_state = 1
        return int(mv101_state), int(p101_state), int(p102_state)

    def apply_observation_attack(self, state: P1State) -> P1State:
        attacked = state.copy()
        normal_lit = float(attacked.lit101_obs)

        if not self.is_active(attacked.t):
            self._remember_replay_value(normal_lit)
            return attacked

        elapsed = max(0, attacked.t - self.config.start_step)
        if self.name == "LIT101_FDI":
            attacked.lit101_obs = normal_lit + self.config.lit101_fdi_offset
        elif self.name == "LIT101_DRIFT":
            attacked.lit101_obs = normal_lit + self.config.lit101_drift_rate * elapsed
        elif self.name == "LIT101_REPLAY":
            attacked.lit101_obs = self._replay_value(elapsed, normal_lit)
        elif self.name == "COMBINED_LIT101_FDI_MV101_OPEN":
            attacked.lit101_obs = normal_lit + self.config.lit101_fdi_offset
        return attacked

    def ground_truth_trust(self, t: int) -> dict[str, int]:
        trust = {"LIT101": 1, "FIT101": 1, "MV101": 1, "P101": 1, "P102": 1, "PLC1": 1}
        if not self.is_active(t):
            return trust
        if self.name in {"LIT101_FDI", "LIT101_DRIFT", "LIT101_REPLAY", "COMBINED_LIT101_FDI_MV101_OPEN"}:
            trust["LIT101"] = 0
        if self.name in {"MV101_STUCK_OPEN", "MV101_STUCK_CLOSED", "COMBINED_LIT101_FDI_MV101_OPEN"}:
            trust["MV101"] = 0
        if self.name == "P101_FORCED_OFF":
            trust["P101"] = 0
        return trust

    def _remember_replay_value(self, value: float) -> None:
        self.replay_buffer.append(float(value))
        max_len = max(1, int(self.config.replay_window))
        if len(self.replay_buffer) > max_len:
            self.replay_buffer = self.replay_buffer[-max_len:]

    def _replay_value(self, elapsed: int, fallback: float) -> float:
        if not self.replay_buffer:
            return fallback
        return float(self.replay_buffer[elapsed % len(self.replay_buffer)])


def create_attack(name: str, config: AttackConfig | dict[str, Any] | None = None) -> P1Attack:
    return P1Attack(name=name, config=config)
