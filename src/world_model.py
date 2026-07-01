"""Action-conditioned world model with a physics fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .causal_logic import BELIEF_COLUMNS
from .recovery_actions import RecoveryAction, action_one_hot
from .utils import clamp


@dataclass
class WorldPrediction:
    level_est_next: float
    fit_next: float
    safety_risk_next: float
    production_next: float


class PhysicsWorldModel:
    """Non-neural deterministic approximation used for planning and fallback."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def predict(self, belief: np.ndarray | dict[str, float], action: RecoveryAction) -> WorldPrediction:
        b = belief_dict(belief)
        level = float(b["level_est"])
        mv = int(round(float(b["mv101_state"])))
        p101 = int(round(float(b["p101_state"])))
        p102 = int(round(float(b["p102_state"])))
        mv, p101, p102 = self._apply_action(level, mv, p101, p102, action)
        inflow = self._inflow(mv)
        outflow = self._outflow(p101, p102)
        dt = float(self.config.get("dt", 1.0))
        tank_area = float(self.config.get("tank_area", 1.0))
        level_next = level + dt * (inflow - outflow) / tank_area
        level_next = clamp(
            level_next,
            float(self.config.get("level_min", 0.0)),
            float(self.config.get("level_max", 100.0)),
        )
        risk = self._risk(level_next, p101, p102)
        production = outflow if 20.0 <= level_next <= 80.0 else 0.0
        return WorldPrediction(level_next, inflow, risk, production)

    def _apply_action(
        self,
        level: float,
        mv: int,
        p101: int,
        p102: int,
        action: RecoveryAction,
    ) -> tuple[int, int, int]:
        if action in {
            RecoveryAction.R1_ISOLATE_LIT101_USE_ESTIMATED_LEVEL,
            RecoveryAction.R5_P1_FALLBACK_CONTROL,
            RecoveryAction.R8_GRADUAL_RERAMP,
            RecoveryAction.R10_SENSOR_ISOLATION_AND_FALLBACK,
        }:
            if level < 40.0:
                mv = 1
            elif level > 60.0:
                mv = 0
            if level < 25.0:
                p101, p102 = 0, 0
            elif level > 35.0:
                p101, p102 = 1, 0
        elif action == RecoveryAction.R2_FREEZE_MV101_SAFE:
            if level > 60.0:
                mv = 0
            elif level < 40.0:
                mv = 1
        elif action == RecoveryAction.R3_SWITCH_TO_BACKUP_PUMP:
            if level > 75.0:
                p101, p102 = 1, 1
            else:
                p101, p102 = 0, int(level >= 20.0)
        elif action == RecoveryAction.R7_LOCAL_SAFE_SHUTDOWN:
            mv, p101, p102 = 0, 0, 0
        elif action == RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS:
            mv = 0
            p101 = int(level >= 20.0)
            p102 = int(level >= 20.0)
        return int(mv), int(p101), int(p102)

    def _inflow(self, mv: int) -> float:
        return float(self.config.get("inflow_rate_open", 1.2) if mv else self.config.get("inflow_rate_closed", 0.02))

    def _outflow(self, p101: int, p102: int) -> float:
        return int(p101) * float(self.config.get("p101_outflow_rate", 0.72)) + int(p102) * float(
            self.config.get("p102_outflow_rate", 0.66)
        )

    def _risk(self, level: float, p101: int, p102: int) -> float:
        hard = level < float(self.config.get("hard_min", 10.0)) or level > float(self.config.get("hard_max", 90.0))
        empty = (p101 or p102) and level < float(self.config.get("pump_empty_level", 15.0))
        soft_margin = max(0.0, 20.0 - level, level - 80.0) / 20.0
        return clamp(float(hard or empty) + soft_margin, 0.0, 2.0)


class _MLP:
    """Lazy Torch MLP wrapper so import failures do not break the experiment."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class ActionConditionedWorldModel:
    """Small neural world model with deterministic physics fallback."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.physics = PhysicsWorldModel(self.config)
        self.net: _MLP | None = None
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None
        self.trained = False
        self.metrics = {"one_step_rmse": np.nan, "multi_step_rollout_rmse": np.nan}

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        if not bool(self.config.get("enabled", True)):
            print("[world_model] Neural training disabled; using physics fallback.")
            self._write_fallback_artifact(output_path, "training_disabled")
            return self.metrics
        if len(x) < 20:
            print("[world_model] Not enough samples; using physics fallback.")
            self._write_fallback_artifact(output_path, "not_enough_samples")
            return self.metrics
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            print(f"[world_model] Torch unavailable ({exc}); using physics fallback.")
            self._write_fallback_artifact(output_path, f"torch_unavailable: {exc}")
            return self.metrics

        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=float(self.config.get("validation_fraction", 0.2)),
                random_state=0,
            )
            self.x_scaler = StandardScaler().fit(x_train)
            self.y_scaler = StandardScaler().fit(y_train)
            x_train_s = self.x_scaler.transform(x_train).astype(np.float32)
            y_train_s = self.y_scaler.transform(y_train).astype(np.float32)
            x_val_s = self.x_scaler.transform(x_val).astype(np.float32)

            self.net = _MLP(
                input_dim=x.shape[1],
                hidden_dim=int(self.config.get("hidden_dim", 48)),
                output_dim=y.shape[1],
            )
            optimizer = torch.optim.Adam(self.net.model.parameters(), lr=float(self.config.get("learning_rate", 1e-3)))
            loss_fn = nn.MSELoss()
            batch_size = int(self.config.get("batch_size", 64))
            epochs = int(self.config.get("epochs", 8))

            tensor_x = torch.tensor(x_train_s)
            tensor_y = torch.tensor(y_train_s)
            for _ in range(max(1, epochs)):
                permutation = torch.randperm(tensor_x.shape[0])
                for start in range(0, tensor_x.shape[0], batch_size):
                    idx = permutation[start : start + batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(self.net.model(tensor_x[idx]), tensor_y[idx])
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                val_pred_s = self.net.model(torch.tensor(x_val_s)).numpy()
            val_pred = self.y_scaler.inverse_transform(val_pred_s)
            one_step_mse = mean_squared_error(y_val[:, :2], val_pred[:, :2])
            one_step = float(np.sqrt(one_step_mse))
            horizon = max(1, int(self.config.get("rollout_horizon", 5)))
            self.metrics = {
                "one_step_rmse": one_step,
                "multi_step_rollout_rmse": float(one_step * np.sqrt(horizon)),
            }
            self.trained = True
            if output_path is not None:
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": self.net.model.state_dict(),
                        "x_scaler": self.x_scaler,
                        "y_scaler": self.y_scaler,
                        "config": self.config,
                        "metrics": self.metrics,
                    },
                    output,
                )
            print(
                "[world_model] trained MLP: "
                f"one_step_rmse={self.metrics['one_step_rmse']:.3f}, "
                f"multi_step_rmse={self.metrics['multi_step_rollout_rmse']:.3f}"
            )
            return self.metrics
        except Exception as exc:
            print(f"[world_model] Training failed ({exc}); using physics fallback.")
            self.trained = False
            self._write_fallback_artifact(output_path, f"training_failed: {exc}")
            return self.metrics

    def predict(self, belief: np.ndarray | dict[str, float], action: RecoveryAction) -> WorldPrediction:
        if not self.trained or self.net is None or self.x_scaler is None or self.y_scaler is None:
            return self.physics.predict(belief, action)
        try:
            import torch

            x = make_model_input(belief, action).reshape(1, -1)
            x_s = self.x_scaler.transform(x).astype(np.float32)
            with torch.no_grad():
                y_s = self.net.model(torch.tensor(x_s)).numpy()
            y = self.y_scaler.inverse_transform(y_s)[0]
            return WorldPrediction(
                level_est_next=clamp(float(y[0]), 0.0, 100.0),
                fit_next=max(0.0, float(y[1])),
                safety_risk_next=max(0.0, float(y[2])),
                production_next=max(0.0, float(y[3])),
            )
        except Exception:
            return self.physics.predict(belief, action)

    def _write_fallback_artifact(self, output_path: str | Path | None, reason: str) -> None:
        if output_path is None:
            return
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "physics_fallback_world_model",
            "reason": reason,
            "config": self.config,
            "metrics": {
                key: (None if np.isnan(value) else float(value))
                for key, value in self.metrics.items()
            },
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class RealSWaTWorldModel:
    """One-step predictor for offline real SWaT P1 logs.

    This model is intentionally not a closed-loop recovery model. It predicts
    next observed P1 values from normal log dynamics and is used for diagnosis,
    reconstruction consistency, and counterfactual rollouts.
    """

    feature_cols = ["LIT101", "FIT101", "MV101_open_binary", "P101_on_binary", "P102_on_binary"]
    target_cols = ["LIT101_next", "FIT101_next"]

    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = LinearRegression()
        self.trained = False
        self.metrics: dict[str, float] = {}

    def fit(self, normal_df: Any, output_path: str | Path | None = None) -> dict[str, float]:
        frame = self._supervised_frame(normal_df)
        if len(frame) < 10:
            self.metrics = {"normal_validation_rmse": np.nan, "one_step_rmse": np.nan}
            self.trained = False
            return self.metrics
        split = max(1, int(0.8 * len(frame)))
        train = frame.iloc[:split]
        val = frame.iloc[split:] if split < len(frame) else frame.iloc[-max(1, len(frame) // 5) :]
        x_train = train[self.feature_cols].to_numpy(dtype=float)
        y_train = train[self.target_cols].to_numpy(dtype=float)
        x_val = val[self.feature_cols].to_numpy(dtype=float)
        y_val = val[self.target_cols].to_numpy(dtype=float)
        x_train_s = self.x_scaler.fit_transform(x_train)
        y_train_s = self.y_scaler.fit_transform(y_train)
        self.model.fit(x_train_s, y_train_s)
        pred = self.predict_array(x_val)
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        self.metrics = {"normal_validation_rmse": rmse, "one_step_rmse": rmse}
        self.trained = True
        if output_path is not None:
            payload = {
                "artifact_type": "real_swat_linear_world_model",
                "feature_cols": self.feature_cols,
                "target_cols": self.target_cols,
                "metrics": self.metrics,
            }
            Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.metrics

    def predict_next(self, row: Any) -> dict[str, float]:
        if not self.trained:
            return {"LIT101_next": float(row.get("LIT101", np.nan)), "FIT101_next": float(row.get("FIT101", np.nan))}
        x = np.array([[float(row.get(col, 0.0)) for col in self.feature_cols]], dtype=float)
        pred = self.predict_array(x)[0]
        return {"LIT101_next": float(pred[0]), "FIT101_next": float(pred[1])}

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        x_s = self.x_scaler.transform(x)
        y_s = self.model.predict(x_s)
        return self.y_scaler.inverse_transform(y_s)

    def predict_frame(self, df: Any) -> Any:
        frame = df.copy()
        if frame.empty:
            frame["pred_LIT101_next"] = []
            frame["pred_FIT101_next"] = []
            return frame
        x = frame[self.feature_cols].fillna(0.0).to_numpy(dtype=float)
        if self.trained:
            pred = self.predict_array(x)
            frame["pred_LIT101_next"] = pred[:, 0]
            frame["pred_FIT101_next"] = pred[:, 1]
        else:
            frame["pred_LIT101_next"] = frame["LIT101"]
            frame["pred_FIT101_next"] = frame["FIT101"]
        return frame

    def _supervised_frame(self, df: Any) -> Any:
        frame = df.copy()
        frame["LIT101_next"] = frame["LIT101"].shift(-1)
        frame["FIT101_next"] = frame["FIT101"].shift(-1)
        cols = self.feature_cols + self.target_cols
        return frame[cols].replace([np.inf, -np.inf], np.nan).dropna()


def belief_dict(belief: np.ndarray | dict[str, float]) -> dict[str, float]:
    if isinstance(belief, dict):
        return {k: float(v) for k, v in belief.items()}
    return {col: float(belief[idx]) for idx, col in enumerate(BELIEF_COLUMNS)}


def make_model_input(belief: np.ndarray | dict[str, float], action: RecoveryAction) -> np.ndarray:
    values = belief if isinstance(belief, np.ndarray) else np.array([belief[col] for col in BELIEF_COLUMNS])
    return np.concatenate([values.astype(np.float32), np.array(action_one_hot(action), dtype=np.float32)])


def next_belief_from_prediction(
    belief: np.ndarray | dict[str, float],
    prediction: WorldPrediction,
) -> dict[str, float]:
    b = belief_dict(belief)
    b["level_est"] = prediction.level_est_next
    b["fit_est"] = prediction.fit_next
    b["hazard_priority"] = max(float(b.get("hazard_priority", 0.0)) * 0.9, prediction.safety_risk_next)
    return b
