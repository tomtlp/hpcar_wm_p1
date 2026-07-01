"""Calibration of a lightweight P1 model from real SWaT normal data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def calibrate_p1_from_normal(
    normal_df: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if normal_df.empty or "LIT101" not in normal_df or "FIT101" not in normal_df:
        calibration = {"ok": False, "warning": "normal data missing LIT101/FIT101"}
        _write_calibration(output, calibration)
        return calibration

    lit = pd.to_numeric(normal_df["LIT101"], errors="coerce").dropna()
    q = cfg.get("calibration", {})
    safe_low = float(lit.quantile(float(q.get("normal_quantile_safe_low", 0.05))))
    safe_high = float(lit.quantile(float(q.get("normal_quantile_safe_high", 0.95))))
    target_low = float(lit.quantile(float(q.get("normal_quantile_target_low", 0.35))))
    target_high = float(lit.quantile(float(q.get("normal_quantile_target_high", 0.65))))

    mv_open_rate = float(normal_df.loc[normal_df.get("MV101_open_binary", 0) == 1, "FIT101"].mean())
    mv_closed_rate = float(normal_df.loc[normal_df.get("MV101_open_binary", 0) == 0, "FIT101"].mean())
    if not np.isfinite(mv_open_rate):
        mv_open_rate = float(normal_df["FIT101"].mean())
    if not np.isfinite(mv_closed_rate):
        mv_closed_rate = 0.0

    frame = normal_df[["LIT101", "FIT101", "P101_on_binary", "P102_on_binary"]].copy()
    frame["LIT101_next"] = normal_df["LIT101"].shift(-1)
    frame = frame.dropna()
    if len(frame) < 5:
        calibration = {"ok": False, "warning": "not enough rows for calibration"}
        _write_calibration(output, calibration)
        return calibration
    x = frame[["LIT101", "FIT101", "P101_on_binary", "P102_on_binary"]].to_numpy(dtype=float)
    y = frame["LIT101_next"].to_numpy(dtype=float)
    try:
        model = HuberRegressor().fit(x, y)
        model_type = "HuberRegressor"
        pred = model.predict(x)
        coef = model.coef_.tolist()
        intercept = float(model.intercept_)
    except Exception:
        model = LinearRegression().fit(x, y)
        model_type = "LinearRegression"
        pred = model.predict(x)
        coef = model.coef_.tolist()
        intercept = float(model.intercept_)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred)) if len(y) > 1 else np.nan
    calibration = {
        "ok": True,
        "model_type": model_type,
        "features": ["LIT101", "FIT101", "P101_on_binary", "P102_on_binary"],
        "intercept": intercept,
        "coefficients": {
            "beta_lit": float(coef[0]),
            "beta_fit": float(coef[1]),
            "beta_p101": float(coef[2]),
            "beta_p102": float(coef[3]),
        },
        "rmse": rmse,
        "r2": r2,
        "safe_low": safe_low,
        "safe_high": safe_high,
        "target_low": target_low,
        "target_high": target_high,
        "mv101_fit_open_mean": mv_open_rate,
        "mv101_fit_closed_mean": mv_closed_rate,
        "warning": "weak_calibration" if (not np.isfinite(r2) or r2 < 0.2) else "",
    }
    _write_calibration(output, calibration)
    return calibration


def predict_lit_next_from_calibration(row: pd.Series, calibration: dict[str, Any]) -> float:
    if not calibration.get("ok"):
        return float(row.get("LIT101", np.nan))
    b = calibration["coefficients"]
    return float(
        calibration["intercept"]
        + b["beta_lit"] * float(row.get("LIT101", 0.0))
        + b["beta_fit"] * float(row.get("FIT101", 0.0))
        + b["beta_p101"] * float(row.get("P101_on_binary", 0.0))
        + b["beta_p102"] * float(row.get("P102_on_binary", 0.0))
    )


def simulator_config_from_calibration(base_config: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base_config)
    if not calibration.get("ok"):
        return cfg
    cfg["safe_min"] = float(calibration.get("safe_low", cfg.get("safe_min", 20.0)))
    cfg["safe_max"] = float(calibration.get("safe_high", cfg.get("safe_max", 80.0)))
    cfg["target_min"] = float(calibration.get("target_low", cfg.get("target_min", 45.0)))
    cfg["target_max"] = float(calibration.get("target_high", cfg.get("target_max", 60.0)))
    cfg["initial_level"] = (cfg["target_min"] + cfg["target_max"]) / 2.0
    cfg["inflow_rate_open"] = max(0.01, float(calibration.get("mv101_fit_open_mean", cfg.get("inflow_rate_open", 1.2))))
    cfg["inflow_rate_closed"] = max(0.0, float(calibration.get("mv101_fit_closed_mean", cfg.get("inflow_rate_closed", 0.02))))
    return cfg


def _write_calibration(output: Path, calibration: dict[str, Any]) -> None:
    (output / "swat_p1_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([calibration]).to_csv(output / "swat_p1_calibration_report.csv", index=False)
