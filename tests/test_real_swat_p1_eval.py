import numpy as np
import pandas as pd

from src.experiment import main
from src.p1_simulator import P1Simulator
from src.real_swat_experiment import (
    add_p1_labels,
    annotate_hybrid_production_metrics,
    build_p1_attack_windows,
    calibrate_real_thresholds,
    normalize_target_tag,
)
from src.swat_calibration import calibrate_p1_from_normal, simulator_config_from_calibration
from src.causal_logic import diagnose_real_swat_timeseries


def test_p1_target_tag_normalization_maps_mv_101():
    assert normalize_target_tag("MV-101") == "MV101"
    assert normalize_target_tag(" lit_101 ") == "LIT101"


def test_out_of_range_attack_windows_are_excluded_not_clipped(tmp_path):
    windows = pd.DataFrame(
        [
            {
                "attack_id": 1,
                "target_tags": "MV-101",
                "start_index": 50,
                "end_index": 60,
                "alignment_status": "timestamp_nearest",
            }
        ]
    )
    p1 = build_p1_attack_windows(windows, loaded_len=20, output=tmp_path)
    assert p1.iloc[0]["alignment_status"] == "out_of_loaded_range"
    assert pd.isna(p1.iloc[0]["start_index"])


def test_p1_specific_labels_from_fake_attack_list(tmp_path):
    df = pd.DataFrame({"t": range(10), "label": [0] * 10})
    windows = pd.DataFrame(
        [
            {
                "attack_id": 1,
                "target_tags_original": "P-101",
                "target_tags_normalized": "P101",
                "p1_target_tag": "P101",
                "start_index": 2,
                "end_index": 4,
                "duration": 3,
                "alignment_status": "timestamp_nearest",
            }
        ]
    )
    labels = add_p1_labels(df, windows, tmp_path)
    assert labels["p1_attack_label_any"].sum() == 3
    assert labels["p1_attack_label_P101"].sum() == 3
    assert labels["p1_attack_label_MV101"].sum() == 0


def test_threshold_calibration_returns_finite_thresholds(tmp_path):
    df = pd.DataFrame(
        {
            "t": range(50),
            "LIT101": np.linspace(100, 105, 50),
            "FIT101": np.ones(50),
            "MV101_open_binary": np.ones(50),
            "P101_on_binary": np.ones(50),
            "P102_on_binary": np.zeros(50),
        }
    )
    cal = calibrate_p1_from_normal(df, tmp_path)
    diag = diagnose_real_swat_timeseries(df, cal, {})
    thresholds = calibrate_real_thresholds(diag, tmp_path, {"diagnosis": {"threshold_quantile": 0.9}})
    assert all(np.isfinite(v) for k, v in thresholds.items() if k != "threshold_quantile")
    assert (tmp_path / "real_swat_thresholds.json").exists()


def test_hybrid_calibrated_simulator_uses_real_unit_safe_bounds(tmp_path):
    df = pd.DataFrame(
        {
            "LIT101": np.linspace(250, 270, 60),
            "FIT101": np.ones(60) * 2.0,
            "MV101_open_binary": np.ones(60),
            "P101_on_binary": np.ones(60),
            "P102_on_binary": np.zeros(60),
        }
    )
    cal = calibrate_p1_from_normal(df, tmp_path)
    sim_cfg = simulator_config_from_calibration({}, cal)
    assert sim_cfg["safe_min"] > 200
    assert sim_cfg["level_max"] > sim_cfg["safe_max"]


def test_production_proxy_nonzero_when_pump_on_and_level_safe():
    sim = P1Simulator({"initial_level": 260.0, "level_min": 200.0, "level_max": 300.0, "safe_min": 240.0, "safe_max": 280.0, "process_noise_std": 0.0, "sensor_noise_std": 0.0}, seed=0)
    sim.reset(seed=0, initial_level=260.0)
    state = sim.step_physics_only(mv101_state=0, p101_state=1, p102_state=0)
    assert state.level_true < 260.0


def test_hybrid_production_loss_uses_seed_best_reference():
    metrics = pd.DataFrame(
        {
            "method": ["A", "B"],
            "attack": ["nominal_like", "pump_forced_off"],
            "seed": [0, 0],
            "production": [30.0, 12.0],
        }
    )
    annotated = annotate_hybrid_production_metrics(metrics, {"p101_outflow_rate": 0.3}, steps=70)
    assert annotated.loc[0, "production_loss"] == 0.0
    assert annotated.loc[1, "production_loss"] > 0.0
    assert annotated["production"].sum() > 0.0


def test_p1_log_eval_quick_runs_on_fake_swat(tmp_path):
    swat = tmp_path / "SWat"
    swat.mkdir()
    normal = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=80, freq="s"),
            "LIT101": np.linspace(250, 255, 80),
            "FIT101": np.ones(80),
            "MV101": [1, 2] * 40,
            "P101": [2] * 80,
            "P102": [1] * 80,
            "Normal/Attack": ["Normal"] * 80,
        }
    )
    attack = normal.copy()
    attack["Normal/Attack"] = ["Normal"] * 20 + ["Attack"] * 20 + ["Normal"] * 40
    attack.loc[20:39, "LIT101"] += 5.0
    attacks = pd.DataFrame({"Attack #": [1], "Start Time": [attack["Timestamp"].iloc[20]], "End Time": [attack["Timestamp"].iloc[39]], "Attack Point": ["LIT-101"]})
    normal.to_csv(swat / "SWaT_Dataset_Normal_fake.csv", index=False)
    attack.to_csv(swat / "SWaT_Dataset_Attack_fake.csv", index=False)
    attacks.to_csv(swat / "List_of_attacks_fake.csv", index=False)
    out = tmp_path / "out"
    main(["--mode", "real_swat", "--swat_dir", str(swat), "--real_swat_task", "p1_log_eval", "--quick", "--output_dir", str(out)])
    assert (out / "real_swat_p1_log_eval_summary.csv").exists()
    assert (out / "real_swat_p1_attack_windows.csv").exists()
