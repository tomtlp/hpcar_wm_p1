import numpy as np
import pandas as pd

from src.causal_logic import diagnose_real_swat_timeseries
from src.p1_simulator import P1Simulator
from src.real_swat_experiment import generate_case_studies, valid_p1_windows
from src.recovery_actions import RecoveryAction
from src.swat_attack_windows import parse_attack_windows


def _attack_frame(n=10):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00:00", periods=n, freq="s"),
            "label": [0] * n,
        }
    )


def test_out_of_range_attack_window_is_excluded_not_last_aligned(tmp_path):
    attack_df = _attack_frame()
    attack_list = pd.DataFrame(
        {
            "Attack #": [1],
            "Start Time": [pd.Timestamp("2020-01-01 00:01:00")],
            "End Time": [pd.Timestamp("2020-01-01 00:02:00")],
            "Attack Point": ["MV101"],
        }
    )
    path = tmp_path / "List_of_attacks.csv"
    attack_list.to_csv(path, index=False)

    windows = parse_attack_windows(path, attack_df, tmp_path)
    row = windows.iloc[0]

    assert row["alignment_status"] == "out_of_loaded_range_after"
    assert bool(row["exclude_from_eval"])
    assert pd.isna(row["start_index"])
    assert pd.isna(row["end_index"])


def test_partial_overlap_window_is_clipped_to_loaded_range(tmp_path):
    attack_df = _attack_frame()
    attack_list = pd.DataFrame(
        {
            "Attack #": [1],
            "Start Time": [pd.Timestamp("2019-12-31 23:59:58")],
            "End Time": [pd.Timestamp("2020-01-01 00:00:03")],
            "Attack Point": ["LIT101"],
        }
    )
    path = tmp_path / "List_of_attacks.csv"
    attack_list.to_csv(path, index=False)

    windows = parse_attack_windows(path, attack_df, tmp_path)
    row = windows.iloc[0]

    assert row["alignment_status"] == "partial_overlap"
    assert int(row["start_index"]) == 0
    assert int(row["end_index"]) == 3
    assert not bool(row["exclude_from_eval"])


def test_valid_p1_windows_excludes_invalid_rows():
    windows = pd.DataFrame(
        [
            {"attack_id": 1, "start_index": 0, "end_index": 5, "duration": 6, "exclude_from_eval": False, "alignment_status": "inside_loaded_range"},
            {"attack_id": 2, "start_index": np.nan, "end_index": np.nan, "duration": 0, "exclude_from_eval": True, "alignment_status": "out_of_loaded_range_after"},
            {"attack_id": 3, "start_index": 8, "end_index": 8, "duration": 1, "exclude_from_eval": False, "alignment_status": "inside_loaded_range"},
        ]
    )

    valid = valid_p1_windows(windows)

    assert valid["attack_id"].tolist() == [1]


def test_mv101_actuator_diagnosis_detects_closed_high_fit_mismatch():
    df = pd.DataFrame(
        {
            "t": range(8),
            "LIT101": np.linspace(50, 52, 8),
            "FIT101": [1.2] * 8,
            "MV101_open_binary": [0] * 8,
            "P101_on_binary": [0] * 8,
            "P102_on_binary": [0] * 8,
        }
    )
    cal = {"ok": False, "rmse": 0.1, "mv101_fit_open_mean": 1.2, "mv101_fit_closed_mean": 0.0, "safe_low": 40, "safe_high": 80, "target_low": 45, "target_high": 60}

    diag = diagnose_real_swat_timeseries(df, cal, {"persistence_steps": 2})

    assert diag["mv101_suspicion_score"].max() > 0
    assert (diag["trust_MV101"] == 0).any()


def test_pump_diagnosis_detects_pump_on_without_level_decrease():
    df = pd.DataFrame(
        {
            "t": range(8),
            "LIT101": np.linspace(50, 55, 8),
            "FIT101": [1.0] * 8,
            "MV101_open_binary": [1] * 8,
            "P101_on_binary": [1] * 8,
            "P102_on_binary": [0] * 8,
        }
    )
    cal = {"ok": False, "rmse": 0.1, "mv101_fit_open_mean": 1.0, "mv101_fit_closed_mean": 0.0, "safe_low": 40, "safe_high": 80, "target_low": 45, "target_high": 60}

    diag = diagnose_real_swat_timeseries(df, cal, {"persistence_steps": 2})

    assert diag["p101_suspicion_score"].max() > 0


def test_r9_emergency_drain_changes_outflow_and_level_delta():
    sim = P1Simulator(seed=0)
    sim.reset(seed=0, initial_level=85.0)

    _, info = sim.step(
        RecoveryAction.R9_EMERGENCY_DRAIN_BOTH_PUMPS,
        level_est=85.0,
        recovery_context={"trust_P101": 1, "trust_P102": 1},
    )

    assert info["outflow"] > 0
    assert info["level_delta"] < 0


def test_case_study_generation_skips_invalid_window(tmp_path):
    attack_df = pd.DataFrame({"t": range(5), "LIT101": np.linspace(50, 51, 5)})
    windows = pd.DataFrame(
        [
            {
                "attack_id": "1",
                "p1_target_tag": "MV101",
                "start_index": np.nan,
                "end_index": np.nan,
                "duration": 0,
                "alignment_status": "out_of_loaded_range_after",
                "alignment_status_detail": "outside quick range",
                "exclude_from_eval": True,
            }
        ]
    )

    generate_case_studies(attack_df, windows, pd.DataFrame(), pd.DataFrame(), {}, tmp_path)

    summary = pd.read_csv(tmp_path / "case_study_attack_1_MV101_summary.csv")
    assert summary.loc[0, "status"] == "skipped"
