import pandas as pd

from src.experiment import main


def _write_fake_swat(root):
    root.mkdir(parents=True, exist_ok=True)
    normal = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=80, freq="s"),
            "LIT 101": [50 + i * 0.01 for i in range(80)],
            "FIT_101": [1.0] * 40 + [1.2] * 40,
            "MV 101": [1] * 40 + [2] * 40,
            "P101": [1, 2] * 40,
            "P_102": [1] * 80,
            "Normal/Attack": ["Normal"] * 80,
        }
    )
    attack = normal.copy()
    attack["Normal/Attack"] = ["Normal"] * 30 + ["Attack"] * 30 + ["Normal"] * 20
    attack.loc[30:60, "LIT 101"] += 5.0
    normal.to_csv(root / "SWaT_Dataset_Normal_fake.csv", index=False)
    attack.to_csv(root / "SWaT_Dataset_Attack_fake.csv", index=False)


def test_real_swat_mode_gracefully_skips_missing_path(tmp_path):
    result = main(
        [
            "--mode",
            "real_swat",
            "--swat_dir",
            str(tmp_path / "missing"),
            "--real_swat_task",
            "log_eval",
            "--quick",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    assert (tmp_path / "out" / "real_swat_error.txt").exists()
    assert result["output_dir"].exists()


def test_real_swat_quick_runs_on_fake_csv(tmp_path):
    swat = tmp_path / "dataset" / "SWat"
    _write_fake_swat(swat)
    out = tmp_path / "out"
    main(
        [
            "--mode",
            "real_swat",
            "--swat_dir",
            str(swat),
            "--real_swat_task",
            "log_eval",
            "--quick",
            "--output_dir",
            str(out),
        ]
    )
    assert (out / "swat_file_inventory.csv").exists()
    assert (out / "swat_column_mapping.json").exists()
    assert (out / "real_swat_log_eval_summary.csv").exists()
    assert (out / "real_swat_world_model_eval.csv").exists()
    assert (out / "real_swat_trust_timeseries.csv").exists()
