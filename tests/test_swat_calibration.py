import numpy as np
import pandas as pd

from src.swat_calibration import calibrate_p1_from_normal


def test_calibration_returns_coefficients_and_finite_rmse(tmp_path):
    n = 40
    df = pd.DataFrame(
        {
            "LIT101": np.linspace(45, 55, n),
            "FIT101": np.ones(n),
            "MV101_open_binary": np.ones(n),
            "P101_on_binary": np.ones(n),
            "P102_on_binary": np.zeros(n),
        }
    )
    result = calibrate_p1_from_normal(df, tmp_path)
    assert result["ok"]
    assert "coefficients" in result
    assert np.isfinite(result["rmse"])
    assert (tmp_path / "swat_p1_calibration.json").exists()
