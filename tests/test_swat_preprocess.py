import pandas as pd

from src.swat_preprocess import infer_actuator_mapping, parse_label_series, preprocess_swat_dataframe


def test_label_parser_maps_normal_attack():
    labels = parse_label_series(pd.Series(["Normal", "Attack", "Normal", "Attack"]))
    assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_actuator_mapper_infers_mv101_open_from_higher_fit():
    df = pd.DataFrame({"MV101": [1, 1, 2, 2], "FIT101": [0.1, 0.2, 2.0, 2.2]})
    mapping, meta = infer_actuator_mapping(df, "MV101", "FIT101")
    assert mapping[2] == 1
    assert mapping[1] == 0
    assert meta["evidence_used"] == "higher_mean_FIT101"


def test_preprocessor_handles_missing_values(tmp_path):
    df = pd.DataFrame(
        {
            " LIT 101 ": [50.0, None, 52.0],
            "fit_101": [1.0, None, 1.2],
            "mv 101": [1, 2, 2],
            "P_101": [1, 1, 2],
            "p 102": [1, 1, 1],
            "Normal/Attack": ["Normal", "Attack", "Attack"],
        }
    )
    out, mapping, actuator = preprocess_swat_dataframe(df, tmp_path)
    assert out["LIT101"].isna().sum() == 0
    assert out["FIT101"].isna().sum() == 0
    assert "MV101_open_binary" in out
    assert mapping["LIT101"] == " LIT 101 "
