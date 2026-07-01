import pandas as pd

from src.swat_loader import discover_swat_files, find_tag_columns


def test_column_mapper_finds_p1_variants():
    columns = [" Timestamp ", "lit 101", "FIT_101", " mv_101 ", "p 101", "P_102", "Normal/Attack"]
    mapping = find_tag_columns(columns)
    assert mapping["LIT101"] == "lit 101"
    assert mapping["FIT101"] == "FIT_101"
    assert mapping["MV101"] == " mv_101 "
    assert mapping["P101"] == "p 101"
    assert mapping["P102"] == "P_102"


def test_discover_swat_files_writes_inventory(tmp_path):
    swat = tmp_path / "SWat"
    swat.mkdir()
    pd.DataFrame({"LIT101": [1], "FIT101": [2], "MV101": [1], "P101": [1], "P102": [1]}).to_csv(
        swat / "SWaT_Dataset_Normal_fake.csv", index=False
    )
    inventory = discover_swat_files(swat, tmp_path)
    assert not inventory.empty
    assert (tmp_path / "swat_file_inventory.csv").exists()
    assert "normal_csv" in set(inventory["role"])
