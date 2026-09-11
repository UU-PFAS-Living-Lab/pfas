import pytest
from importlib import resources

import pytest

from pfas.data_loader import (
    available_datasets,
    load_dataset,
    load_json_file,
)


def test_load_json_file_loads_pfass_data_file():
    """load_json_file loads the packaged PFASs.json file from its path."""
    
    data = load_json_file("src/pfas/data/PFASs.json")

    assert data["PFOA"]["name"] == "PFOA"
    assert data["PFOA"]["M"]["value"] == 414.07
    assert data["PFOA"]["K_oc"]["unit"] == "L/kg"


def test_load_dataset_loads_pfass_by_name():
    """load_dataset loads PFASs.json when called with its dataset name."""
    data = load_dataset("PFASs")

    assert data["PFOS"]["name"] == "PFOS"
    assert data["PFOS"]["M"]["value"] == 500.13
    assert data["PFOS"]["structural_properties"]["n_SO3"] == 1


def test_available_datasets_includes_pfass():
    """available_datasets returns PFASs without the .json extension."""
    datasets = available_datasets()

    assert "PFASs" in datasets


#Negative tests 
def test_load_json_file_raises_file_not_found_error_for_missing_file(tmp_path):
    """load_json_file raises FileNotFoundError when the file does not exist."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_json_file(missing_path)

def test_load_dataset_raises_value_error_for_invalid_dataset_name():
    """load_dataset raises ValueError for an unknown packaged dataset name."""
    invalid_name = "not_a_real_dataset"

    with pytest.raises(
        ValueError,
        match=f"Invalid dataset '{invalid_name}'",
    ):
        load_dataset(invalid_name)