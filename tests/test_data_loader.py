import json

import pytest
from pint import Quantity

from pfas.data_loader import available_datasets, load_dataset, load_json_file


def test_load_json_file_returns_raw_json(tmp_path):
    """load_json_file should not convert JSON values into quantities."""
    data_path = tmp_path / "sample.json"
    data_path.write_text(
        json.dumps({"mass": {"value": 12.5, "unit": "g/mol"}}),
        encoding="utf-8",
    )

    data = load_json_file(data_path)

    assert data == {"mass": {"value": 12.5, "unit": "g/mol"}}


def test_load_dataset_converts_packaged_unit_values_to_quantities():
    """Load applicable packaged dataset values as Pint quantities."""
    data = load_dataset("PFASs")
    pfhxa = data["PFHxA"]

    assert pfhxa["name"] == "PFHxA"
    assert isinstance(pfhxa["M"], Quantity)
    assert pfhxa["M"].to("gram / mole").magnitude == pytest.approx(314.05)
    assert isinstance(pfhxa["diffusivity"], Quantity)
    assert pfhxa["diffusivity"].to("centimeter ** 2 / second").magnitude == pytest.approx(
        7.8e-6
    )
    assert isinstance(pfhxa["K_oc"], Quantity)
    assert pfhxa["K_oc"].to("liter / kilogram").magnitude == pytest.approx(15.0)

    # Explicitly unitless and null fields remain plain values.
    assert pfhxa["Szyszkowski_params"]["b"] == 0.21
    assert not isinstance(pfhxa["Szyszkowski_params"]["b"], Quantity)
    assert pfhxa["K_mo"] is None
    assert pfhxa["structural_properties"]["n_CFx"] == 5


def test_load_dataset_can_skip_unit_conversion():
    """load_units=False should retain the source JSON value/unit structure."""
    data = load_dataset("PFASs", load_units=False)

    assert data["PFHxA"]["M"] == {"value": 314.05, "unit": "g/mol"}
    assert data["PFHxA"]["K_oc"] == {"value": 15.0, "unit": "L/kg"}


def test_load_dataset_converts_unit_values_from_user_path(tmp_path):
    """A JSON file path should use the same unit conversion as a packaged set."""
    data_path = tmp_path / "custom.json"
    data_path.write_text(
        json.dumps(
            {
                "with_units": {"value": 2.5, "unit": "m"},
                "unitless": {"value": 3, "unit": "-"},
                "empty": {"value": None, "unit": "kg"},
                "plain": 4,
            }
        ),
        encoding="utf-8",
    )

    data = load_dataset(data_path)

    assert isinstance(data["with_units"], Quantity)
    assert data["with_units"].to("centimeter").magnitude == pytest.approx(250)
    assert data["unitless"] == 3
    assert data["empty"] is None
    assert data["plain"] == 4


def test_available_datasets_includes_pfass():
    """available_datasets returns PFASs without the .json extension."""
    assert "PFASs" in available_datasets()


def test_load_json_file_raises_file_not_found_error_for_missing_file(tmp_path):
    """load_json_file raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_json_file(tmp_path / "missing.json")


def test_load_dataset_raises_value_error_for_invalid_dataset_name():
    """load_dataset raises ValueError for an unknown packaged dataset name."""
    invalid_name = "not_a_real_dataset"

    with pytest.raises(ValueError, match=f"Invalid dataset '{invalid_name}'"):
        load_dataset(invalid_name)
