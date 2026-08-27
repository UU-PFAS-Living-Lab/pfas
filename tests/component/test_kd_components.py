import pytest

from pfas.component.kd import (
    LinearSPsorption,
    FreundlichSPsorption,
)


class Dummy:
    """Simple object to bypass Pydantic validation."""
    pass


@pytest.fixture
def sorption_linear_direct() -> dict:
    """Linear isotherm with direct Kd input."""
    return {
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 0.042,
        },
    }


@pytest.fixture
def sorption_linear_fp() -> dict:
    """Linear isotherm using Fabregat-Palau method."""
    return {
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "fabregat_palau",
            "n_CFx": 7,
            "f_oc": 0.0004,
            "f_silt_clay": 0.0,
        },
    }


@pytest.fixture
def sorption_freundlich() -> dict:
    """Freundlich isotherm configuration."""
    return {
        "sorption_isotherm": "freundlich",
        "freundlich": {
            "K_freund": 0.8,
            "n_freund": 0.7,
            "C_rep": 1.0,
        },
    }


def test_linear_kd_direct(sorption_linear_direct: dict):
    """Test linear Kd using direct input."""
    obj = Dummy()
    obj.sorption_solid = sorption_linear_direct

    out = LinearSPsorption.compute(obj)
    assert "Kd" in out
    assert out["Kd"] == 0.042


def test_linear_kd_fabregat_palau(sorption_linear_fp: dict):
    """Test linear Kd using Fabregat-Palau correlation."""
    obj = Dummy()
    obj.sorption_solid = sorption_linear_fp

    out = LinearSPsorption.compute(obj)
    assert "Kd" in out
    assert out["Kd"] > 0.0


def test_linear_kd_missing_linear_key():
    """Test missing 'linear' key raises an error."""
    obj = Dummy()
    obj.sorption_solid = {"sorption_isotherm": "linear"}

    with pytest.raises(ValueError):
        LinearSPsorption.compute(obj)


def test_linear_kd_missing_direct_input_value():
    """Test missing Kd for direct_input raises an error."""
    obj = Dummy()
    obj.sorption_solid = {
        "sorption_isotherm": "linear",
        "linear": {"Kd_method": "direct_input"},
    }

    with pytest.raises(ValueError):
        LinearSPsorption.compute(obj)


def test_linear_kd_missing_fp_keys():
    """Test missing Fabregat-Palau keys raise an error."""
    obj = Dummy()
    obj.sorption_solid = {
        "sorption_isotherm": "linear",
        "linear": {"Kd_method": "fabregat_palau"},
    }

    with pytest.raises(ValueError):
        LinearSPsorption.compute(obj)


def test_freundlich_kd_compute(sorption_freundlich: dict):
    """Test Freundlich Kd calculation."""
    obj = Dummy()
    obj.sorption_solid = sorption_freundlich

    out = FreundlichSPsorption.compute(obj)
    assert "Kd" in out
    assert out["Kd"] > 0.0


def test_freundlich_missing_key():
    """Test missing freundlich key raises an error."""
    obj = Dummy()
    obj.sorption_solid = {"sorption_isotherm": "freundlich"}

    with pytest.raises(ValueError):
        FreundlichSPsorption.compute(obj)


def test_freundlich_missing_parameters():
    """Test missing K_freund or n_freund raises an error."""
    obj = Dummy()
    obj.sorption_solid = {
        "sorption_isotherm": "freundlich",
        "freundlich": {"K_freund": 0.8},
    }

    with pytest.raises(ValueError):
        FreundlichSPsorption.compute(obj)

