import pytest
from pint import Quantity

from pfas import ureg
from pfas.component.kd import FreundlichSPsorption, LinearSPsorption


@pytest.fixture(params=[False, True], ids=["unitless", "pint"])
def with_units(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def sorption_linear_direct(with_units: bool) -> dict:
    """Linear isotherm with direct Kd input."""
    kd = 0.042
    if with_units:
        kd *= ureg.liter / ureg.kilogram
    return {
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": kd,
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
def sorption_freundlich(with_units: bool) -> dict:
    """Freundlich configuration with consistently unitless or Pint values."""
    k_freund = 0.8
    c_rep = 1.0
    if with_units:
        concentration_unit = ureg.mole / ureg.liter
        k_freund *= (
            ureg.liter / ureg.kilogram * concentration_unit**0.3
        )
        c_rep *= concentration_unit
    return {
        "sorption_isotherm": "freundlich",
        "freundlich": {
            "K_freund": k_freund,
            "n_freund": 0.7,
            "C_rep": c_rep,
        },
    }


def assert_kd_result(result: dict, with_units: bool) -> None:
    assert "Kd" in result
    kd = result["Kd"]
    if with_units:
        assert isinstance(kd, Quantity)
        assert kd.check("[length] ** 3 / [mass]")
        assert kd.magnitude > 0.0
    else:
        assert isinstance(kd, float)
        assert kd > 0.0


def test_linear_kd_direct(sorption_linear_direct: dict, with_units: bool):
    """Test direct-input linear Kd with and without Pint quantities."""
    component = LinearSPsorption(sorption_solid=sorption_linear_direct)
    result = component.compute()

    assert_kd_result(result, with_units)
    if with_units:
        assert result["Kd"].to("liter / kilogram").magnitude == pytest.approx(0.042)
    else:
        assert result["Kd"] == 0.042


def test_linear_kd_fabregat_palau(sorption_linear_fp: dict):
    """Test Fabregat-Palau Kd returns a quantity in L/kg."""
    component = LinearSPsorption(sorption_solid=sorption_linear_fp)
    result = component.compute()

    assert "Kd" in result
    assert isinstance(result["Kd"], Quantity)
    assert result["Kd"].check("[length] ** 3 / [mass]")
    assert result["Kd"].to("liter / kilogram").magnitude > 0.0


def test_linear_kd_missing_linear_key():
    """Test missing 'linear' key raises an error."""
    component = LinearSPsorption(sorption_solid={"sorption_isotherm": "linear"})

    with pytest.raises(ValueError, match="'linear' key is missing"):
        component.compute()


def test_linear_kd_missing_direct_input_value():
    """Test missing Kd for direct_input raises an error."""
    component = LinearSPsorption(
        sorption_solid={
            "sorption_isotherm": "linear",
            "linear": {"Kd_method": "direct_input"},
        }
    )

    with pytest.raises(ValueError, match="requires 'Kd'"):
        component.compute()


def test_linear_kd_missing_fp_keys():
    """Test missing Fabregat-Palau keys raise an error."""
    component = LinearSPsorption(
        sorption_solid={
            "sorption_isotherm": "linear",
            "linear": {"Kd_method": "fabregat_palau"},
        }
    )

    with pytest.raises(ValueError, match="requires 'n_CFx'"):
        component.compute()


def test_freundlich_kd_compute(sorption_freundlich: dict, with_units: bool):
    """Test Freundlich Kd calculation with and without Pint quantities."""
    component = FreundlichSPsorption(sorption_solid=sorption_freundlich)
    result = component.compute()

    assert_kd_result(result, with_units)
    if with_units:
        assert result["Kd"].to("liter / kilogram").magnitude == pytest.approx(0.8)
    else:
        assert result["Kd"] == pytest.approx(0.8)


def test_freundlich_missing_key():
    """Test missing freundlich key raises an error."""
    component = FreundlichSPsorption(
        sorption_solid={"sorption_isotherm": "freundlich"}
    )

    with pytest.raises(ValueError, match="configuration is missing"):
        component.compute()


def test_freundlich_missing_parameters():
    """Test missing Freundlich parameters raise an error."""
    component = FreundlichSPsorption(
        sorption_solid={
            "sorption_isotherm": "freundlich",
            "freundlich": {"K_freund": 0.8},
        }
    )

    with pytest.raises(ValueError, match="requires"):
        component.compute()
