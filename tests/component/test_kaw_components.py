import pytest
from pint import Quantity
from pydantic import ValidationError

from pfas import ureg
from pfas.component import Le2021_asymptote, Le2021_langmuir, Szyszkowski


@pytest.fixture(params=[False, True], ids=["unitless", "pint"])
def with_units(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def structural_props() -> dict:
    """Provide structural properties valid for Le2021 models."""
    return {
        "n_CFx": 7,
        "n_CHx": 1,
        "n_COO": 0,
        "n_COOH": 0,
        "n_SO3": 0,
        "n_R4N": 0,
        "n_OH": 0,
        "n_OSO3": 0,
        "n__O_": 0,
        "n__S_": 0,
        "n_N_CH3_2_CH2_COO": 0,
    }


@pytest.fixture
def szyszkowski_params(with_units: bool) -> dict:
    """Szyszkowski parameters, consistently unitless or Pint-backed."""
    params = {
        "sigma0": 0.072,
        "a": 0.5,
        "b": 1.2,
        "chi": 2,
        "T": 298.0,
    }
    if with_units:
        params["sigma0"] *= ureg.newton / ureg.meter
        params["a"] *= ureg.mole / ureg.liter
        params["T"] *= ureg.kelvin
    return params


def assert_kaw_result(result: dict, with_units: bool) -> None:
    assert "Kaw" in result
    kaw = result["Kaw"]
    if with_units:
        assert isinstance(kaw, Quantity)
        assert kaw.check("[length]")
        assert kaw.magnitude > 0.0
    else:
        assert isinstance(kaw, float)
        assert kaw > 0.0


def test_le2021_asymptote_component_compute(structural_props: dict):
    """Component wraps Kaw_0_Le2021 and exposes it via compute()."""
    component = Le2021_asymptote(structural_properties=structural_props)
    result = component.compute()

    assert "Kaw" in result
    assert result["Kaw"] > 0.0
    assert component.outputs == ["Kaw"]


def test_le2021_asymptote_component_missing_keys():
    """Missing structural keys should raise a validation error at construction."""
    with pytest.raises(ValidationError):
        Le2021_asymptote(structural_properties={"n_CFx": 7})


def test_le2021_langmuir_component_compute(
    structural_props: dict, with_units: bool
):
    """Component computes Kaw with unitless or Pint inputs."""
    concentration = 1e-6
    kwargs = {"omega": 55.3, "T": 298.0}
    if with_units:
        concentration *= ureg.mole / ureg.liter
        kwargs = {
            "omega": 55.3 * ureg.mole / ureg.liter,
            "T": 298.0 * ureg.kelvin,
        }

    component = Le2021_langmuir(
        structural_properties=structural_props,
        Cw=concentration,
        **kwargs,
    )
    result = component.compute()

    assert_kaw_result(result, with_units)
    assert component.outputs == ["Kaw"]


def test_le2021_langmuir_component_missing_keys():
    """Missing structural keys should raise a validation error at construction."""
    with pytest.raises(ValidationError):
        Le2021_langmuir(structural_properties={"n_CFx": 7}, Cw=1e-6)


def test_le2021_langmuir_component_varies_with_concentration(
    structural_props: dict, with_units: bool
):
    """Kaw should differ between concentrations in either unit mode."""
    low_cw = 1e-9
    high_cw = 1e-3
    kwargs = {"omega": 55.3, "T": 298.0}
    if with_units:
        concentration_unit = ureg.mole / ureg.liter
        low_cw *= concentration_unit
        high_cw *= concentration_unit
        kwargs = {
            "omega": 55.3 * concentration_unit,
            "T": 298.0 * ureg.kelvin,
        }

    low = Le2021_langmuir(
        structural_properties=structural_props, Cw=low_cw, **kwargs
    )
    high = Le2021_langmuir(
        structural_properties=structural_props, Cw=high_cw, **kwargs
    )

    assert low.compute()["Kaw"] != high.compute()["Kaw"]


def test_szyszkowski_component_compute(
    szyszkowski_params: dict, with_units: bool
):
    """Component computes Kaw with unitless or Pint inputs."""
    concentration = 1e-6
    if with_units:
        concentration *= ureg.mole / ureg.liter
    component = Szyszkowski(**szyszkowski_params, Cw=concentration)
    result = component.compute()

    assert_kaw_result(result, with_units)
    assert component.outputs == ["Kaw"]


def test_szyszkowski_component_missing_required_field():
    """Omitting Cw (required, no default) should raise a validation error."""
    with pytest.raises(ValidationError):
        Szyszkowski(sigma0=0.072, a=0.5, b=1.2, chi=2, T=298.0)


def test_szyszkowski_component_defaults():
    """sigma0, chi, and T should retain their documented defaults."""
    component = Szyszkowski(a=0.5, b=1.2, Cw=1e-6)

    assert component.sigma0 == 0.072 * ureg.newton / ureg.meter
    assert component.chi == 2
    assert component.T == 298.0 * ureg.kelvin
