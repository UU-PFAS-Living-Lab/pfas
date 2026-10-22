import pytest
from pint import Quantity
from pydantic import ValidationError

from pfas import ureg
from pfas.component.retardation import Retardation
from pfas.data_structure import Adsorption, HydrologicalProperties


@pytest.fixture(params=[False, True], ids=["unitless", "pint"])
def with_units(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def hydro(with_units: bool) -> HydrologicalProperties:
    water_content = 0.3
    if with_units:
        water_content *= ureg.dimensionless
    return HydrologicalProperties(
        water_content=water_content,
        pore_velocity=0.0001,
        dispersion_coefficient=1e-6,
    )


@pytest.fixture
def retardation_inputs(with_units: bool) -> dict:
    """Dimensionally consistent retardation inputs."""
    inputs = {
        "Kd": 0.5,
        "Kaw": 0.02,
        "aaw": 10.0,
        "bulk_density": 1.6,
    }
    if with_units:
        inputs.update(
            Kd=0.5 * ureg.liter / ureg.kilogram,
            Kaw=0.02 * ureg.centimeter,
            aaw=10.0 / ureg.centimeter,
            bulk_density=1.6 * ureg.kilogram / ureg.liter,
        )
    return inputs


def assert_retardation_value(value: float | Quantity, expected: float, with_units: bool):
    if with_units:
        assert isinstance(value, Quantity)
        assert value.check("")
        assert value.to("dimensionless").magnitude == pytest.approx(expected)
    else:
        assert isinstance(value, float)
        assert value == pytest.approx(expected)


def test_retardation_equilibrium(
    hydro: HydrologicalProperties, retardation_inputs: dict, with_units: bool
):
    """Test equilibrium retardation with and without Pint quantities."""
    component = Retardation(
        **retardation_inputs,
        kinetic=False,
        hydro_properties=hydro,
    )
    adsorption = component.compute()["adsorption"]

    assert isinstance(adsorption, Adsorption)
    assert_retardation_value(adsorption.sp_retardation, 8 / 3, with_units)
    assert_retardation_value(adsorption.awi_retardation, 2 / 3, with_units)
    assert adsorption.rate_const == 0.0
    assert adsorption.frac_int == 1.0


def test_retardation_kinetic(
    hydro: HydrologicalProperties, retardation_inputs: dict, with_units: bool
):
    """Test kinetic retardation with and without Pint quantities."""
    component = Retardation(
        **retardation_inputs,
        kinetic=True,
        kin_params={"rate_const": 0.1, "frac_int": 0.3},
        hydro_properties=hydro,
    )
    adsorption = component.compute()["adsorption"]

    assert isinstance(adsorption, Adsorption)
    assert adsorption.rate_const == 0.1
    assert adsorption.frac_int == 0.3
    assert_retardation_value(adsorption.sp_retardation, 8 / 3, with_units)
    assert_retardation_value(adsorption.awi_retardation, 2 / 3, with_units)


def test_retardation_missing_kin_params(
    hydro: HydrologicalProperties, retardation_inputs: dict
):
    """Missing kinetic parameters should fail model validation."""
    with pytest.raises(ValidationError, match="kin_params must be provided"):
        Retardation(
            **retardation_inputs,
            kinetic=True,
            kin_params=None,
            hydro_properties=hydro,
        )
