import pytest
from pint import Quantity

from pfas import ureg
from pfas.component.awi import (
    D50AWI,
    GSSAAWI,
    GuoTracer,
    NonlinearD50AWI,
    SWCsorption,
)
from pfas.data_structure import HydrologicalProperties


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
        pore_velocity=1e-4,
        dispersion_coefficient=1e-6,
    )


@pytest.fixture
def soil_swc(with_units: bool) -> dict:
    alpha = 0.8
    if with_units:
        alpha /= ureg.meter
    return {
        "porosity": 0.4,
        "van_genuchten_alpha": alpha,
        "van_genuchten_n": 1.6,
        "residual_water_content": 0.05,
    }


@pytest.fixture
def soil_d50(with_units: bool) -> dict:
    d50 = 0.02
    if with_units:
        d50 *= ureg.centimeter
    return {
        "porosity": 0.4,
        "d50": d50,
    }


@pytest.fixture
def guo_params(with_units: bool) -> dict:
    coefficient_unit = 1
    if with_units:
        coefficient_unit /= ureg.centimeter
    return {
        "Guo": {
            "guo_x0": 0.1 * coefficient_unit,
            "guo_x1": 0.2 * coefficient_unit,
            "guo_x2": 0.3 * coefficient_unit,
        }
    }


def assert_aaw_result(out: dict, with_units: bool) -> None:
    assert "aaw" in out
    aaw = out["aaw"]
    if with_units:
        assert isinstance(aaw, Quantity)
        assert aaw.check("1/[length]")
        assert aaw.magnitude >= 0.0
    else:
        assert isinstance(aaw, float)
        assert aaw >= 0.0


def test_swc_sorption_compute(
    hydro: HydrologicalProperties, soil_swc: dict, with_units: bool
):
    """Test thermodynamic AWI calculation with and without Pint quantities."""
    swc = SWCsorption(
        hydro_properties=hydro,
        sigma0=0.072 * ureg.newton / ureg.meter if with_units else 0.072,
        scaling_factor_awi=1.0,
        porosity=soil_swc["porosity"],
        van_genuchten_alpha=soil_swc["van_genuchten_alpha"],
        van_genuchten_n=soil_swc["van_genuchten_n"],
        residual_water_content=soil_swc["residual_water_content"],
        water_density=1000 * ureg.kg / ureg.meter**3 if with_units else 1000.0,
        gravity=9.81 * ureg.meter / ureg.second**2 if with_units else 9.81,
    )

    assert_aaw_result(swc.compute(), with_units)


def test_guo_tracer_compute(
    hydro: HydrologicalProperties,
    guo_params: dict,
    soil_swc: dict,
    with_units: bool,
):
    """Test empirical Guo tracer AWI correlation with and without units."""
    tracer = GuoTracer(
        hydro_properties=hydro,
        AWI=guo_params,
        soil=soil_swc,
    )

    assert_aaw_result(tracer.compute(), with_units)


def test_guo_tracer_missing_keys(hydro: HydrologicalProperties, soil_swc: dict):
    """Test that missing Guo parameters raise a validation error."""
    with pytest.raises(ValueError, match="missing required keys"):
        GuoTracer(
            hydro_properties=hydro,
            AWI={"Guo": {"guo_x0": 1.0}},
            soil=soil_swc,
        )


def test_guo_tracer_missing_guo_entry(hydro: HydrologicalProperties, soil_swc: dict):
    """Test that missing 'Guo' entry raises a validation error."""
    with pytest.raises(ValueError, match="must contain a 'Guo' entry"):
        GuoTracer(
            hydro_properties=hydro,
            AWI={"NotGuo": {}},
            soil=soil_swc,
        )


def test_gssa_awi_compute(
    hydro: HydrologicalProperties, soil_d50: dict, with_units: bool
):
    """Test GSSA-based AWI calculation with and without Pint quantities."""
    gssa = GSSAAWI(
        hydro_properties=hydro,
        soil=soil_d50,
    )

    assert_aaw_result(gssa.compute(), with_units)


def test_d50_awi_compute(
    hydro: HydrologicalProperties, soil_d50: dict, with_units: bool
):
    """Test linear d50 AWI correlation with and without Pint quantities."""
    d50 = D50AWI(
        hydro_properties=hydro,
        soil=soil_d50,
    )

    assert_aaw_result(d50.compute(), with_units)


def test_nonlinear_d50_awi_compute(
    hydro: HydrologicalProperties, soil_d50: dict, with_units: bool
):
    """Test nonlinear d50 AWI correlation with and without Pint quantities."""
    nd50 = NonlinearD50AWI(
        hydro_properties=hydro,
        soil=soil_d50,
    )

    assert_aaw_result(nd50.compute(), with_units)
