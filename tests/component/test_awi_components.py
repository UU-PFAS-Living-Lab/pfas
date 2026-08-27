import pytest
import numpy as np

from pfas.component.awi import (
    SWCsorption,
    GuoTracer,
    GSSAAWI,
    D50AWI,
    NonlinearD50AWI,
)
from pfas.data_structure import HydrologicalProperties


class Dummy:
    """Simple object to bypass Pydantic validation."""
    pass


@pytest.fixture
def soil_swc() -> dict:
    """Soil parameters required for SWCsorption (thermodynamic AWI)."""
    return {
        "porosity": 0.4,
        "van_genuchten_alpha": 0.8,
        "van_genuchten_n": 1.6,
        "residual_water_content": 0.05,
    }


@pytest.fixture
def soil_d50() -> dict:
    """Soil parameters required for GSSA and d50-based AWI models."""
    return {
        "porosity": 0.4,
        "d50": 0.02,
    }


@pytest.fixture
def guo_params() -> dict:
    """Parameters required for the Guo tracer AWI model."""
    return {
        "Guo": {
            "guo_x0": 0.1,
            "guo_x1": 0.2,
            "guo_x2": 0.3,
        }
    }


def test_swc_sorption_compute(hydro: HydrologicalProperties, soil_swc: dict):
    """Test thermodynamic AWI calculation using van Genuchten parameters."""
    swc = Dummy()
    swc.hydro_properties = hydro
    swc.sigma0 = 0.072
    swc.scaling_factor_awi = 1.0

    swc.porosity = soil_swc["porosity"]
    swc.van_genuchten_alpha = soil_swc["van_genuchten_alpha"]
    swc.van_genuchten_n = soil_swc["van_genuchten_n"]
    swc.residual_water_content = soil_swc["residual_water_content"]

    out = SWCsorption.compute(swc)
    assert "aaw" in out
    assert out["aaw"] >= 0.0


def test_guo_tracer_compute(hydro: HydrologicalProperties, guo_params: dict, soil_swc: dict):
    """Test empirical Guo tracer AWI correlation."""
    tracer = Dummy()
    tracer.hydro_properties = hydro
    tracer.AWI = guo_params

    tracer.porosity = soil_swc["porosity"]
    tracer.van_genuchten_alpha = soil_swc["van_genuchten_alpha"]
    tracer.van_genuchten_n = soil_swc["van_genuchten_n"]
    tracer.residual_water_content = soil_swc["residual_water_content"]

    out = GuoTracer.compute(tracer)
    assert "aaw" in out
    assert out["aaw"] >= 0.0


def test_guo_tracer_missing_keys(hydro: HydrologicalProperties, soil_swc: dict):
    """Test that missing Guo parameters raise a validation error."""
    bad = {"Guo": {"guo_x0": 1.0}}

    tracer = Dummy()
    tracer.hydro_properties = hydro
    tracer.AWI = bad

    with pytest.raises(ValueError):
        GuoTracer.validate_guo_inputs(tracer)


def test_guo_tracer_missing_guo_entry(hydro: HydrologicalProperties, soil_swc: dict):
    """Test that missing 'Guo' entry raises a validation error."""
    tracer = Dummy()
    tracer.hydro_properties = hydro
    tracer.AWI = {"NotGuo": {}}

    with pytest.raises(ValueError):
        GuoTracer.validate_guo_inputs(tracer)


def test_gssa_awi_compute(hydro: HydrologicalProperties, soil_d50: dict):
    """Test GSSA-based AWI calculation using geometric surface area."""
    gssa = Dummy()
    gssa.hydro_properties = hydro
    gssa.soil = soil_d50

    out = GSSAAWI.compute(gssa)
    assert "aaw" in out
    assert out["aaw"] >= 0.0


def test_d50_awi_compute(hydro: HydrologicalProperties, soil_d50: dict):
    """Test linear d50 correlation for AWI."""
    d50 = Dummy()
    d50.hydro_properties = hydro
    d50.soil = soil_d50

    out = D50AWI.compute(d50)
    assert "aaw" in out
    assert out["aaw"] >= 0.0


def test_nonlinear_d50_awi_compute(hydro: HydrologicalProperties, soil_d50: dict):
    """Test nonlinear d50 correlation with saturation-dependent correction."""
    nd50 = Dummy()
    nd50.hydro_properties = hydro
    nd50.soil = soil_d50

    out = NonlinearD50AWI.compute(nd50)
    assert "aaw" in out
    assert out["aaw"] >= 0.0

