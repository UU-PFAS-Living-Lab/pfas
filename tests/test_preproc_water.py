import pytest
from pydantic import ValidationError

from pfas.preprocessing import WaterPreprocessor, HydrologicalProperties

def test_outputs_property(valid_water_preprocessor):
    assert valid_water_preprocessor.outputs == ["hydro_properties"]

def test_compute_returns_expected_key(result_water):
    assert "hydro_properties" in result_water

def test_compute_returns_hydrological_properties(result_water):
    hydro = result_water["hydro_properties"]
    assert isinstance(hydro, HydrologicalProperties)

# Test pydantic calidation

@pytest.mark.parametrize(
    "field,value",
    [
        ("average_infiltration_rate", -1e-8),
        ("hydraulic_conductivity", 0.0),
        ("dispersivity", -0.1),
        ("van_genuchten_n", 0.0),
    ],
)
def test_positive_constraints(field, value):
    kwargs = dict(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        init_sat=0.5,
        residual_water_content=0.05,
    )
    kwargs[field] = value

    with pytest.raises(ValidationError):
        WaterPreprocessor(**kwargs)

@pytest.mark.parametrize(
    "field,value",
    [
        ("porosity", 1.5),
        ("init_sat", -0.1),
        ("residual_water_content", 2.0),
    ],
)
def test_unit_interval_constraints(field, value):
    kwargs = dict(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        init_sat=0.5,
        residual_water_content=0.05,
    )
    kwargs[field] = value

    with pytest.raises(ValidationError):
        WaterPreprocessor(**kwargs)
