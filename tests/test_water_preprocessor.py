import numpy as np
import pytest
from pfas.preprocessing import WaterPreprocessor
from pfas.data_structure import HydrologicalProperties


def test_water_preprocessor_basic():
    """
    Test that WaterPreprocessor.compute() returns a valid HydrologicalProperties
    object and that the computed hydraulic properties (theta, v, D) fall within
    physically meaningful ranges.

    This verifies:
    - Correct execution of the van Genuchten relative permeability root solve
    - Theta is between residual water content and porosity
    - Velocity is positive
    - Dispersion coefficient matches v * dispersivity
    """
    wp = WaterPreprocessor(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        van_genuchten_l=None,
        residual_water_content=0.05,
    )

    out = wp.compute()
    assert "hydro_properties" in out
    hydro = out["hydro_properties"]
    assert isinstance(hydro, HydrologicalProperties)
    theta = hydro.water_content
    v = hydro.pore_velocity
    D = hydro.dispersion_coefficient

    assert theta > wp.residual_water_content
    assert theta <= wp.porosity
    assert v > 0
    assert D == pytest.approx(v * wp.dispersivity)


def test_water_preprocessor_default_l():
    """
    Test that van_genuchten_l defaults to 0.5 when None is provided.

    This verifies:
    - The custom field_validator correctly interprets None as "use default"
    """
    wp = WaterPreprocessor(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        van_genuchten_l=None,
        residual_water_content=0.05,
    )
    assert wp.van_genuchten_l == 0.5


def test_water_preprocessor_null_string_l():
    """
    Test that the string "null" is treated the same as None.

    This verifies:
    - The validator handles string-based null values (common in TOML/JSON)
    """
    wp = WaterPreprocessor(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        van_genuchten_l="null",
        residual_water_content=0.05,
    )
    assert wp.van_genuchten_l == 0.5


def test_water_preprocessor_outputs_property():
    """
    Test that the outputs property returns the correct output key.

    This verifies:
    - Consistency with the Model.compute() orchestration mechanism
    """
    wp = WaterPreprocessor(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        residual_water_content=0.05,
    )
    assert wp.outputs == ["hydro_properties"]
