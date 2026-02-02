from pathlib import Path

import pytest

from pfas.configuration import read_toml
from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor

@pytest.fixture(scope="session")
def configuration():
    config_path = Path("examples", "data", "config.toml")
    return read_toml(config_path)

@pytest.fixture
def valid_water_preprocessor():
    """A valid WaterPreprocessor instance with realistic parameters."""
    return WaterPreprocessor(
        average_infiltration_rate=1e-8,
        hydraulic_conductivity=1e-5,
        porosity=0.4,
        dispersivity=0.1,
        van_genuchten_n=2.0,
        init_sat=0.5,
        residual_water_content=0.05,
    )

@pytest.fixture
def result_water(valid_water_preprocessor):
    """Computed result dictionary."""
    return valid_water_preprocessor.compute()

@pytest.fixture
def valid_boundary_preprocessor():
    """A valid BoundaryPreprocessor instance."""
    return BoundaryPreprocessor(
        average_infiltration_rate=1e-8,
        solute_concentration_influx=100.0,  # mg/L
        pulse_duration=3600.0,               # s
    )

@pytest.fixture
def result_boundary(valid_boundary_preprocessor):
    """Computed result dictionary."""
    return valid_boundary_preprocessor.compute()
