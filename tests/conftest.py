from pathlib import Path

import numpy as np
import pytest

from pfas.analytical_soln import SimulationGrid
from pfas.configuration import read_toml
from pfas.preprocessing import (
    BoundaryPreprocessor,
    GridGenerator,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    SWCAdsorptionPreprocessor,
    WaterPreprocessor,
)


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

@pytest.fixture
def valid_grid_generator():
    return GridGenerator(
        domain_length=10.0,
        spatial_resolution=1.0,
        time_resolution=2.0,
        time_total=10.0,
    )

@pytest.fixture
def sorption_solid_linear():
    return {
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 0.8,
        },
    }

@pytest.fixture
def valid_sp_retardation_preprocessor(result_water, sorption_solid_linear):
    return SpRetardationPreprocessor(
        sorption_solid=sorption_solid_linear,
        bulk_density=1600.0,
        hydro_properties=result_water["hydro_properties"],
    )

@pytest.fixture
def soil_params(valid_water_preprocessor):
    """Soil parameters consistent with WaterPreprocessor."""
    return {
        "porosity": valid_water_preprocessor.porosity,
        "van_genuchten_alpha": 1.0,
        "van_genuchten_n": valid_water_preprocessor.van_genuchten_n,
        "residual_water_content": valid_water_preprocessor.residual_water_content,
    }


@pytest.fixture
def awi_swc_based():
    return {
        "AWI_type": "SWC-based",
    }


@pytest.fixture
def awi_guo():
    return {
        "AWI_type": "Guo",
        "Guo": {
            "guo_x0": 0.1,
            "guo_x1": 0.5,
            "guo_x2": 2.0,
        },
    }


@pytest.fixture
def valid_swc_adsorption_swc(
    result_water,
    soil_params,
    awi_swc_based,
):
    return SWCAdsorptionPreprocessor(
        hydro_properties=result_water["hydro_properties"],
        scaling_factor_awi=1.0,
        AWI=awi_swc_based,
        soil=soil_params,
    )


@pytest.fixture
def valid_swc_adsorption_guo(
    result_water,
    soil_params,
    awi_guo,
):
    return SWCAdsorptionPreprocessor(
        hydro_properties=result_water["hydro_properties"],
        scaling_factor_awi=1.0,
        AWI=awi_guo,
        soil=soil_params,
    )

@pytest.fixture
def valid_sorption_solid_awi():
    """Solid-phase sorption parameters for AWI sorption."""
    return {
        "rate_const": 1.0e-4,
        "fraction_instantaneous": 0.8,
    }

@pytest.fixture
def valid_sorption_kawi_direct_input(
    result_water,
    valid_sp_retardation_preprocessor,
    valid_swc_adsorption_swc,
):
    aaw = valid_swc_adsorption_swc.compute()["aaw"]
    sp_ret = valid_sp_retardation_preprocessor.compute()["sp_retardation"]

    return SorptionKawiDirectInput(
        kaw=0.5,
        hydro_properties=result_water["hydro_properties"],
        Kd=0.8,
        aaw=aaw,
        sorption_solid={},      # optional fields tested elsewhere
        sp_retardation=sp_ret,
    )

@pytest.fixture
def valid_simulation_runner(
    valid_grid_generator,
    valid_boundary_preprocessor,
    result_water,
    valid_sorption_kawi_direct_input,
):
    """A valid SimulationRunner instance."""
    # Generate depth and time arrays based on grid generator parameters
    depth = np.linspace(0, valid_grid_generator.domain_length, int(valid_grid_generator.domain_length / valid_grid_generator.spatial_resolution) + 1)
    time = np.linspace(0, valid_grid_generator.time_total, int(valid_grid_generator.time_total / valid_grid_generator.time_resolution) + 1)

    # Create SimulationGrid instance
    grid = SimulationGrid(depth=depth, time=time)

    # Initialize BoundaryConditions properly
    boundary_conditions_data = valid_boundary_preprocessor.compute()
    boundary_conditions = boundary_conditions_data["boundary_conditions"]

    return SimulationRunner(
        grid=grid,
        bulk_density=1600.0,  # example value in kg/m³
        boundary_conditions=boundary_conditions,
        hydro_properties=result_water["hydro_properties"],
        adsorption=valid_sorption_kawi_direct_input.compute()["adsorption"],
        kinetic_sorption=False,
        volume_averaged=False,
    )

@pytest.fixture
def make_simulation_runner(
    valid_grid_generator,
    valid_boundary_preprocessor,
    result_water,
    valid_sorption_kawi_direct_input,
):
    """Factory fixture to create SimulationRunner instances with different parameters."""

    def _make_simulation_runner(kinetic_sorption=False, volume_averaged=False):
        # Generate depth and time arrays based on grid generator parameters
        depth = np.linspace(
            0,
            valid_grid_generator.domain_length,
            int(valid_grid_generator.domain_length / valid_grid_generator.spatial_resolution) + 1,
        )
        time = np.linspace(
            0,
            valid_grid_generator.time_total,
            int(valid_grid_generator.time_total / valid_grid_generator.time_resolution) + 1,
        )

        # Create SimulationGrid instance
        grid = SimulationGrid(depth=depth, time=time)

        # Initialize BoundaryConditions properly
        boundary_conditions_data = valid_boundary_preprocessor.compute()
        boundary_conditions = boundary_conditions_data["boundary_conditions"]

        return SimulationRunner(
            grid=grid,
            bulk_density=1600.0,  # example value in kg/m³
            boundary_conditions=boundary_conditions,
            hydro_properties=result_water["hydro_properties"],
            adsorption=valid_sorption_kawi_direct_input.compute()["adsorption"],
            kinetic_sorption=kinetic_sorption,
            volume_averaged=volume_averaged,
        )

    return _make_simulation_runner

