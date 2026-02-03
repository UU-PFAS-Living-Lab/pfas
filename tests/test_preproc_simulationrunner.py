import pytest
import numpy as np

from pfas.preprocessing import SimulationRunner
from pfas.analytical_soln import Adsorption

def test_simulation_runner_computation(valid_simulation_runner):
    """Test that the compute method returns expected results."""
    result = valid_simulation_runner.compute()

    # Ensure the keys are in the result
    assert "C1" in result
    assert "C2" in result
    assert "C_tot" in result

    assert isinstance(result["C1"], np.ndarray)
    #NOTE: urrently C2 is None -> False
    assert isinstance(result["C2"], np.ndarray)
    assert isinstance(result["C_tot"], np.ndarray)

    assert len(result["C1"]) == len(valid_simulation_runner.grid.depth)
    assert len(result["C2"]) == len(valid_simulation_runner.grid.depth)
    assert len(result["C_tot"]) == len(valid_simulation_runner.grid.depth)

def test_kinetic_sorption_flag(valid_simulation_runner):
    """Test that the kinetic_sorption flag affects the computation."""
    # First with kinetic sorption as False
    result_equilibrium = valid_simulation_runner.compute()

    # Now with kinetic sorption as True
    valid_simulation_runner.kinetic_sorption = True
    result_kinetic = valid_simulation_runner.compute()

    # If the kinetic sorption changes the results, there should be a difference
    assert not np.array_equal(result_equilibrium["C1"], result_kinetic["C1"])
    assert not np.array_equal(result_equilibrium["C2"], result_kinetic["C2"])
    assert not np.array_equal(result_equilibrium["C_tot"], result_kinetic["C_tot"])


def test_volume_averaging_flag(valid_simulation_runner):
    """Test that the volume_averaged flag affects the computation."""
    # First with volume_averaged as False
    result_non_avg = valid_simulation_runner.compute()

    # Now with volume_averaged as True
    valid_simulation_runner.volume_averaged = True
    result_avg = valid_simulation_runner.compute()

    # If the volume averaging flag changes the results, there should be a difference
    assert not np.array_equal(result_non_avg["C1"], result_avg["C1"])
    #NOTE: Again C2 gives problems
    assert not np.array_equal(result_non_avg["C2"], result_avg["C2"])
    assert not np.array_equal(result_non_avg["C_tot"], result_avg["C_tot"])

def test_invalid_bulk_density():
    """Test that invalid bulk_density raises a validation error."""
    with pytest.raises(ValueError):
        SimulationRunner(
            grid=None,
            bulk_density=-1600.0,  # invalid (should be positive)
            boundary_conditions={},
            hydro_properties={},
            adsorption={},
            kinetic_sorption=False,
            volume_averaged=False,
        )


def test_extra_fields_forbidden(valid_simulation_runner):
    """Test that extra fields are forbidden by Pydantic."""
    with pytest.raises(ValueError):
        SimulationRunner(
            grid=valid_simulation_runner.grid,
            bulk_density=1600.0,
            boundary_conditions=valid_simulation_runner.boundary_conditions,
            hydro_properties=valid_simulation_runner.hydro_properties,
            adsorption=valid_simulation_runner.adsorption,
            kinetic_sorption=False,
            volume_averaged=False,
            extra_field=123,  # This should raise a ValueError (Pydantic's `extra='forbid'`)
        )

def test_boundary_conditions(valid_simulation_runner):
    """Test that the boundary conditions are correctly passed to the simulation."""
    result = valid_simulation_runner.compute()

    # For now, let's check if boundary_conditions are correctly included in the input
    assert "boundary_conditions" in valid_simulation_runner.__dict__
