import numpy as np
import pytest

from pfas.preprocessing import SimulationRunner


def test_kinetic_sorption_flag(make_simulation_runner):
    """Test that the kinetic_sorption flag affects the computation."""
    runner_equilibrium = make_simulation_runner(kinetic_sorption=False)
    result_equilibrium = runner_equilibrium.compute()

    runner_kinetic = make_simulation_runner(kinetic_sorption=True)
    result_kinetic = runner_kinetic.compute()

    assert not np.array_equal(result_equilibrium["C1"], result_kinetic["C1"])
    assert not np.array_equal(result_equilibrium["C2"], result_kinetic["C2"])
    assert not np.array_equal(result_equilibrium["C_tot"], result_kinetic["C_tot"])


def test_volume_averaging_flag(make_simulation_runner):
    """Test that the volume_averaged flag affects the computation."""
    runner_non_avg = make_simulation_runner(volume_averaged=False, kinetic_sorption=True)
    result_non_avg = runner_non_avg.compute()

    runner_avg = make_simulation_runner(volume_averaged=True, kinetic_sorption=True)
    result_avg = runner_avg.compute()

    assert not np.array_equal(result_non_avg["C1"], result_avg["C1"])
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
            sorption_solid={},
            awi_retardation=0.0,
            kinetic_sorption=False,
            volume_averaged=False,
        )


def test_extra_fields_forbidden(make_simulation_runner):
    """Test that extra fields are forbidden by Pydantic."""
    runner = make_simulation_runner()

    with pytest.raises(ValueError):
        SimulationRunner(
            grid=runner.grid,
            bulk_density=1600.0,
            boundary_conditions=runner.boundary_conditions,
            hydro_properties=runner.hydro_properties,
            sorption_solid=runner.sorption_solid,
            awi_retardation=runner.awi_retardation,
            kinetic_sorption=False,
            volume_averaged=False,
            extra_field=123,  # This should raise a ValueError (Pydantic's `extra='forbid'`)
        )


def test_boundary_conditions(make_simulation_runner):
    """Test that the boundary conditions are correctly passed to the simulation."""
    runner = make_simulation_runner()
    result = runner.compute()

    assert "boundary_conditions" in runner.__dict__


@pytest.mark.parametrize(
    "kinetic_sorption,volume_averaged",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_factory_fixture_all_combinations(
    make_simulation_runner, kinetic_sorption, volume_averaged
):
    """Test factory fixture with all combinations of kinetic_sorption and volume_averaged."""
    runner = make_simulation_runner(
        kinetic_sorption=kinetic_sorption, volume_averaged=volume_averaged
    )

    assert runner is not None
    assert runner.kinetic_sorption is kinetic_sorption
    assert runner.volume_averaged is volume_averaged
    assert runner.bulk_density == 1600.0


def test_factory_fixture_kinetic_sorption_true(make_simulation_runner):
    """Test factory fixture creates runner with kinetic_sorption=True."""
    runner = make_simulation_runner(kinetic_sorption=True, volume_averaged=False)

    assert runner.kinetic_sorption is True
    assert runner.volume_averaged is False

    result = runner.compute()
    assert "C1" in result
    assert "C2" in result
    assert "C_tot" in result

    # For kinetic solver: both C1 and C2 should be arrays
    assert isinstance(result["C1"], np.ndarray)
    assert isinstance(result["C2"], np.ndarray)


def test_factory_fixture_volume_averaged_true(make_simulation_runner):
    """Test factory fixture creates runner with volume_averaged=True."""
    runner = make_simulation_runner(kinetic_sorption=False, volume_averaged=True)

    assert runner.kinetic_sorption is False
    assert runner.volume_averaged is True

    result = runner.compute()
    assert "C1" in result
    assert "C2" in result
    assert "C_tot" in result

<<<<<<< pyfas-data
    # For equilibrium solver: C1 should be array, C2 should be None
=======
>>>>>>> main
    assert isinstance(result["C1"], np.ndarray)
    assert result["C2"] is None


def test_factory_fixture_both_true(make_simulation_runner):
    """Test factory fixture creates runner with both flags True."""
    runner = make_simulation_runner(kinetic_sorption=True, volume_averaged=True)

    assert runner.kinetic_sorption is True
    assert runner.volume_averaged is True

    result = runner.compute()
    assert "C1" in result
    assert "C2" in result
    assert "C_tot" in result

<<<<<<< pyfas-data
    # For kinetic solver: both C1 and C2 should be arrays
=======
>>>>>>> main
    assert isinstance(result["C1"], np.ndarray)
    assert isinstance(result["C2"], np.ndarray)


def test_factory_fixture_both_false(make_simulation_runner):
    """Test factory fixture creates runner with both flags False (default)."""
    runner = make_simulation_runner()

    assert runner.kinetic_sorption is False
    assert runner.volume_averaged is False

    result = runner.compute()
    assert "C1" in result
    assert "C2" in result
    assert "C_tot" in result

<<<<<<< pyfas-data
    # For equilibrium solver: C1 should be array, C2 should be None
=======
>>>>>>> main
    assert isinstance(result["C1"], np.ndarray)
    assert result["C2"] is None
