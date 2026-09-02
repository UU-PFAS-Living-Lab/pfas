import pytest
from pydantic import ValidationError
import numpy as np
from pfas.component.solver import KineticSolver


def test_kinetic_solver_basic(grid, hydro, bc_constant,
                              adsorption_kinetic, dim_kinetic, initial):
    """Test KineticSolver using canonical fixtures."""
    solver = KineticSolver(
        grid=grid,
        hydro_properties=hydro,
        adsorption=adsorption_kinetic,
        boundary_conditions=bc_constant,
        bulk_density=1.6,
        initial_contaminant_concentration=initial,
        volume_averaged=True,
    )

    out = solver.compute()
    C1 = out["C1"]
    C2 = out["C2"]
    C_tot = out["C_tot"]

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C2.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))

    assert np.all(C1 >= 0)
    assert np.all(C2 >= 0)
    assert np.all(C_tot >= 0)

    assert np.any(C2[:, -1] > C2[:, 0])

def test_kinetic_solver_missing_adsorption(grid, hydro, bc_constant, initial):
    """KineticSolver fails when adsorption model is missing."""
    with pytest.raises(ValidationError):
        KineticSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            bulk_density=1.6,
            volume_averaged=True,
        )


def test_kinetic_solver_wrong_initial_shape(grid, hydro, bc_constant,
                                            adsorption_kinetic):
    """KineticSolver fails when initial concentration has wrong shape."""
    bad_initial = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        sol = KineticSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=bad_initial,
            adsorption=adsorption_kinetic,
            bulk_density=1.6,
            volume_averaged=True,
        )
        sol.compute()


def test_kinetic_solver_invalid_hydro(grid, hydro, bc_constant,
                                      adsorption_kinetic, initial):
    """KineticSolver fails when hydro properties are invalid."""
    hydro.water_content = -0.1
    with pytest.raises(ValueError):
        KineticSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            adsorption=adsorption_kinetic,
            bulk_density=1.6,
            volume_averaged=True,
        )


def test_kinetic_solver_invalid_bulk_density(grid, hydro, bc_constant,
                                             adsorption_kinetic, initial):
    """KineticSolver fails when bulk density is invalid."""
    with pytest.raises(ValueError):
        KineticSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            adsorption=adsorption_kinetic,
            bulk_density=-1.0,
            volume_averaged=True,
        )


def test_kinetic_solver_invalid_bc(grid, hydro,
                                   adsorption_kinetic, initial, bc_constant):
    """KineticSolver fails when boundary conditions are invalid."""
    bc_constant.C_list = None
    with pytest.raises(TypeError):
        sol = KineticSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            adsorption=adsorption_kinetic,
            bulk_density=1.6,
            volume_averaged=True,
        )
        sol.compute()
