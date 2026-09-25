import pytest
from pydantic import ValidationError
import numpy as np
from pfas.component.solver import EquilibriumSolver


def test_equilibrium_solver_basic(grid, hydro, bc_constant,
                                  adsorption_equilibrium, dim_equilibrium, initial):
    """Test EquilibriumSolver using canonical fixtures."""
    solver = EquilibriumSolver(
        grid=grid,
        hydro_properties=hydro,
        adsorption=adsorption_equilibrium,
        boundary_conditions=bc_constant,
        initial_contaminant_concentration=initial,
        bc="resident",
    )

    out = solver.compute()
    C1 = out["C1"]
    C_tot = out["C_tot"]

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))

    assert np.all(C1 >= 0)
    assert np.all(C_tot >= 0)

    #assert np.allclose(C1[:, 0], initial)
    assert np.any(C1[:, -1] > C1[:, 0])

## Negative tests
def test_equilibrium_solver_missing_adsorption(grid, hydro, bc_constant, initial):
    """EquilibriumSolver fails when adsorption model is missing."""
    with pytest.raises(ValidationError):
        EquilibriumSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            bc="resident",
        )


def test_equilibrium_solver_wrong_initial_shape(grid, hydro, bc_constant,
                                                adsorption_equilibrium):
    """EquilibriumSolver fails when initial concentration has wrong shape."""
    bad_initial = np.array([1.0, 2.0])  # wrong shape
    with pytest.raises(ValueError):
        sol = EquilibriumSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=bad_initial,
            adsorption=adsorption_equilibrium,
            bc="resident",
        )
        sol.compute()


def test_equilibrium_solver_invalid_hydro(grid, hydro, bc_constant,
                                          adsorption_equilibrium, initial):
    """EquilibriumSolver fails when hydro properties are invalid."""
    hydro.water_content = -0.1
    with pytest.raises(ValueError):
        EquilibriumSolver(
            grid=grid,
            hydro_properties=hydro,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            adsorption=adsorption_equilibrium,
            bc="resident",
        )


def test_equilibrium_solver_invalid_bc(grid, hydro,
                                       adsorption_equilibrium, initial, bc_constant):
    """EquilibriumSolver fails when boundary conditions are invalid."""
    bc_constant.C_list=None  # wrong shape
    with pytest.raises(TypeError):
        sol = EquilibriumSolver(
                grid=grid,
                hydro_properties=hydro,
                boundary_conditions=bc_constant,
                initial_contaminant_concentration=initial,
                adsorption=adsorption_equilibrium,
                bc="resident",
            )
        sol.compute()

