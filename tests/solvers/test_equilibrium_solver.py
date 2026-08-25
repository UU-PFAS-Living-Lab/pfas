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

    assert np.allclose(C1[:, 0], initial)
    assert np.any(C1[:, -1] > C1[:, 0])

    assert np.allclose(C_tot, C1 * hydro.water_content)
