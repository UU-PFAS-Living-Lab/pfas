import numpy as np
from pfas.solvers import equilibrium_solver


def test_equilibrium_solver_basic(grid, hydro, bc_constant,
                                  adsorption_equilibrium, dim_equilibrium, initial):
    """
    Test equilibrium solver using canonical fixtures.
    Verifies:
    - correct output shapes
    - non-negativity
    - initial condition preservation
    - concentration increase over time
    - C_tot = theta * C1
    """
    C1, C_tot = equilibrium_solver(
        adsorption_equilibrium.total_retardation,
        dim_equilibrium,
        bc_constant.C_list,
        initial,
        hydro.water_content,
    )

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))

    assert np.all(C1 >= 0)
    assert np.all(C_tot >= 0)

    assert np.allclose(C1[:, 0], initial)
    assert np.any(C1[:, -1] > C1[:, 0])

    assert np.allclose(C_tot, C1 * hydro.water_content)
