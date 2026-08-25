import numpy as np
from pfas.solvers import kinetic_solver


def test_kinetic_solver_basic(grid, hydro, bc_constant,
                              adsorption_kinetic, dim_kinetic, initial):
    """
    Test kinetic solver using canonical fixtures.
    Verifies:
    - correct output shapes
    - non-negativity
    - sorbed phase increases over time
    """
    C1, C2, C_tot = kinetic_solver(
        adsorption_kinetic.total_retardation,
        dim_kinetic,
        bc_constant.C_list,
        initial,
        adsorption_kinetic.beta_s,
        adsorption_kinetic.beta,
        True,  # volume averaged
        adsorption_kinetic.sp_retardation,
        adsorption_kinetic.frac_int,
        adsorption_kinetic.Kd,
        hydro.water_content,
        1.6,   # bulk_density passed positionally
    )

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C2.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))

    assert np.all(C1 >= 0)
    assert np.all(C2 >= 0)
    assert np.all(C_tot >= 0)

    assert np.any(C2[:, -1] > C2[:, 0])
