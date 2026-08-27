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

