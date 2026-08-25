import pytest
import numpy as np
from pfas.analytical_soln import analytical_soln, HydrologicalProperties


def test_analytical_equilibrium(grid, hydro, bc_constant,
                                adsorption_equilibrium, initial):
    """
    Test analytical_soln() with kinetic=False.
    Verifies:
    - correct output shapes
    - C2 is None
    - non-negativity
    - initial condition preserved
    """
    C1, C2, C_tot = analytical_soln(
        grid=grid,
        bulk_density=1.6,
        boundary_conditions=bc_constant,
        initial_contaminant_concentration=initial,
        hydro_properties=hydro,
        adsorption=adsorption_equilibrium,
        kinetic=False,
        volume_averaged=True,
    )

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))
    assert C2 is None

    assert np.all(C1 >= 0)
    assert np.all(C_tot >= 0)

    # initial condition preserved
    assert np.allclose(C1[:, 0], initial)

def test_analytical_kinetic(grid, hydro, bc_constant,
                            adsorption_kinetic, initial):
    """
    Test analytical_soln() with kinetic=True.
    Verifies:
    - correct output shapes
    - C2 is not None
    - sorbed phase increases over time
    """
    C1, C2, C_tot = analytical_soln(
        grid=grid,
        bulk_density=1.6,
        boundary_conditions=bc_constant,
        initial_contaminant_concentration=initial,
        hydro_properties=hydro,
        adsorption=adsorption_kinetic,
        kinetic=True,
        volume_averaged=True,
    )

    assert C1.shape == (len(grid.depth), len(grid.time))
    assert C2.shape == (len(grid.depth), len(grid.time))
    assert C_tot.shape == (len(grid.depth), len(grid.time))

    assert np.all(C1 >= 0)
    assert np.all(C2 >= 0)
    assert np.all(C_tot >= 0)

    # sorbed phase increases over time
    assert np.any(C2[:, -1] > C2[:, 0])


def test_analytical_soln_zero_velocity(grid, bc_constant,
                                       adsorption_equilibrium, initial):
    hydro_bad = HydrologicalProperties(
        water_content=0.3,
        pore_velocity=0.0,            # invalid
        dispersion_coefficient=1e-6,
    )

    with pytest.raises(ValueError):
        analytical_soln(
            grid=grid,
            bulk_density=1.6,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            hydro_properties=hydro_bad,
            adsorption=adsorption_equilibrium,
            kinetic=False,
        )


def test_analytical_soln_zero_dispersion(grid, bc_constant,
                                         adsorption_equilibrium, initial):
    hydro_bad = HydrologicalProperties(
        water_content=0.3,
        pore_velocity=1e-4,
        dispersion_coefficient=0.0,   # invalid
    )

    with pytest.raises(ValueError):
        analytical_soln(
            grid=grid,
            bulk_density=1.6,
            boundary_conditions=bc_constant,
            initial_contaminant_concentration=initial,
            hydro_properties=hydro_bad,
            adsorption=adsorption_equilibrium,
            kinetic=False,
        )

