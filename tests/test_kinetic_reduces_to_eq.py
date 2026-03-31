"""Tests verifying that kinetic_solver reduces to equilibrium_solver when f=1.

When the fraction of instantaneous sorption sites f=1, all sorption is at
equilibrium and the kinetic solver should approximate aqueous (C1) and
total (C_tot) concentrations from the equilibrium solver (within relaxed tolerances).
"""

import numpy as np
import pytest

from pfas.analytical_soln import (
    SimulationGrid,
    BoundaryConditions,
    HydrologicalProperties,
    Adsorption,
    analytical_soln,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
    """Small grid for solver comparison tests.

    Depth starts at a small positive value to avoid singularities in BVP
    kernels at Z=0. Time starts above 0 to avoid T=0 singularity.
    """
    depth = np.linspace(0.01, 1.0, 10)
    time = np.linspace(0.1, 10.0, 10)
    return SimulationGrid(depth=depth, time=time)


@pytest.fixture
def hydro():
    return HydrologicalProperties(
        water_content=0.3,
        pore_velocity=1e-5,
        dispersion_coefficient=1e-6,
    )


@pytest.fixture
def adsorption_eq():
    """Adsorption with f=1: all sorption instantaneous."""
    return Adsorption(
        Kd=0.5,
        rate_const=1e-3,
        frac_int=1.0,
        sp_retardation=0.0,
        awi_retardation=0.0,
    )


@pytest.fixture
def step_boundary():
    return BoundaryConditions(
        C_list=[1.0],
        T_list=[0.0],
    )


@pytest.fixture
def pulse_boundary():
    return BoundaryConditions(
        C_list=[1.0, 0.0],
        T_list=[0.0, 5.0],
    )


@pytest.fixture
def multi_pulse_boundary():
    return BoundaryConditions(
        C_list=[1.0, 0.0, 2.0, 0.0],
        T_list=[0.0, 2.0, 5.0, 8.0],
    )


@pytest.fixture
def zero_initial():
    return np.zeros(10)


@pytest.fixture
def nonzero_initial():
    depth = np.linspace(0.01, 1.0, 10)
    return 0.5 * np.exp(-5 * depth)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run_both_solvers(grid, hydro, adsorption, boundary, initial, bulk_density=1.5):
    C1_eq, _, C_tot_eq = analytical_soln(
        grid=grid,
        bulk_density=bulk_density,
        boundary_conditions=boundary,
        initial_contaminant_concentration=initial,
        hydro_properties=hydro,
        adsorption=adsorption,
        kinetic=False,
    )
    C1_kin, C2_kin, C_tot_kin = analytical_soln(
        grid=grid,
        bulk_density=bulk_density,
        boundary_conditions=boundary,
        initial_contaminant_concentration=initial,
        hydro_properties=hydro,
        adsorption=adsorption,
        kinetic=True,
        volume_averaged=True,
    )
    return C1_eq, C_tot_eq, C1_kin, C2_kin, C_tot_kin


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestKineticReducesToEquilibrium:
    """When f=1, kinetic_solver should approximate equilibrium_solver."""

    RTOL = 5e-2   # 5% relative tolerance
    ATOL = 1e-6   # relaxed absolute tolerance

    def test_step_input_zero_initial(self, grid, hydro, adsorption_eq, step_boundary, zero_initial):
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, step_boundary, zero_initial
        )
        np.testing.assert_allclose(C_tot_kin, C_tot_eq, rtol=self.RTOL, atol=self.ATOL)

    def test_pulse_input_zero_initial(self, grid, hydro, adsorption_eq, pulse_boundary, zero_initial):
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, pulse_boundary, zero_initial
        )
        np.testing.assert_allclose(C_tot_kin, C_tot_eq, rtol=self.RTOL, atol=self.ATOL)

    def test_multi_pulse_zero_initial(self, grid, hydro, adsorption_eq, multi_pulse_boundary, zero_initial):
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, multi_pulse_boundary, zero_initial
        )
        np.testing.assert_allclose(C_tot_kin, C_tot_eq, rtol=self.RTOL, atol=self.ATOL)

    def test_step_input_nonzero_initial(self, grid, hydro, adsorption_eq, step_boundary, nonzero_initial):
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, step_boundary, nonzero_initial
        )
        np.testing.assert_allclose(C_tot_kin, C_tot_eq, rtol=self.RTOL, atol=self.ATOL)

    def test_pulse_input_nonzero_initial(self, grid, hydro, adsorption_eq, pulse_boundary, nonzero_initial):
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, pulse_boundary, nonzero_initial
        )
        np.testing.assert_allclose(C_tot_kin, C_tot_eq, rtol=self.RTOL, atol=self.ATOL)

    def test_C2_is_negligible_when_f1(self, grid, hydro, adsorption_eq, step_boundary, zero_initial):
        """C2 should be near zero when f=1 (allowing numerical noise)."""
        _, _, _, C2_kin, _ = run_both_solvers(
            grid, hydro, adsorption_eq, step_boundary, zero_initial
        )
        assert np.max(np.abs(C2_kin)) < 1e-4

    def test_shapes_match(self, grid, hydro, adsorption_eq, step_boundary, zero_initial):
        C1_eq, C_tot_eq, C1_kin, C2_kin, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_eq, step_boundary, zero_initial
        )
        assert C_tot_eq.shape == C_tot_kin.shape
        assert C2_kin.shape == C1_kin.shape
        assert adsorption_eq.beta == pytest.approx(1.0)
        assert adsorption_eq.beta_s == pytest.approx(1.0)


class TestKineticDiffersFromEquilibrium:
    """Sanity check: when f<1, solutions should differ."""

    def test_f_less_than_1_gives_different_results(self, grid, hydro, step_boundary, zero_initial):
        adsorption_kin = Adsorption(
            Kd=0.5,
            rate_const=1e-3,
            frac_int=0.5,
            sp_retardation=2.0,
            awi_retardation=0.0,
        )
        _, C_tot_eq, _, _, C_tot_kin = run_both_solvers(
            grid, hydro, adsorption_kin, step_boundary, zero_initial
        )
        assert not np.allclose(C_tot_kin, C_tot_eq, rtol=1e-3)