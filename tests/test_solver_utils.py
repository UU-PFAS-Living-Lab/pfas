"""Unit tests for functions in :mod:`pfas.solver_utils`.

"""

import numpy as np
import pytest

from pfas.solver_utils import (
    compute_dimensionless_params,
    _FT,
    _bvp_neq,
    DimensionlessParams,
)
from pfas.analytical_soln import (
    SimulationGrid,
    BoundaryConditions,
    HydrologicalProperties,
    Adsorption,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture

def simple_grid():
    """A tiny spatial/temporal grid for tests."""
    depth = np.linspace(0, 1.0, 5)
    time = np.linspace(0, 10.0, 5)
    return SimulationGrid(depth=depth, time=time)


@pytest.fixture

def simple_hydro():
    """Non-zero hydrological properties used by tests."""
    return HydrologicalProperties(
        water_content=0.4,
        pore_velocity=1e-6,
        dispersion_coefficient=1e-7,
    )


@pytest.fixture

def simple_boundary():
    """Boundary conditions with a single pulse from 0 to 5 seconds."""
    return BoundaryConditions(pulse_intervals=[(0.0, 5.0)], contaminant_release_rate=1.0)


@pytest.fixture

def simple_adsorption():
    """Adsorption parameters suitable for kinetic tests."""
    return Adsorption(
        Kd=0.1,
        rate_const=1e-4,
        frac_int=0.5,
        sp_retardation=2.0,
        awi_retardation=0.5,
    )


# ---------------------------------------------------------------------------
# compute_dimensionless_params
# ---------------------------------------------------------------------------

class TestComputeDimensionlessParams:
    def test_basic_conversion(self, simple_grid, simple_boundary, simple_hydro):
        dim = compute_dimensionless_params(
            simple_grid,
            simple_hydro,
            pulse_intervals=simple_boundary.pulse_intervals,
            adsorption=None,
            kinetic=False,
        )
        # Check returned type and attributes
        assert isinstance(dim, DimensionlessParams)
        assert dim.Z.shape == simple_grid.depth.shape
        assert dim.T.shape == simple_grid.time.shape
        # pulses should be scaled by v/L = 1e-6/1 = 1e-6
        assert dim.pulses[0][0] == pytest.approx(0.0)
        assert dim.pulses[0][1] == pytest.approx(5.0 * simple_hydro.pore_velocity / simple_grid.depth[-1])

    def test_infinite_interval(self, simple_grid, simple_boundary, simple_hydro):
        b = BoundaryConditions(pulse_intervals=[(0.0, np.inf)], contaminant_release_rate=1.0)
        dim = compute_dimensionless_params(
            simple_grid,
            simple_hydro,
            pulse_intervals=b.pulse_intervals,
            adsorption=None,
            kinetic=False,
        )
        assert dim.pulses[0][1] == np.inf

    @pytest.mark.parametrize("velocity,dispersion", [(0.0, 1e-7), (1e-6, 0.0)])
    def test_zero_velocity_or_dispersion_raises(self, simple_grid, simple_boundary, velocity, dispersion):
        hydro = HydrologicalProperties(
            water_content=0.4,
            pore_velocity=velocity,
            dispersion_coefficient=dispersion,
        )
        with pytest.raises(ValueError):
            compute_dimensionless_params(
                simple_grid,
                hydro,
                pulse_intervals=simple_boundary.pulse_intervals,
                adsorption=None,
                kinetic=False,
            )

    def test_missing_adsorption_for_kinetic(self, simple_grid, simple_boundary, simple_hydro):
        with pytest.raises(ValueError):
            compute_dimensionless_params(
                simple_grid,
                simple_hydro,
                pulse_intervals=simple_boundary.pulse_intervals,
                adsorption=None,
                kinetic=True,
            )

    def test_invalid_interval(self, simple_grid, simple_boundary, simple_hydro):
        bad_bc = BoundaryConditions(pulse_intervals=[(5.0, 1.0)], contaminant_release_rate=1.0)
        with pytest.raises(ValueError):
            compute_dimensionless_params(
                simple_grid,
                simple_hydro,
                pulse_intervals=bad_bc.pulse_intervals,
                adsorption=None,
                kinetic=False,
            )


# ---------------------------------------------------------------------------
# kernels: _FT and _bvp_neq
# ---------------------------------------------------------------------------

class TestInternalKernels:
    def test_ft_flux_vs_volume(self):
        tau = np.linspace(0.1, 1.0, 5)
        Z, P, R, beta = 0.5, 10.0, 2.0, 0.6
        vol = _FT(tau, Z, P, R, beta, True)
        flux = _FT(tau, Z, P, R, beta, False)
        assert np.all(vol >= 0)
        assert np.all(flux >= 0)
        assert not np.allclose(vol, flux)

    def test_bvp_neq_flag_effect(self):
        # choose parameters where kernel difference is obvious
        vals = _bvp_neq(
            Z=0.5,
            T=0.7,
            omega=0.2,
            beta_s=0.6,
            beta=0.4,
            P=5.0,
            R=3.0,
            R_s=1.0,
            volume_averaged=True,
        )
        vals2 = _bvp_neq(
            Z=0.5,
            T=0.7,
            omega=0.2,
            beta_s=0.6,
            beta=0.4,
            P=5.0,
            R=3.0,
            R_s=1.0,
            volume_averaged=False,
        )
        assert vals != vals2
        # both outputs should be floats
        assert isinstance(vals[0], float)
        assert isinstance(vals2[1], float)

