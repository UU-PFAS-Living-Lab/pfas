"""Unit tests for PFAS analytical solver utility functions."""

import numpy as np
import pytest

from pfas.utils import (
    aaw_func_GSSA,
    aaw_func_d50,
    aaw_func_nonlinear_d50,
)
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------




@pytest.fixture
def vg_params():
    """van Genuchten parameters for aaw_func_thermo."""
    return dict(
        sigma0=72.0,
        poro=0.4,
        alpha=0.05,
        n=2.0,
        thr=0.05,
        ths=0.40,
        sf=1.0,
        d50=0.03,
    )


# ---------------------------------------------------------------------------
# aaw_func_GSSA
# ---------------------------------------------------------------------------

class TestAawFuncGSSA:
    def test_returns_positive_area(self, vg_params):
        Aaw = aaw_func_GSSA(th=0.20, **vg_params)
        assert Aaw > 0

    def test_area_increases_as_saturation_decreases(self, vg_params):
        """Lower water content → more air-water interface."""
        Aaw_wet = aaw_func_GSSA(th=0.35, **vg_params)
        Aaw_dry = aaw_func_GSSA(th=0.10, **vg_params)
        assert Aaw_dry > Aaw_wet

    def test_scaling_factor_proportional(self, vg_params):
        Aaw_sf1 = aaw_func_thermo(th=0.20, **vg_params)
        Aaw_sf2 = aaw_func_thermo(th=0.20, **{**vg_params, "sf": 2.0})
        assert Aaw_sf2 == pytest.approx(2 * Aaw_sf1, rel=1e-6)

    def test_returns_float(self, vg_params):
        Aaw = aaw_func_thermo(th=0.20, **vg_params)
        assert isinstance(float(Aaw), float)

    def test_finite_output(self, vg_params):
        Aaw = aaw_func_thermo(th=0.20, **vg_params)
        assert np.isfinite(Aaw)


# ---------------------------------------------------------------------------
# aaw_func_d50
# ---------------------------------------------------------------------------

class TestAawFuncTracer:
    def test_scalar_input(self):
        Aaw = aaw_func_tracer(sw=0.5, x2=-10.0, x1=8.0, x0=1.0)
        expected = -10.0 * 0.5**2 + 8.0 * 0.5 + 1.0
        assert Aaw == pytest.approx(expected)

    def test_array_input(self):
        sw = np.array([0.2, 0.5, 0.8])
        Aaw = aaw_func_tracer(sw=sw, x2=-10.0, x1=8.0, x0=1.0)
        assert Aaw.shape == (3,)

    def test_zero_coefficients(self):
        """All-zero polynomial should return zero."""
        assert aaw_func_tracer(0.5, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_constant_polynomial(self):
        """x2=0, x1=0 → result equals x0 for any sw."""
        assert aaw_func_tracer(0.3, 0.0, 0.0, 5.0) == pytest.approx(5.0)
        assert aaw_func_tracer(0.9, 0.0, 0.0, 5.0) == pytest.approx(5.0)

    def test_linear_polynomial(self):
        assert aaw_func_tracer(sw=2.0, x2=0.0, x1=3.0, x0=1.0) == pytest.approx(7.0)

# ---------------------------------------------------------------------------
# aaw_func_nonlinear_d50
# ---------------------------------------------------------------------------
