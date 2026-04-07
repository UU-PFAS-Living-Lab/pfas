"""Unit tests for PFAS analytical solver utility functions."""

import numpy as np
import pytest

from pfas.utils import (
    aaw_func_thermo,
    aaw_func_tracer,
    k_oc_fabregat_palau2021,
    k_sc_fabregat_palau2021,
    kd_fabregat_palau,
    kd_freundlich,
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
    )


# ---------------------------------------------------------------------------
# aaw_func_thermo
# ---------------------------------------------------------------------------

class TestAawFuncThermo:
    def test_returns_positive_area(self, vg_params):
        Aaw = aaw_func_thermo(th=0.20, **vg_params)
        assert Aaw > 0

    def test_area_increases_as_saturation_decreases(self, vg_params):
        """Lower water content → more air-water interface."""
        Aaw_wet = aaw_func_thermo(th=0.35, **vg_params)
        Aaw_dry = aaw_func_thermo(th=0.10, **vg_params)
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
# aaw_func_tracer
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
# k_oc_fabregat_palau2021
# ---------------------------------------------------------------------------

class TestKocFabregatPalau2021:
    @pytest.mark.parametrize("n_CFx", [4, 6, 8])
    def test_returns_positive(self, n_CFx):
        assert k_oc_fabregat_palau2021(n_CFx) > 0

    def test_increases_with_chain_length(self):
        """Longer chain → larger k_oc."""
        assert k_oc_fabregat_palau2021(8) > k_oc_fabregat_palau2021(6)
        assert k_oc_fabregat_palau2021(6) > k_oc_fabregat_palau2021(4)

    def test_known_value(self):
        """10^(0.41*6 - 0.7) = 10^1.76."""
        expected = 10 ** (0.41 * 6 - 0.7)
        assert k_oc_fabregat_palau2021(6) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# k_sc_fabregat_palau2021
# ---------------------------------------------------------------------------

class TestKscFabregatPalau2021:
    @pytest.mark.parametrize("n_CFx", [4, 6, 8])
    def test_returns_positive(self, n_CFx):
        assert k_sc_fabregat_palau2021(n_CFx) > 0

    def test_increases_with_chain_length(self):
        assert k_sc_fabregat_palau2021(8) > k_sc_fabregat_palau2021(6)

    def test_known_value(self):
        """10^(0.32*6 - 1.7) = 10^0.22."""
        expected = 10 ** (0.32 * 6 - 1.7)
        assert k_sc_fabregat_palau2021(6) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# kd_fabregat_palau
# ---------------------------------------------------------------------------

class TestKdFabregatPalau:
    def test_returns_positive(self):
        Kd = kd_fabregat_palau(n_CFx=6, f_oc=0.01, f_silt_clay=0.3)
        assert Kd > 0

    def test_additive_components(self):
        """Kd should equal k_oc*f_oc + k_sc*f_silt_clay."""
        n = 6
        f_oc = 0.01
        f_sc = 0.3
        expected = k_oc_fabregat_palau2021(n) * f_oc + k_sc_fabregat_palau2021(n) * f_sc
        assert kd_fabregat_palau(n, f_oc, f_sc) == pytest.approx(expected, rel=1e-9)

    def test_zero_fractions(self):
        assert kd_fabregat_palau(6, 0.0, 0.0) == pytest.approx(0.0)

    def test_increases_with_chain_length(self):
        Kd6 = kd_fabregat_palau(6, f_oc=0.01, f_silt_clay=0.2)
        Kd8 = kd_fabregat_palau(8, f_oc=0.01, f_silt_clay=0.2)
        assert Kd8 > Kd6


# ---------------------------------------------------------------------------
# kd_freundlich
# ---------------------------------------------------------------------------

class TestKdFreundlich:
    def test_linear_isotherm(self):
        """n_freund=1 → Kd = K_freund regardless of concentration."""
        assert kd_freundlich(C_rep=5.0, K_freund=2.0, n_freund=1.0) == pytest.approx(2.0)
        assert kd_freundlich(C_rep=0.1, K_freund=2.0, n_freund=1.0) == pytest.approx(2.0)

    def test_zero_concentration_returns_kfreund(self):
        """C_rep=0 should return K_freund directly."""
        assert kd_freundlich(C_rep=0.0, K_freund=3.5, n_freund=0.7) == pytest.approx(3.5)

    def test_nonlinear_isotherm(self):
        """Known calculation: Kd = K * C^(n-1)."""
        K, n, C = 2.0, 0.8, 4.0
        expected = K * C ** (n - 1)
        assert kd_freundlich(C_rep=C, K_freund=K, n_freund=n) == pytest.approx(expected, rel=1e-9)

    def test_returns_float(self):
        result = kd_freundlich(C_rep=2.0, K_freund=1.5, n_freund=0.9)
        assert isinstance(result, float)