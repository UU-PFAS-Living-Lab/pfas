<<<<<<< Updated upstream
from pathlib import Path
=======
"""Tests for PFAS analytical solver utility functions."""
>>>>>>> Stashed changes

import numpy as np
import pytest

<<<<<<< Updated upstream
from pfas.analytical_soln import SimulationGrid
from pfas.configuration import read_toml
from pfas.preprocessing import (
    BoundaryPreprocessor,
    GridGenerator,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    SWCAdsorptionPreprocessor,
    WaterPreprocessor,
)
from pfas.utils import aaw_func_thermo, aaw_func_tracer, k_oc_fabregat_palau2021, k_sc_fabregat_palau2021, kd_fabregat_palau, ABfunc, kd_freundlich

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def ab_params():
    return dict(
        Z=0.5,
        T=1.0,
        ws=0.2,
        betas=0.7,
        beta=0.8,
        P=10.0,
        R=1.5,
        Rs=0.5,
        m=5,
    )


@pytest.fixture
def thermo_params():
    return dict(
        sigma0=72.0,     # dyn/cm
        poro=0.4,
        alpha=0.01,
        n=1.6,
        th=0.25,
        thr=0.05,
        ths=0.4,
        sf=1.0,
    )


# ---------------------------------------------------------------------
# ABfunc tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("cflag", [0, 1])
def test_abfunc_returns_finite_positive(ab_params, cflag):
    A, B = ABfunc(cflag=cflag, **ab_params)

    assert np.isfinite(A)
    assert np.isfinite(B)
    assert A >= 0
    assert B >= 0


def test_abfunc_betas_equal_one(ab_params):
    """When betas == 1, Jab and Jba should be unity."""
    params = ab_params.copy()
    params["betas"] = 1.0

    A, B = ABfunc(cflag=1, **params)

    assert np.isfinite(A)
    assert np.isfinite(B)
    assert A >= 0
    assert B >= 0


# ---------------------------------------------------------------------
# aaw_func_thermo tests
# ---------------------------------------------------------------------

def test_aaw_func_thermo_positive(thermo_params):
    Aaw = aaw_func_thermo(**thermo_params)
    assert np.isfinite(Aaw)
    assert Aaw >= 0


def test_aaw_func_thermo_scaling(thermo_params):
    A1 = aaw_func_thermo(**thermo_params)
    thermo_params_scaled = thermo_params.copy()
    thermo_params_scaled["sf"] = 2.0

    A2 = aaw_func_thermo(**thermo_params_scaled)

    assert pytest.approx(A2, rel=1e-6) == 2.0 * A1


# ---------------------------------------------------------------------
# aaw_func_tracer tests
# ---------------------------------------------------------------------

def test_aaw_func_tracer_scalar():
    sw = 0.5
    x2, x1, x0 = 2.0, -1.0, 0.5
    expected = x2 * sw**2 + x1 * sw + x0

    result = aaw_func_tracer(sw, x2, x1, x0)
    assert result == pytest.approx(expected)


def test_aaw_func_tracer_array():
    sw = np.array([0.0, 0.5, 1.0])
    x2, x1, x0 = 1.0, 0.0, 0.0

    result = aaw_func_tracer(sw, x2, x1, x0)
    expected = sw**2

    assert np.allclose(result, expected)


# ---------------------------------------------------------------------
# Fabregat-Palau Kd model tests
# ---------------------------------------------------------------------

def test_k_sc_formula():
    n = 6
    expected = 10 ** (0.32 * n - 1.7)
    assert k_sc_fabregat_palau2021(n) == pytest.approx(expected)


def test_k_oc_formula():
    n = 6
    expected = 10 ** (0.41 * n - 0.7)
    assert k_oc_fabregat_palau2021(n) == pytest.approx(expected)


def test_kd_fabregat_linear_combination():
    n = 6
    f_oc = 0.02
    f_silt_clay = 0.3

    k_oc = k_oc_fabregat_palau2021(n)
    k_sc = k_sc_fabregat_palau2021(n)

    expected = k_oc * f_oc + k_sc * f_silt_clay
    result = kd_fabregat_palau(n, f_oc, f_silt_clay)

    assert result == pytest.approx(expected)


def test_kd_fabregat_monotonic_in_chain_length():
    """Kd should increase with increasing CFx count."""
    f_oc = 0.02
    f_silt_clay = 0.3

    kd_short = kd_fabregat_palau(4, f_oc, f_silt_clay)
    kd_long = kd_fabregat_palau(8, f_oc, f_silt_clay)

    assert kd_long > kd_short


# ---------------------------------------------------------------------
# Freundlich model tests
# ---------------------------------------------------------------------

def test_kd_freundlich_zero_concentration():
    Kf = 5.0
    n = 0.9

    result = kd_freundlich(0.0, Kf, n)
    assert result == Kf


def test_kd_freundlich_linear_case():
    """n_freund = 1 should recover constant Kd."""
    Kf = 5.0
    n = 1.0

    for C in [0.1, 1.0, 10.0]:
        assert kd_freundlich(C, Kf, n) == pytest.approx(Kf)


def test_kd_freundlich_power_law():
    C = 2.0
    Kf = 3.0
    n = 0.8

    expected = Kf * C ** (n - 1)
    result = kd_freundlich(C, Kf, n)

    assert result == pytest.approx(expected)
=======
from pfas.utils import (
    ABfunc,
    aaw_func_thermo,
    aaw_func_tracer,
    kd_fabregat_palau,
    kd_freundlich,
    k_oc_fabregat_palau2021,
    k_sc_fabregat_palau2021,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_abfunc_params():
    """Default parameters for ABfunc producing a valid output."""
    return {
        "Z": 0.5,
        "T": 2.0,
        "ws": 1.0,
        "betas": 0.8,
        "beta": 0.9,
        "P": 10.0,
        "R": 2.0,
        "Rs": 1.5,
        "m": 10,
        "cflag": 1,
    }


@pytest.fixture
def default_aaw_thermo_params():
    """Default parameters for aaw_func_thermo."""
    return {
        "sigma0": 72.0,   # dyn/cm, surface tension of water at ~25°C
        "poro": 0.4,
        "alpha": 0.1,     # cm⁻¹
        "n": 2.0,
        "th": 0.2,
        "thr": 0.05,
        "ths": 0.4,
        "sf": 1.0,
    }


# ---------------------------------------------------------------------------
# ABfunc tests
# ---------------------------------------------------------------------------

class TestABfunc:

    def test_returns_two_values(self, default_abfunc_params):
        result = ABfunc(**default_abfunc_params)
        assert len(result) == 2

    def test_output_is_finite_flux_averaged(self, default_abfunc_params):
        A, B = ABfunc(**default_abfunc_params)
        assert np.isfinite(A)
        assert np.isfinite(B)

    def test_output_is_finite_volume_averaged(self, default_abfunc_params):
        params = {**default_abfunc_params, "cflag": 0}
        A, B = ABfunc(**params)
        assert np.isfinite(A)
        assert np.isfinite(B)

    def test_betas_equals_one_sets_j_functions_to_unity(self, default_abfunc_params):
        """When betas=1 there is no kinetic sorption; B should be ~0."""
        params = {**default_abfunc_params, "betas": 1.0, "beta": 1.0}
        A, B = ABfunc(**params)
        assert np.isfinite(A)
        assert np.isclose(B, 0.0, atol=1e-6)

    def test_a_non_negative(self, default_abfunc_params):
        A, _ = ABfunc(**default_abfunc_params)
        assert A >= 0.0

    def test_different_m_values_consistent(self, default_abfunc_params):
        """Increasing m should not drastically change results (convergence)."""
        A10, B10 = ABfunc(**{**default_abfunc_params, "m": 10})
        A20, B20 = ABfunc(**{**default_abfunc_params, "m": 20})
        assert abs(A10 - A20) < 0.05 * max(abs(A10), 1e-10)
        assert abs(B10 - B20) < 0.05 * max(abs(B10), 1e-10)

    def test_large_z_reduces_concentration(self, default_abfunc_params):
        """Deeper locations should see less solute (lower A)."""
        A_shallow, _ = ABfunc(**{**default_abfunc_params, "Z": 0.2})
        A_deep, _ = ABfunc(**{**default_abfunc_params, "Z": 0.8})
        assert A_shallow >= A_deep


# ---------------------------------------------------------------------------
# aaw_func_thermo tests
# ---------------------------------------------------------------------------

class TestAawFuncThermo:

    def test_returns_positive_value(self, default_aaw_thermo_params):
        aaw = aaw_func_thermo(**default_aaw_thermo_params)
        assert aaw > 0.0

    def test_scaling_factor_scales_linearly(self, default_aaw_thermo_params):
        aaw1 = aaw_func_thermo(**{**default_aaw_thermo_params, "sf": 1.0})
        aaw2 = aaw_func_thermo(**{**default_aaw_thermo_params, "sf": 2.0})
        assert np.isclose(aaw2, 2.0 * aaw1)

    def test_higher_water_content_lower_aaw(self, default_aaw_thermo_params):
        """More water → less air-water interface."""
        aaw_dry = aaw_func_thermo(**{**default_aaw_thermo_params, "th": 0.1})
        aaw_wet = aaw_func_thermo(**{**default_aaw_thermo_params, "th": 0.35})
        assert aaw_dry > aaw_wet

    def test_output_is_finite(self, default_aaw_thermo_params):
        aaw = aaw_func_thermo(**default_aaw_thermo_params)
        assert np.isfinite(aaw)


# ---------------------------------------------------------------------------
# aaw_func_tracer tests
# ---------------------------------------------------------------------------

class TestAawFuncTracer:

    def test_scalar_input(self):
        aaw = aaw_func_tracer(sw=0.5, x2=1.0, x1=0.5, x0=0.1)
        expected = 1.0 * 0.5**2 + 0.5 * 0.5 + 0.1
        assert np.isclose(aaw, expected)

    def test_array_input(self):
        sw = np.array([0.2, 0.5, 0.8])
        aaw = aaw_func_tracer(sw=sw, x2=1.0, x1=0.5, x0=0.1)
        expected = 1.0 * sw**2 + 0.5 * sw + 0.1
        np.testing.assert_allclose(aaw, expected)

    def test_zero_coefficients_returns_intercept(self):
        aaw = aaw_func_tracer(sw=0.7, x2=0.0, x1=0.0, x0=3.5)
        assert np.isclose(aaw, 3.5)

    def test_output_shape_matches_input(self):
        sw = np.linspace(0.1, 0.9, 50)
        aaw = aaw_func_tracer(sw=sw, x2=1.0, x1=0.5, x0=0.1)
        assert aaw.shape == sw.shape


# ---------------------------------------------------------------------------
# kd_fabregat_palau tests
# ---------------------------------------------------------------------------

class TestKdFabregatPalau:

    def test_returns_positive_value(self):
        Kd = kd_fabregat_palau(n_CFx=6, f_oc=0.02, f_silt_clay=0.3)
        assert Kd > 0.0

    def test_higher_n_CFx_higher_Kd(self):
        Kd4 = kd_fabregat_palau(n_CFx=4, f_oc=0.02, f_silt_clay=0.3)
        Kd8 = kd_fabregat_palau(n_CFx=8, f_oc=0.02, f_silt_clay=0.3)
        assert Kd8 > Kd4

    def test_higher_f_oc_higher_Kd(self):
        Kd_low = kd_fabregat_palau(n_CFx=6, f_oc=0.01, f_silt_clay=0.3)
        Kd_high = kd_fabregat_palau(n_CFx=6, f_oc=0.05, f_silt_clay=0.3)
        assert Kd_high > Kd_low

    def test_zero_fractions_returns_zero(self):
        Kd = kd_fabregat_palau(n_CFx=6, f_oc=0.0, f_silt_clay=0.0)
        assert np.isclose(Kd, 0.0)

    def test_additive_contributions(self):
        """Kd should equal k_oc*f_oc + k_sc*f_silt_clay."""
        n_CFx = 6
        f_oc = 0.02
        f_silt_clay = 0.3
        expected = k_oc_fabregat_palau2021(n_CFx) * f_oc + k_sc_fabregat_palau2021(n_CFx) * f_silt_clay
        Kd = kd_fabregat_palau(n_CFx=n_CFx, f_oc=f_oc, f_silt_clay=f_silt_clay)
        assert np.isclose(Kd, expected)


# ---------------------------------------------------------------------------
# k_oc_fabregat_palau2021 tests
# ---------------------------------------------------------------------------

class TestKocFabregatPalau:

    def test_known_value(self):
        """k_oc = 10^(0.41*n - 0.7)."""
        n = 6
        expected = 10 ** (0.41 * n - 0.7)
        assert np.isclose(k_oc_fabregat_palau2021(n), expected)

    def test_monotonically_increasing(self):
        values = [k_oc_fabregat_palau2021(n) for n in range(3, 12)]
        assert all(v2 > v1 for v1, v2 in zip(values, values[1:]))

    def test_positive(self):
        assert k_oc_fabregat_palau2021(4) > 0


# ---------------------------------------------------------------------------
# k_sc_fabregat_palau2021 tests
# ---------------------------------------------------------------------------

class TestKscFabregatPalau:

    def test_known_value(self):
        """k_sc = 10^(0.32*n - 1.7)."""
        n = 6
        expected = 10 ** (0.32 * n - 1.7)
        assert np.isclose(k_sc_fabregat_palau2021(n), expected)

    def test_monotonically_increasing(self):
        values = [k_sc_fabregat_palau2021(n) for n in range(3, 12)]
        assert all(v2 > v1 for v1, v2 in zip(values, values[1:]))

    def test_positive(self):
        assert k_sc_fabregat_palau2021(4) > 0


# ---------------------------------------------------------------------------
# kd_freundlich tests
# ---------------------------------------------------------------------------

class TestKdFreundlich:

    def test_linear_isotherm(self):
        """n_freund=1 should give Kd = K_freund regardless of C_rep."""
        Kd = kd_freundlich(C_rep=5.0, K_freund=2.0, n_freund=1.0)
        assert np.isclose(Kd, 2.0)

    def test_zero_concentration_returns_K_freund(self):
        Kd = kd_freundlich(C_rep=0.0, K_freund=3.5, n_freund=0.7)
        assert np.isclose(Kd, 3.5)

    def test_favourable_sorption_n_less_than_1(self):
        """n < 1: Kd decreases with increasing concentration."""
        Kd_low = kd_freundlich(C_rep=1.0, K_freund=2.0, n_freund=0.7)
        Kd_high = kd_freundlich(C_rep=10.0, K_freund=2.0, n_freund=0.7)
        assert Kd_low > Kd_high

    def test_known_value(self):
        """Kd = K_freund * C_rep^(n-1) = 2.0 * 4.0^0.5 = 4.0."""
        Kd = kd_freundlich(C_rep=4.0, K_freund=2.0, n_freund=1.5)
        expected = 2.0 * 4.0 ** 0.5
        assert np.isclose(Kd, expected)

    def test_positive_output(self):
        Kd = kd_freundlich(C_rep=2.0, K_freund=1.5, n_freund=0.8)
        assert Kd > 0.0
>>>>>>> Stashed changes
