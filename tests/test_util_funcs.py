from pathlib import Path

import numpy as np
import pytest

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