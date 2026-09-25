"""Mathematical primitives and preprocessing utilities for ADE analytical solvers.

This module contains analytical solutions based on van Genuchten and Alves
:cite:`vangenuchtenalves1982`, the CXTFIT formulation :cite:`toride1995`, and
its corresponding non-equilibrium transport formulation :cite:`vangenuchten1981`,
with the Bessel-series approximation following Lindstrom and Stone
:cite:`lindstrom1974`.

The module provides:

* dimensionless-parameter computation and the :class:`DimensionlessParams`
  container;
* BVP helper functions for equilibrium sorption;
* IVP helper functions for equilibrium and kinetic sorption;
* kinetic-sorption BVP functions using the Goldstein J-function and its Bessel
  approximation; and
* kinetic-sorption IVP convolution kernels H0 and Hs.

Units
-----
Dimensionless parameters are used throughout the module. Mutually consistent units are supported.
For example, length may be supplied in metres or centimetres, and time may be supplied
in seconds or days.

The required physical dimensions are:

* depth and domain length: length;
* time and switching times: time;
* pore velocity: length / time;
* dispersion coefficient: length**2 / time;
* first-order rate constants: 1 / time;
* mass-transfer coefficients: 1 / time;
* bulk density: mass / length**3; and
* distribution coefficients: length**3 / mass.

Documentation examples generally use SI units, but SI units are not required
provided that all dimensional quantities are mutually consistent.


"""
# ruff: noqa: N803, N806

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import erfc, iv

# Number of modified Bessel function series terms for the Goldstein J-function
# approximation following Lindstrom and Stone (1974).
_BESSEL_TERMS: int = 30


class DimensionlessParams(NamedTuple):
    """Dimensionless parameters for the ADE analytical solution.

    All fields in this container are dimensionless numerical values.

    Attributes
    ----------
    Z : ndarray
        Dimensionless depth, ``Z = z / L``.
    T : ndarray
        Dimensionless time, ``T = t * v / L``.
    T_list : list of float
        Dimensionless switching times corresponding to the physical switching
        times supplied to :func:`compute_dimensionless_params`.
    P : float
        Péclet number, ``P = v * L / D``.
    omega : float or None
        Dimensionless mass-transfer coefficient. ``None`` when ``kinetic`` is
        ``False``.
    """

    Z: NDArray[np.float64]
    T: NDArray[np.float64]
    T_list: list[float]
    P: float
    omega: float | None


def compute_dimensionless_params(
    grid,
    hydro_properties,
    T_list: list[float],
    adsorption=None,
    kinetic: bool = False,
) -> DimensionlessParams:
    """Compute dimensionless parameters for the ADE analytical solution.

    The dimensional inputs must already be represented as numerical values in
    one mutually consistent unit system.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization. ``grid.depth`` has dimensions of
        length and ``grid.time`` has dimensions of time before conversion to
        numerical magnitudes. The final depth value defines the domain length
        ``L``.
    hydro_properties : HydrologicalProperties
        Hydrological properties. ``pore_velocity`` has dimensions of
        length/time and ``dispersion_coefficient`` has dimensions of
        length**2/time.
    T_list : list of float
        Physical switching times, with dimensions of time before unit
        conversion.
        ``T`` values. ``T_list[0]`` should normally be 0.
        Examples include ``[0]`` for a continuous step, ``[0, 5000]`` for a
        pulse from time zero, and ``[0, 2000, 5000]`` for a delayed pulse.
    adsorption : Adsorption, optional
        Adsorption parameters. Required when ``kinetic=True``. The rate
        constant has dimensions of 1/time; the remaining parameters used here
        are dimensionless.
    kinetic : bool, optional
        If True, also compute the dimensionless mass-transfer coefficient
        ``omega``. Default is False.

    Returns
    -------
    DimensionlessParams
        ``Z``, ``T``, ``T_list``, ``P``, and, when requested, ``omega``.
        All returned values are dimensionless plain numerical values.

    Raises
    ------
    ValueError
        If pore velocity or dispersion coefficient is zero, or if kinetic
        transport is requested without adsorption parameters.
    """
    v = hydro_properties.pore_velocity
    D = hydro_properties.dispersion_coefficient

    if v == 0:
        raise ValueError("Pore velocity must be non-zero.")
    if D == 0:
        raise ValueError("Dispersion coefficient must be non-zero.")
    if kinetic and adsorption is None:
        raise ValueError("Adsorption parameters required when kinetic=True.")

    L = grid.depth[-1]
    scale = v / L

    Z = grid.depth / L
    T = grid.time * scale
    T_list_dim = [t * scale for t in T_list]

    omega = None
    if kinetic:
        omega = (
            adsorption.rate_const
            * (1 - adsorption.beta_s)
            * (1 + adsorption.sp_retardation)
            * L
            / v
        )

    return DimensionlessParams(
        Z=Z,
        T=T,
        T_list=T_list_dim,
        P=v * L / D,
        omega=omega,
    )


# ---------------------------------------------------------------------------
# BVP helpers — equilibrium sorption
# All arguments are dimensionless numerical values.
# ---------------------------------------------------------------------------


def _bvp_flux_bc(
    T: float,
    R: float,
    Z: NDArray[np.float64],
    P: float,
) -> NDArray[np.float64]:
    """Return the equilibrium flux-boundary BVP solution.

    Parameters are dimensionless: ``T`` is time, ``R`` is retardation,
    ``Z`` is depth, and ``P`` is the Péclet number. This is the third-type
    solution from Toride et al. :cite:`toride1995`, Table 2.3, with
    ``Omega = 0``.
    """
    arg = np.sqrt(0.25 * P / R / T)
    term1 = 0.5 * erfc(arg * (R * Z - T))
    term2 = np.exp(P * Z) * erfc(arg * (R * Z + T))
    term3 = np.sqrt(P / np.pi / R) * np.exp(-(arg * (R * Z - T)) ** 2)
    return term1 + term3 - 0.5 * (1.0 + P * Z + P * T / R) * term2


def _bvp_resident_bc(
    T: float,
    R: float,
    Z: NDArray[np.float64],
    P: float,
) -> NDArray[np.float64]:
    """Return the equilibrium resident-boundary BVP solution.

    Parameters are dimensionless: ``T`` is time, ``R`` is retardation,
    ``Z`` is depth, and ``P`` is the Péclet number. This is the first-type
    solution from Toride et al. :cite:`toride1995`, Table 2.3, with
    ``Omega = 0``.
    """
    arg = np.sqrt(0.25 * P / R / T)
    term1 = 0.5 * erfc(arg * (R * Z - T))
    term2 = np.exp(P * Z) * erfc(arg * (R * Z + T))
    return term1 + 0.5 * term2


_BVP_FUNCTIONS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "flux": _bvp_flux_bc,
    "resident": _bvp_resident_bc,
}


# ---------------------------------------------------------------------------
# IVP helpers — equilibrium sorption
# ---------------------------------------------------------------------------


def _ivp_eq_flux(
    T: float,
    R: float,
    Z: float,
    P: float,
    xi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the equilibrium flux-boundary IVP Green's-function kernel.

    ``T``, ``R``, ``Z``, ``P``, and ``xi`` are all dimensionless.
    """
    if T <= 0.0:
        return np.zeros_like(xi)

    scale = 4.0 * R * T / P
    prefactor = 1.0 / (2.0 * np.sqrt(np.pi * T / (P * R)))
    direct = np.exp(-(R * Z - R * xi - T) ** 2 / scale)
    image = np.exp(-P * xi) * np.exp(-(R * Z + R * xi - T) ** 2 / scale)
    erfc_term = (
        0.5
        * P
        * np.exp(P * Z)
        * erfc((R * Z + R * xi + T) / (2.0 * np.sqrt(T * R / P)))
    )
    return prefactor * (direct + image) - erfc_term


def _ivp_eq_resident(
    T: float,
    R: float,
    Z: float,
    P: float,
    xi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the equilibrium resident-boundary IVP Green's-function kernel.

    ``T``, ``R``, ``Z``, ``P``, and ``xi`` are all dimensionless.
    """
    if T <= 0.0:
        return np.zeros_like(xi)

    scale = 4.0 * R * T / P
    prefactor = 1.0 / (2.0 * np.sqrt(np.pi * T / (P * R)))
    direct = np.exp(-(R * Z - R * xi - T) ** 2 / scale)
    image = np.exp(-P * xi) * np.exp(-(R * Z + R * xi - T) ** 2 / scale)
    return prefactor * (direct - image)


_IVP_FUNCTIONS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "flux": _ivp_eq_flux,
    "resident": _ivp_eq_resident,
}


# ---------------------------------------------------------------------------
# IVP helpers — kinetic (non-equilibrium) sorption
# ---------------------------------------------------------------------------


def _ivp_neq(  # noqa: PLR0913, PLR0917
    T: float,
    R: float,
    Z: float,
    P: float,
    xi: NDArray[np.float64],
    beta: float,
) -> NDArray[np.float64]:
    """Return the kinetic-sorption IVP Green's-function kernel.

    ``T``, ``R``, ``Z``, ``P``, ``xi``, and ``beta`` are dimensionless.
    """
    if T <= 0.0:
        return np.zeros_like(xi)

    return (
        (
            np.exp(-P * beta * R * (Z - xi - T / (beta * R)) ** 2 / (4 * T))
            + np.exp(
                -xi * P
                - P * beta * R * (Z + xi - T / (beta * R)) ** 2 / (4 * T)
            )
        )
        / (2 * np.sqrt(np.pi * T / (beta * R * P)))
        - P
        / 2
        * np.exp(P * Z)
        * erfc((Z + xi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / P)))
    )


def _H0(  # noqa: PLR0913, PLR0917, N802
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Return the dimensionless aqueous-phase convolution kernel H0.

    All arguments are dimensionless numerical values. ``tau`` and ``T`` are
    dimensionless times. The kernel is from CXTFIT Table 3.4
    :cite:`toride1995`.
    """
    if T <= 0.0:
        return np.zeros_like(tau)

    iv_arg = (
        2
        * omega
        / (1 - beta_s)
        / (1 + R_s)
        * np.sqrt(R_s * (1 - f) * (T - tau) * tau)
        / (beta * R)
    )

    return (
        R_s * (1 - f) / (beta * R)
        * np.exp(
            -omega * (T - tau) / (1 - beta_s) / (1 + R_s)
            - omega
            * tau
            * (1 - f)
            * R_s
            / (1 - beta_s)
            / (beta * R)
            / (1 + R_s)
        )
        * (
            iv(0, iv_arg)
            + iv(1, iv_arg)
            * tau
            / np.sqrt(R_s * (1 - f) * (T - tau) * tau / (beta * R))
        )
    )


def _Hs(  # noqa: PLR0913, PLR0917, N802
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Return the dimensionless sorbed-phase convolution kernel Hs.

    All arguments are dimensionless numerical values. ``tau`` and ``T`` are
    dimensionless times. The kernel is from CXTFIT Table 3.4
    :cite:`toride1995`.
    """
    if T <= 0.0:
        return np.zeros_like(tau)

    iv_arg = (
        2
        * omega
        / (1 - beta_s)
        / (1 + R_s)
        * np.sqrt(R_s * (1 - f) * (T - tau) * tau)
        / (beta * R)
    )

    return np.exp(
        -omega * (T - tau) / (1 - beta_s) / (1 + R_s)
        - omega
        * tau
        * (1 - f)
        * R_s
        / (1 - beta_s)
        / (beta * R)
        / (1 + R_s)
    ) * (
        iv(0, iv_arg)
        + np.sqrt(R_s * (1 - f) * (T - tau) / (beta * R) / tau) * iv(1, iv_arg)
    )


# ---------------------------------------------------------------------------
# BVP helpers — kinetic (non-equilibrium) sorption
# ---------------------------------------------------------------------------


def _FT(  # noqa: N802, PLR0913, PLR0917
    tau: float | NDArray[np.float64],
    Z: float,
    P: float,
    R: float,
    beta: float,
    volume_averaged: bool,
) -> float | NDArray[np.float64]:
    """Return the dimensionless non-equilibrium transport kernel.

    All arguments are dimensionless numerical values. ``tau`` is dimensionless
    integration time and ``Z`` is dimensionless depth. ``P``, ``R``, and
    ``beta`` are dimensionless transport parameters.
    """
    R_beta = beta * R
    term0 = np.sqrt(P / (np.pi * R_beta * tau))
    term1 = np.exp(-0.25 * P / R_beta / tau * (R_beta * Z - tau) ** 2)
    term2 = np.exp(P * Z) * erfc(
        np.sqrt(0.25 * P / R_beta / tau) * (R_beta * Z + tau)
    )

    if volume_averaged:
        return term0 * term1 - 0.5 * (P / R_beta) * term2
    return (Z / tau) * np.sqrt(0.25 * P * R_beta / (np.pi * tau)) * term1


def _goldstein_J(  # noqa: N806, N802
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    m: int = _BESSEL_TERMS,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Evaluate Goldstein's dimensionless J-function and its reverse.

    ``a`` and ``b`` are dimensionless. The Bessel-series approximation follows
    Lindstrom and Stone :cite:`lindstrom1974`. For ``a + b > 10`` an
    erfc-based asymptotic approximation is used.
    """
    if a + b > 10:
        Jab = 0.5 * erfc(
            np.sqrt(a) - np.sqrt(b)
            - 1 / (8 * np.sqrt(a))
            - 1 / (8 * np.sqrt(b))
        )
        Jba = 0.5 * erfc(
            np.sqrt(b) - np.sqrt(a)
            - 1 / (8 * np.sqrt(b))
            - 1 / (8 * np.sqrt(a))
        )
    else:
        Iab_sum: float = 0.0
        Iba_sum: float = 0.0
        sqrt_ab = 2 * np.sqrt(a * b)
        if a >= b:
            ratio = b / a
            for j in range(m):
                Iab_sum += float(ratio ** (j / 2.0)) * float(iv(j, sqrt_ab))
            for j in range(1, m + 1):
                Iba_sum += float(ratio ** (j / 2.0)) * float(iv(j, sqrt_ab))
            Jab = np.exp(-a - b) * Iab_sum
            Jba = 1.0 - np.exp(-a - b) * Iba_sum
        else:
            ratio = a / b
            for j in range(1, m + 1):
                Iab_sum += float(ratio ** (j / 2.0)) * float(iv(j, sqrt_ab))
            for j in range(m):
                Iba_sum += float(ratio ** (j / 2.0)) * float(iv(j, sqrt_ab))
            Jab = 1.0 - np.exp(-a - b) * Iab_sum
            Jba = np.exp(-a - b) * Iba_sum

    return Jab, Jba


def _bvp_neq_integrand(  # noqa: PLR0913, PLR0917, N806, N803
    tau: float,
    T: float,
    Z: float,
    P: float,
    R: float,
    R_s: float,
    beta: float,
    beta_s: float,
    omega: float,
    volume_averaged: bool,
    m: int,
) -> tuple[float, float]:
    """Evaluate the dimensionless A1 and A2 BVP integrands."""
    ft = float(_FT(tau, Z, P, R, beta, volume_averaged))

    if beta_s == 1:
        Jab: float = 1.0
        Jba: float = 1.0
    else:
        a: float = omega * tau / (beta * R)
        b: float = omega * (T - tau) / ((1 - beta_s) * (R_s + 1))
        Jab = float(_goldstein_J(a, b, m)[0])
        Jba = float(_goldstein_J(a, b, m)[1])

    return ft * Jab, ft * (1.0 - Jba)


def _bvp_neq(  # noqa: PLR0913, PLR0917, N806, N803
    Z: float,
    T: float,
    omega: float,
    beta_s: float,
    beta: float,
    P: float,
    R: float,
    R_s: float,
    m: int = _BESSEL_TERMS,
    volume_averaged: bool = True,
) -> tuple[float, float]:
    """Compute dimensionless A1 and A2 BVP solutions.

    All arguments and returned values are dimensionless numerical values.
    ``omega`` is the dimensionless mass-transfer coefficient. The integrals
    follow CXTFIT equations 3.21--3.22 :cite:`toride1995` and use adaptive
    quadrature.
    """
    args = (T, Z, P, R, R_s, beta, beta_s, omega, volume_averaged, m)

    tau_peak = beta * R * Z
    points = [tau_peak] if 1e-10 < tau_peak < T - 1e-10 else []

    A1 = quad(
        lambda tau: _bvp_neq_integrand(tau, *args)[0],
        1e-10,
        T - 1e-10,
        points=points,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )[0]
    A2 = quad(
        lambda tau: _bvp_neq_integrand(tau, *args)[1],
        1e-10,
        T - 1e-10,
        points=points,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )[0]

    return A1, A2
