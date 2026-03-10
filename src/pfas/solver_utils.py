"""Mathematical primitives and preprocessing utilities for ADE analytical solvers.

This module contains:
- Dimensionless parameter computation and the :class:`DimensionlessParams` container
- BVP helper functions for equilibrium sorption (one per boundary condition type)
- IVP helper functions for equilibrium and kinetic sorption
- Kinetic sorption BVP functions (Goldstein J-function + Bessel approximation)
- Kinetic sorption IVP convolution kernels H₀ and Hₛ (Bessel function based, CXTFIT Table 3.4)

The high-level solvers in ``solvers.py`` import from here. To add a new boundary
condition, implement a helper with the signature ``f(T, R, Z, P) -> ndarray`` and
register it in ``_BVP_FUNCTIONS``.

References
----------
van Genuchten & Alves (1982): Analytical solutions of the one-dimensional
convective-dispersive solute transport equation.
Toride, Leij & van Genuchten (1995): The CXTFIT Code, Version 2.0,
USDA Research Report No. 137.
van Genuchten, M. Th. (1981): Non-Equilibrium Transport Parameters from
Miscible Displacement Experiments. Research Report No. 119, USDA-ARS.
Lindstrom, F.T. and Stone, W.J. (1974): On the start up or initial phase
of linear mass transport of chemicals in a water saturated sorbing porous
medium. Soil Science Society of America Proceedings.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import erfc, iv


# Number of modified Bessel function series terms for the Goldstein J-function
# approximation following Lindstrom and Stone (1974).
_BESSEL_TERMS: int = 30


# ---------------------------------------------------------------------------
# Dimensionless parameter container and computation
# ---------------------------------------------------------------------------

class DimensionlessParams(NamedTuple):
    """Dimensionless parameters for the ADE analytical solution.

    Attributes
    ----------
    Z : ndarray
        Dimensionless depth (-), Z = z / L.
    T : ndarray
        Dimensionless time (-), T = t * v / L.
    pulses : list of (float, float)
        Dimensionless pulse intervals [(T_start, T_end), ...].
        Each tuple defines one on/off period of the inlet concentration.
        Use ``(0, np.inf)`` for a continuous step input.
        Use ``[(0, T0)]`` for a pulse starting at T=0 of duration T0.
        Use ``[(T_start, T_end)]`` for a delayed pulse.
        Multiple tuples superimpose several pulses.
    P : float
        Péclet number (-), P = v * L / D.
    omega : float or None
        Dimensionless mass transfer coefficient (ω, CXTFIT Table 3.1).
        None when kinetic=False.
    """

    Z: NDArray[np.float64]
    T: NDArray[np.float64]
    pulses: list[tuple[float, float]]
    P: float
    omega: float | None


def compute_dimensionless_params(
    grid,
    boundary_conditions,
    hydro_properties,
    pulse_intervals: list[tuple[float, float]],
    adsorption=None,
    kinetic: bool = False,
) -> DimensionlessParams:
    """Compute dimensionless parameters for the ADE analytical solution.

    Converts physical grid, flow, and transport parameters into the
    dimensionless form required by the equilibrium and kinetic solvers.
    Pulse intervals in physical time (seconds) are converted to dimensionless
    time T = t * v / L.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization. Must have `.depth` (m) and
        `.time` (s) arrays.
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions.
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` (m/s) and
        `.dispersion_coefficient` (m²/s).
    pulse_intervals : list of (float, float)
        Inlet concentration on/off periods in physical time (s), e.g.:
        - Step input:               ``[(0, np.inf)]``
        - Pulse from t=0:           ``[(0, 5000)]``
        - Delayed pulse:            ``[(2000, 5000)]``
        - Multiple pulses:          ``[(0, 1000), (3000, 5000)]``
    adsorption : Adsorption, optional
        Adsorption parameters. Required when kinetic=True. Must have
        `.rate_const`, `.beta_s`, `.sp_retardation`.
    kinetic : bool, optional
        If True, also compute the Damköhler number `omega`. Default is False.

    Returns
    -------
    DimensionlessParams
        Named tuple containing Z, T, pulses (dimensionless), P, and omega.

    Raises
    ------
    ValueError
        If pore_velocity or dispersion_coefficient is zero.
    ValueError
        If kinetic=True but adsorption is None.
    ValueError
        If any pulse interval has t_start >= t_end.
    """
    v = hydro_properties.pore_velocity
    D = hydro_properties.dispersion_coefficient

    if v == 0:
        raise ValueError("Pore velocity must be non-zero.")
    if D == 0:
        raise ValueError("Dispersion coefficient must be non-zero.")
    if kinetic and adsorption is None:
        raise ValueError("Adsorption parameters required when kinetic=True.")
    for t_start, t_end in pulse_intervals:
        if t_start >= t_end:
            raise ValueError(
                f"Invalid pulse interval ({t_start}, {t_end}): "
                "t_start must be strictly less than t_end."
            )

    L = grid.depth[-1]
    scale = v / L

    Z = grid.depth / L
    T = grid.time * scale

    # Convert physical pulse intervals to dimensionless time
    pulses = [
        (t_start * scale, t_end * scale if t_end != np.inf else np.inf)
        for t_start, t_end in pulse_intervals
    ]

    omega = None
    if kinetic:
        omega = (
            adsorption.rate_const
            * (1 - adsorption.beta_s)
            * (1 + adsorption.sp_retardation)
            * L
            / v
        )

    return DimensionlessParams(Z=Z, T=T, pulses=pulses, P=v * L / D, omega=omega)


# ---------------------------------------------------------------------------
# BVP helpers — equilibrium sorption
# One function per boundary condition type, all sharing the signature:
#     f(T, R, Z, P) -> ndarray
# Register new BCs in _BVP_FUNCTIONS below.
# ---------------------------------------------------------------------------

def _bvp_flux_bc(
    T: float,
    R: float,
    Z: NDArray[np.float64],
    P: float,
) -> NDArray[np.float64]:
    """BVP solution for flux (third-type) upper boundary condition.

    Analytical solution to the 1D ADE with a constant step input at the inlet
    using a flux (third-type) boundary condition, evaluated over dimensionless
    depth Z. This is the kernel A1(Z,T) used in pulse superposition in the
    equilibrium solver.

    The solution is:

        C(Z,T) = 0.5 * erfc(arg * (RZ - T))
                 + sqrt(PT / pi R) * exp(-arg^2 * (RZ - T)^2)
                 - 0.5 * (1 + PZ + PT/R) * exp(PZ) * erfc(arg * (RZ + T))

    where arg = sqrt(P / 4RT).

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor.
    Z : ndarray
        Dimensionless depth, Z = z/L.
    P : float
        Péclet number, P = vL/D.

    Returns
    -------
    ndarray
        Dimensionless concentration profile at time T.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 2.1, Case A2 (third-type inlet BC, semi-infinite
    domain). Equivalent to van Genuchten & Alves (1982), eq. B2.
    """
    arg = np.sqrt(0.25 * P / R / T)

    term1 = 0.5 * erfc(arg * (R * Z - T))
    term2 = np.exp(P * Z) * erfc(arg * (R * Z + T))
    term3 = np.sqrt(P * T / np.pi / R) * np.exp(-0.25 * P / R / T * (R * Z - T) ** 2)

    return term1 + term3 - 0.5 * (1.0 + P * Z + P * T / R) * term2


def _bvp_resident_bc(
    T: float,
    R: float,
    Z: NDArray[np.float64],
    P: float,
) -> NDArray[np.float64]:
    """BVP solution for resident (first-type) upper boundary condition.

    Analytical solution to the 1D ADE with a constant step input at the inlet
    using a resident (first-type) boundary condition, evaluated over
    dimensionless depth Z. This is the kernel A1(Z,T) for the first-type BC
    used in pulse superposition in the equilibrium solver.

    The solution is:

        C(Z,T) = 0.5 * erfc(arg * (RZ - T))
                 + 0.5 * exp(PZ) * erfc(arg * (RZ + T))

    where arg = sqrt(P / 4RT).

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor.
    Z : ndarray
        Dimensionless depth, Z = z/L.
    P : float
        Péclet number, P = vL/D.

    Returns
    -------
    ndarray
        Dimensionless concentration profile at time T.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 2.1, Case A1 (first-type inlet BC, semi-infinite
    domain). Equivalent to van Genuchten & Alves (1982), eq. B1.
    """
    arg = np.sqrt(0.25 * P / R / T)

    term1 = 0.5 * erfc(arg * (R * Z - T))
    term2 = np.exp(P * Z) * erfc(arg * (R * Z + T))

    return term1 + 0.5 * term2


# Registry: add new BC helpers here without touching solvers.py
_BVP_FUNCTIONS: dict[str, callable] = {
    "flux": _bvp_flux_bc,
    "resident": _bvp_resident_bc,
}


# ---------------------------------------------------------------------------
# IVP helpers — equilibrium sorption
# ---------------------------------------------------------------------------

def _ivp_eq(
    T: float,
    R: float,
    Z: float,
    P: float,
    xi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """IVP integrand for equilibrium sorption (Green's function kernel).

    Computes the Green's function kernel G(Z, T, xi) for the superposition
    integral over a non-zero initial concentration profile Ci(xi):

        C^I(Z, T) = integral_0^1 G(Z, T, xi) * Ci(xi) dxi

    The kernel combines two Gaussian terms (direct and image source) and an
    erfc term arising from the flux boundary condition at Z=0:

        G(Z,T,xi) = [ exp(-(R(Z-xi)-T)^2 / (4TR/P))
                      + exp(-P*xi - (R(Z+xi)-T)^2 / (4TR/P)) ]
                    / (2*sqrt(pi*T/PR))
                    - P/2 * exp(PZ) * erfc((R(Z+xi)+T) / (2*sqrt(TR/P)))

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor.
    Z : float
        Dimensionless depth at evaluation point (scalar).
    P : float
        Péclet number, P = vL/D.
    xi : ndarray
        Dimensionless depth coordinate (ξ) stepping over the initial
        concentration profile. Corresponds to ξ in van Genuchten & Alves (1982).

    Returns
    -------
    ndarray
        Kernel values G(Z, T, xi) over xi, for numerical integration via trapezoid.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 2.2, Case A2 (third-type inlet BC, semi-infinite
    domain, arbitrary initial condition). Equivalent to van Genuchten &
    Alves (1982), eq. C2.
    """
    return (
        (
            np.exp(-((R * Z - R * xi - T) ** 2) / (4 * T * R / P))
            + np.exp(-P * xi - (R * Z + R * xi - T) ** 2 / (4 * T * R / P))
        )
        / (2 * np.sqrt(np.pi * T / P / R))
        - P / 2 * np.exp(P * Z) * erfc((R * Z + R * xi + T) / (2 * np.sqrt(T * R / P)))
    )


# ---------------------------------------------------------------------------
# IVP helpers — kinetic (non-equilibrium) sorption
# ---------------------------------------------------------------------------

def _ivp_neq(  # noqa: PLR0913
    T: float,
    R: float,
    Z: float,
    P: float,
    xi: NDArray[np.float64],
    beta: float,
) -> NDArray[np.float64]:
    """IVP integrand for non-equilibrium (kinetic) sorption (Green's function kernel).

    Computes the Green's function kernel G_neq(Z, T, xi) for the superposition
    integral over a non-zero initial concentration profile Ci(xi) under kinetic
    sorption conditions:

        C^I_1(Z, T) = integral_0^1 G_neq(Z, T, xi) * Ci(xi) dxi

    The kernel is the nonequilibrium analogue of the equilibrium IVP kernel,
    with the effective retardation modified by the kinetic partitioning
    coefficient beta:

        G_neq(Z,T,xi) = [ exp(-P*beta*R*(Z-xi-T/(betaR))^2 / (4T))
                          + exp(-P*xi - P*beta*R*(Z+xi-T/(betaR))^2 / (4T)) ]
                        / (2*sqrt(pi*T / (beta*R*P)))
                        - P/2 * exp(PZ) * erfc((Z+xi+T/(betaR)) / (2*sqrt(T/(betaR*P))))

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor for aqueous phase.
    Z : float
        Dimensionless depth at evaluation point (scalar).
    P : float
        Péclet number, P = vL/D.
    xi : ndarray
        Dimensionless depth coordinate (ξ) stepping over the initial
        concentration profile.
    beta : float
        Dimensionless partitioning coefficient (β in CXTFIT Table 3.1).

    Returns
    -------
    ndarray
        Kernel values G_neq(Z, T, xi) over xi, for numerical integration via trapezoid.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 3.2, Case A2 (third-type inlet BC, semi-infinite
    domain, nonequilibrium IVP). Equivalent to Toride et al. (1993), Water
    Resour. Res. 29(7), eq. for nonequilibrium IVP Green's function.
    """
    return (
        (
            np.exp(-P * beta * R * (Z - xi - T / (beta * R)) ** 2 / (4 * T))
            + np.exp(-xi * P - P * beta * R * (Z + xi - T / (beta * R)) ** 2 / (4 * T))
        )
        / (2 * np.sqrt(np.pi * T / (beta * R * P)))
        - P / 2 * np.exp(P * Z)
        * erfc((Z + xi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / P)))
    )


def _H0(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel H₀(τ; T) for the aqueous phase kinetic IVP.

    Computes the kernel H₀(τ; T) appearing in the convolution integral for
    the aqueous phase concentration C₁ under kinetic sorption conditions
    (CXTFIT Table 3.4). Used to evaluate the time-history contribution to C₁
    from non-zero initial conditions:

        C1_ivp += (ω / ((1-β_s)·(1+R_s))) · ∫₀ᵀ H₀(τ; T) · G(Z, τ) dτ

    where G(Z, τ) is the nonequilibrium IVP Green's function :func:`_ivp_neq`
    evaluated at intermediate times τ (stored in G_Ztau).

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Overall retardation factor, R = 1 + ρ_b·Kd/θ.
    tau : ndarray
        Dimensionless integration time variable (τ), 0 < τ < T.
    R_s : float
        Retardation factor for kinetic sorption sites (CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at instantaneous equilibrium (CXTFIT
        Table 3.1), 0 <= f <= 1.
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
    omega : float
        Dimensionless mass transfer coefficient ω (CXTFIT Table 3.1).

    Returns
    -------
    ndarray
        Kernel values H₀(τ; T) over tau for numerical integration.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 3.4, kernel H₀ for aqueous phase IVP convolution.
    """
    iv_arg = (
        2 * omega / (1 - beta_s) / (1 + R_s)
        * np.sqrt(R_s * (1 - f) * (T - tau) * tau)
        / (beta * R)
    )

    return (
        R_s * (1 - f) / (beta * R)
        * np.exp(
            -omega * (T - tau) / (1 - beta_s) / (1 + R_s)
            - omega * tau * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s)
        )
        * (
            iv(0, iv_arg)
            + iv(1, iv_arg) * tau / np.sqrt(R_s * (1 - f) * (T - tau) * tau / (beta * R))
        )
    )


def _Hs(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel Hₛ(τ; T) for the sorbed phase kinetic IVP.

    Computes the kernel Hₛ(τ; T) appearing in the convolution integral for
    the sorbed phase concentration C₂ under kinetic sorption conditions
    (CXTFIT Table 3.4). Used to evaluate the time-history contribution to C₂
    from non-zero initial conditions:

        C2_ivp += (ω / ((1-β_s)·(1+R_s))) · (1-f)·Kd · ∫₀ᵀ Hₛ(τ; T) · G(Z, τ) dτ

    where G(Z, τ) is the nonequilibrium IVP Green's function :func:`_ivp_neq`
    evaluated at intermediate times τ (stored in G_Ztau).

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Overall retardation factor, R = 1 + ρ_b·Kd/θ.
    tau : ndarray
        Dimensionless integration time variable (τ), 0 < τ < T.
    R_s : float
        Retardation factor for kinetic sorption sites (CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at instantaneous equilibrium (CXTFIT
        Table 3.1), 0 <= f <= 1.
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
    omega : float
        Dimensionless mass transfer coefficient ω (CXTFIT Table 3.1).

    Returns
    -------
    ndarray
        Kernel values Hₛ(τ; T) over tau for numerical integration.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 3.4, kernel Hₛ for sorbed phase IVP convolution.
    """
    iv_arg = (
        2 * omega / (1 - beta_s) / (1 + R_s)
        * np.sqrt(R_s * (1 - f) * (T - tau) * tau)
        / (beta * R)
    )

    return (
        np.exp(
            -omega * (T - tau) / (1 - beta_s) / (1 + R_s)
            - omega * tau * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s)
        )
        * (
            iv(0, iv_arg)
            + np.sqrt(R_s * (1 - f) * (T - tau) / (beta * R) / tau) * iv(1, iv_arg)
        )
    )


# ---------------------------------------------------------------------------
# BVP helpers — kinetic (non-equilibrium) sorption
# Transport kernel, Goldstein J-function, and A₁/A₂ BVP solutions
# (CXTFIT eqs. 3.21–3.22; van Genuchten, 1981)
# ---------------------------------------------------------------------------

def _FT(
    tau: float | NDArray[np.float64],
    Z: float,
    P: float,
    R: float,
    beta: float,
    volume_averaged: bool,
) -> float | NDArray[np.float64]:
    """Green's function transport kernel Γ₁ᴺ(Z, τ) for the nonequilibrium BVP.

    Computes the advection-dispersion kernel appearing in the integrands of
    A₁ and A₂ (CXTFIT eqs. 3.21–3.22) at dimensionless depth Z.
    Corresponds to FT(τ) in van Genuchten (1981).

    Two forms are available depending on the concentration averaging mode:

    Volume-averaged (resident) concentration — bc=2 in van Genuchten (1981):

        FT(Z, τ) = sqrt(P / (π·βR·τ)) · exp(-P·(βRZ - τ)² / (4βRτ))
                   - (P / 2βR) · exp(PZ) · erfc(sqrt(P / (4βRτ)) · (βRZ + τ))

    Flux-averaged concentration — bc=1 in van Genuchten (1981):

        FT(Z, τ) = (Z/τ) · sqrt(P·βR / (4π·τ)) · exp(-P·(βRZ - τ)² / (4βRτ))

    Parameters
    ----------
    tau : float or ndarray
        Dimensionless integration variable (0 < τ < T).
    Z : float
        Dimensionless depth, Z = z/L.
    P : float
        Péclet number P = vL/D.
    R : float
        Overall retardation factor R = 1 + ρ_b·Kd/θ.
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    volume_averaged : bool
        If True, use the volume-averaged (resident) concentration kernel.
        If False, use the flux-averaged concentration kernel.

    Returns
    -------
    float or ndarray
        Transport kernel value(s) at tau.

    References
    ----------
    van Genuchten (1981), eq. for FT (Research Report No. 119, USDA-ARS).
    CXTFIT eq. 3.21, kernel Γ₁ᴺ(Z, τ).
    """
    R_beta = beta * R
    term0 = np.sqrt(P / (np.pi * R_beta * tau))
    term1 = np.exp(-0.25 * P / R_beta / tau * (R_beta * Z - tau) ** 2)
    term2 = np.exp(P * Z) * erfc(np.sqrt(0.25 * P / R_beta / tau) * (R_beta * Z + tau))

    if volume_averaged:
        # Resident (volume-averaged) concentration — bc=2 in van Genuchten (1981)
        return term0 * term1 - 0.5 * (P / R_beta) * term2
    else:
        # Flux-averaged concentration — bc=1 in van Genuchten (1981)
        return (Z / tau) * np.sqrt(0.25 * P * R_beta / (np.pi * tau)) * term1


def _goldstein_J(
    a: float,
    b: float,
    m: int = _BESSEL_TERMS,
) -> tuple[float, float]:
    """Evaluate Goldstein's J-function J(a,b) and J(b,a) via Bessel series.

    Approximates Goldstein's J-function (Goldstein, 1953) using modified
    Bessel functions Iⱼ following Lindstrom and Stone (1974), with an
    erfc-based asymptotic expansion for large arguments (a+b > 10).
    Returns both J(a,b) and J(b,a) as needed for A₁ and A₂ respectively
    (CXTFIT eqs. 3.21–3.22, Table 3.4).

    For a + b > 10 the asymptotic approximation is used:

        J(a,b) ≈ 0.5 · erfc(√a - √b - 1/(8√a) - 1/(8√b))

    Otherwise the modified Bessel series is evaluated:

        J(a,b) = exp(-a-b) · Σⱼ (b/a)^(j/2) · Iⱼ(2√(ab))   [if a ≥ b]
        J(a,b) = 1 - exp(-a-b) · Σⱼ (a/b)^(j/2) · Iⱼ(2√(ab))  [if a < b]

    Parameters
    ----------
    a : float
        First argument: a = ω·τ / (β·R) (CXTFIT Table 3.4).
    b : float
        Second argument: b = ω·(T-τ) / ((1-β_s)·(R_s+1)) (CXTFIT Table 3.4).
    m : int, optional
        Number of Bessel function series terms. Default is ``_BESSEL_TERMS`` (30).

    Returns
    -------
    Jab : float
        J(a, b) — used in the A₁ integrand (equilibrium phase, eq. 3.21).
    Jba : float
        J(b, a) — used in the A₂ integrand (nonequilibrium phase, eq. 3.22).

    References
    ----------
    Goldstein, S. (1953). Proc. R. Soc. London A, 219, 151–171.
    Lindstrom, F.T. and Stone, W.J. (1974). Soil Sci. Soc. Am. Proc.
    CXTFIT Table 3.4.
    """
    if a + b > 10:
        Jab = 0.5 * erfc(
            np.sqrt(a) - np.sqrt(b)
            - 1 / (8 * np.sqrt(a)) - 1 / (8 * np.sqrt(b))
        )
        Jba = 0.5 * erfc(
            np.sqrt(b) - np.sqrt(a)
            - 1 / (8 * np.sqrt(b)) - 1 / (8 * np.sqrt(a))
        )
    else:
        Iab_sum = 0.0
        Iba_sum = 0.0
        sqrt_ab = 2 * np.sqrt(a * b)
        if a >= b:
            ratio = b / a
            for j in range(m):
                Iab_sum += ratio ** (j / 2.0) * iv(j, sqrt_ab)
            for j in range(1, m + 1):
                Iba_sum += ratio ** (j / 2.0) * iv(j, sqrt_ab)
            Jab = np.exp(-a - b) * Iab_sum
            Jba = 1.0 - np.exp(-a - b) * Iba_sum
        else:
            ratio = a / b
            for j in range(1, m + 1):
                Iab_sum += ratio ** (j / 2.0) * iv(j, sqrt_ab)
            for j in range(m):
                Iba_sum += ratio ** (j / 2.0) * iv(j, sqrt_ab)
            Jab = 1.0 - np.exp(-a - b) * Iab_sum
            Jba = np.exp(-a - b) * Iba_sum

    return Jab, Jba


def _bvp_neq_integrand(  # noqa: PLR0913
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
    """Integrand for the nonequilibrium BVP integrals A₁ and A₂.

    Evaluates FT(Z, τ)·J(a,b) and FT(Z, τ)·[1-J(b,a)] at a single tau value,
    as appearing in CXTFIT eqs. 3.21–3.22. Separated from :func:`_bvp_neq`
    to allow use with ``scipy.integrate.quad`` for adaptive quadrature.

    The arguments to the Goldstein J-function are (CXTFIT Table 3.4):

        a = ω·τ / (β·R)
        b = ω·(T-τ) / ((1-β_s)·(R_s+1))

    Parameters
    ----------
    tau : float
        Dimensionless integration variable (0 < τ < T).
    T : float
        Dimensionless time.
    Z : float
        Dimensionless depth, Z = z/L.
    P : float
        Péclet number.
    R : float
        Overall retardation factor.
    R_s : float
        Retardation factor for kinetic sorption sites.
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
    omega : float
        Dimensionless mass transfer coefficient ω (CXTFIT Table 3.1).
    volume_averaged : bool
        Concentration averaging mode passed to :func:`_FT`.
    m : int
        Number of Bessel series terms passed to :func:`_goldstein_J`.

    Returns
    -------
    integrand_A1 : float
        FT(Z, τ) · J(a, b) — integrand for A₁ (eq. 3.21, equilibrium phase).
    integrand_A2 : float
        FT(Z, τ) · [1 - J(b, a)] — integrand for A₂ (eq. 3.22, nonequilibrium phase).
    """
    ft = _FT(tau, Z, P, R, beta, volume_averaged)

    if beta_s == 1:
        Jab, Jba = 1.0, 1.0
    else:
        a = omega * tau / (beta * R)
        b = omega * (T - tau) / ((1 - beta_s) * (R_s + 1))
        Jab, Jba = _goldstein_J(a, b, m)

    return ft * Jab, ft * (1.0 - Jba)


def _bvp_neq(  # noqa: PLR0913
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
    """Compute A₁ and A₂: nonequilibrium BVP solutions (CXTFIT eqs. 3.21–3.22).

    Evaluates the equilibrium-phase (A₁, k=1) and nonequilibrium-phase (A₂, k=2)
    BVP solutions for the nonequilibrium ADE with first-order kinetic sorption,
    via adaptive quadrature (``scipy.integrate.quad``) over the product of the
    transport kernel :func:`_FT` and Goldstein's J-function :func:`_goldstein_J`.

    The integrals are (CXTFIT eqs. 3.21–3.22):

        A₁(Z,T) = ∫₀ᵀ FT(τ) · J(a, b) dτ

        A₂(Z,T) = (ω / (ω+μ₂)) · ∫₀ᵀ FT(τ) · [1 - J(b, a)] dτ

    where a and b are defined in CXTFIT Table 3.4 and FT is the transport
    kernel from van Genuchten (1981), evaluated at depth Z.

    Parameters
    ----------
    Z : float
        Dimensionless depth (Z = z/L).
    T : float
        Dimensionless time (T = vt/L).
    omega : float
        Dimensionless mass transfer coefficient ω (CXTFIT Table 3.1).
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
        Set to 1 for fully equilibrium sorption (no kinetic sites).
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    P : float
        Péclet number P = vL/D.
    R : float
        Overall retardation factor R = 1 + ρ_b·Kd/θ.
    R_s : float
        Retardation factor for kinetic sorption sites (CXTFIT Table 3.1).
    m : int, optional
        Number of Bessel function series terms for :func:`_goldstein_J`.
        Default is ``_BESSEL_TERMS`` (30).
    volume_averaged : bool, optional
        If True, use volume-averaged (resident) concentration kernel in
        :func:`_FT`. Default is True.

    Returns
    -------
    A1 : float
        Equilibrium-phase BVP contribution, k=1 (CXTFIT eq. 3.21).
    A2 : float
        Nonequilibrium-phase BVP contribution, k=2 (CXTFIT eq. 3.22).

    References
    ----------
    Toride, Leij & van Genuchten (1995). CXTFIT Version 2.0. Research Report
    No. 137, USDA-ARS. Eqs. (3.20)–(3.22), Table 3.1, Table 3.4.

    van Genuchten, M. Th. (1981). Research Report No. 119, USDA-ARS.

    Lindstrom, F.T. and Stone, W.J. (1974). Soil Sci. Soc. Am. Proc.
    """
    args = (T, Z, P, R, R_s, beta, beta_s, omega, volume_averaged, m)

    # The Gaussian in _FT peaks sharply near tau = beta*R*Z. Passing this as
    # a breakpoint tells quad to subdivide around the peak and avoids
    # systematic underestimation of the integral for narrow peaks.
    tau_peak = beta * R * Z
    points = [tau_peak] if 1e-10 < tau_peak < T - 1e-10 else []

    A1 = quad(
        lambda tau: _bvp_neq_integrand(tau, *args)[0],
        1e-10, T - 1e-10,
        points=points,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )[0]
    A2 = quad(
        lambda tau: _bvp_neq_integrand(tau, *args)[1],
        1e-10, T - 1e-10,
        points=points,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )[0]

    return A1, A2