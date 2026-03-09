"""Mathematical primitives and preprocessing utilities for ADE analytical solvers.

This module contains:
- Dimensionless parameter computation and the :class:`DimensionlessParams` container
- BVP helper functions for equilibrium sorption (one per boundary condition type)
- IVP helper functions for equilibrium and kinetic sorption
- Kinetic sorption convolution kernels (Bessel function based)

The high-level solvers in ``solvers.py`` import from here. To add a new boundary
condition, implement a helper with the signature ``f(T, R, Z, P) -> ndarray`` and
register it in ``_BVP_FUNCTIONS``.

References
----------
van Genuchten & Alves (1982): Analytical solutions of the one-dimensional
convective-dispersive solute transport equation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, iv


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
    T0 : float
        Dimensionless pulse duration (-), T0 = t_pulse * v / L.
    P : float
        Péclet number (-), P = v * L / D.
    omega : float or None
        Damköhler number for kinetic sorption (-).
        None when kinetic=False.
    """

    Z: NDArray[np.float64]
    T: NDArray[np.float64]
    T0: float
    P: float
    omega: float | None


def compute_dimensionless_params(
    grid,
    boundary_conditions,
    hydro_properties,
    adsorption=None,
    kinetic: bool = False,
) -> DimensionlessParams:
    """Compute dimensionless parameters for the ADE analytical solution.

    Converts physical grid, flow, and transport parameters into the
    dimensionless form required by the equilibrium and kinetic solvers.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization. Must have `.depth` (m) and
        `.time` (s) arrays.
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions. Must have `.pulse_time` (s).
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` (m/s) and
        `.dispersion_coefficient` (m²/s).
    adsorption : Adsorption, optional
        Adsorption parameters. Required when kinetic=True. Must have
        `.rate_const`, `.beta_s`, `.sp_retardation`.
    kinetic : bool, optional
        If True, also compute the Damköhler number `omega`. Default is False.

    Returns
    -------
    DimensionlessParams
        Named tuple containing Z, T, T0, P, and omega (None if kinetic=False).

    Raises
    ------
    ValueError
        If pore_velocity or dispersion_coefficient is zero.
    ValueError
        If kinetic=True but adsorption is None.
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

    Z = grid.depth / L
    T = grid.time * (v / L)
    T0 = boundary_conditions.pulse_time * (v / L)
    P = v * L / D

    omega = None
    if kinetic:
        omega = (
            adsorption.rate_const
            * (1 - adsorption.beta_s)
            * (1 + adsorption.sp_retardation)
            * L
            / v
        )

    return DimensionlessParams(Z=Z, T=T, T0=T0, P=P, omega=omega)


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
        Kernel values G(Z, T, xi) over xi, for numerical integration via trapz.

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
# IVP and convolution helpers — kinetic sorption
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
        Kinetic sorption partitioning coefficient (beta = 1 - (1-f)*Kd*rhob/theta).

    Returns
    -------
    ndarray
        Kernel values G_neq(Z, T, xi) over xi, for numerical integration via trapz.

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


def _kinetic_kernel_aqueous(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel for aqueous phase kinetic sorption.

    Computes the kernel function used in the convolution integral for the
    aqueous phase concentration under kinetic sorption conditions.

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor, R = 1 + rho_b * Kd / theta.
    tau : ndarray
        Dimensionless integration time variable (τ), 0 <= τ <= T.
    R_s : float
        Retardation factor for kinetic sorption sites (R_s in CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at equilibrium (f in CXTFIT Table 3.1),
        0 <= f <= 1.
    beta : float
        Dimensionless partitioning coefficient (β in CXTFIT Table 3.1).
    beta_s : float
        Partitioning coefficient for the solid phase (β_s in CXTFIT Table 3.1).
    omega : float
        Dimensionless mass transfer coefficient (ω in CXTFIT Table 3.1).

    Returns
    -------
    ndarray
        Kernel values H_0(τ; T) over tau for numerical integration (Table 3.4).

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 3.4, kernel H_0 for aqueous phase IVP convolution.
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


def _kinetic_kernel_sorbed(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    R_s: float,
    f: float,
    beta: float,
    beta_s: float,
    omega: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel for sorbed phase kinetic sorption.

    Computes the kernel function used in the convolution integral for the
    sorbed phase concentration under kinetic sorption conditions.

    Parameters
    ----------
    T : float
        Dimensionless time, T = vt/L.
    R : float
        Retardation factor, R = 1 + rho_b * Kd / theta.
    tau : ndarray
        Dimensionless integration time variable (τ), 0 <= τ <= T.
    R_s : float
        Retardation factor for kinetic sorption sites (R_s in CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at equilibrium (f in CXTFIT Table 3.1),
        0 <= f <= 1.
    beta : float
        Dimensionless partitioning coefficient (β in CXTFIT Table 3.1).
    beta_s : float
        Partitioning coefficient for the solid phase (β_s in CXTFIT Table 3.1).
    omega : float
        Dimensionless mass transfer coefficient (ω in CXTFIT Table 3.1).

    Returns
    -------
    ndarray
        Kernel values H_s(τ; T) over tau for numerical integration (Table 3.4).

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Table 3.4, kernel H_s for sorbed phase IVP convolution.
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