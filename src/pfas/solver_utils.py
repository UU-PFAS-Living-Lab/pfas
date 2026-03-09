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
    ws : float or None
        Damköhler number for kinetic sorption (-).
        None when kinetic=False.
    """

    Z: NDArray[np.float64]
    T: NDArray[np.float64]
    T0: float
    P: float
    ws: float | None


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
        `.rate_const`, `.betas`, `.sp_retardation`.
    kinetic : bool, optional
        If True, also compute the Damköhler number `ws`. Default is False.

    Returns
    -------
    DimensionlessParams
        Named tuple containing Z, T, T0, P, and ws (None if kinetic=False).

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

    ws = None
    if kinetic:
        ws = (
            adsorption.rate_const
            * (1 - adsorption.betas)
            * (1 + adsorption.sp_retardation)
            * L
            / v
        )

    return DimensionlessParams(Z=Z, T=T, T0=T0, P=P, ws=ws)


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

    Analytical solution to the 1D ADE with a constant flux boundary condition
    at the inlet, evaluated over dimensionless depth Z. Corresponds to bc=2
    in van Genuchten & Alves (1982).

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    Z : ndarray
        Dimensionless depth.
    P : float
        Péclet number.

    Returns
    -------
    ndarray
        Dimensionless concentration profile at time T.
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

    Analytical solution to the 1D ADE with a constant resident concentration
    boundary condition at the inlet, evaluated over dimensionless depth Z.
    Corresponds to bc=1 in van Genuchten & Alves (1982).

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    Z : ndarray
        Dimensionless depth.
    P : float
        Péclet number.

    Returns
    -------
    ndarray
        Dimensionless concentration profile at time T.
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
    kesi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """IVP integrand for equilibrium sorption.

    Computes the concentration kernel for the initial value problem under
    equilibrium sorption, integrated over the initial condition profile.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    Z : float
        Dimensionless depth (scalar, single evaluation point).
    P : float
        Péclet number.
    kesi : ndarray
        Dimensionless depth coordinate for the initial concentration profile.

    Returns
    -------
    ndarray
        Kernel values over kesi for numerical integration.
    """
    return (
        (
            np.exp(-((R * Z - R * kesi - T) ** 2) / (4 * T * R / P))
            + np.exp(-P * kesi - (R * Z + R * kesi - T) ** 2 / (4 * T * R / P))
        )
        / (2 * np.sqrt(np.pi * T / P / R))
        - P / 2 * np.exp(P * Z) * erfc((R * Z + R * kesi + T) / (2 * np.sqrt(T * R / P)))
    )


# ---------------------------------------------------------------------------
# IVP and convolution helpers — kinetic sorption
# ---------------------------------------------------------------------------

def _ivp_neq(  # noqa: PLR0913
    T: float,
    R: float,
    Z: float,
    P: float,
    kesi: NDArray[np.float64],
    beta: float,
) -> NDArray[np.float64]:
    """IVP integrand for non-equilibrium (kinetic) sorption.

    Computes the concentration kernel for the initial value problem under
    kinetic sorption, integrated over the initial condition profile.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor for aqueous phase.
    Z : float
        Dimensionless depth (scalar).
    P : float
        Péclet number.
    kesi : ndarray
        Dimensionless depth coordinate for the initial concentration profile.
    beta : float
        Kinetic sorption retardation factor.

    Returns
    -------
    ndarray
        Kernel values over kesi for numerical integration.
    """
    return (
        (
            np.exp(-P * beta * R * (Z - kesi - T / (beta * R)) ** 2 / (4 * T))
            + np.exp(-kesi * P - P * beta * R * (Z + kesi - T / (beta * R)) ** 2 / (4 * T))
        )
        / (2 * np.sqrt(np.pi * T / (beta * R * P)))
        - P / 2 * np.exp(P * Z)
        * erfc((Z + kesi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / P)))
    )


def _kinetic_kernel_aqueous(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    Rs: float,
    Fs: float,
    beta: float,
    betas: float,
    ws: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel for aqueous phase kinetic sorption.

    Computes the kernel function used in the convolution integral for the
    aqueous phase concentration under kinetic sorption conditions.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    tau : ndarray
        Integration time variable.
    Rs : float
        Solid phase retardation factor.
    Fs : float
        Fraction of sorption sites kinetically controlled (0 to 1).
    beta : float
        Total kinetic sorption retardation factor.
    betas : float
        Kinetic sorption retardation factor for solid phase.
    ws : float
        Damköhler number for kinetic sorption.

    Returns
    -------
    ndarray
        Kernel values over tau for numerical integration.
    """
    iv_arg = (
        2 * ws / (1 - betas) / (1 + Rs)
        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
        / (beta * R)
    )

    return (
        Rs * (1 - Fs) / (beta * R)
        * np.exp(
            -ws * (T - tau) / (1 - betas) / (1 + Rs)
            - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs)
        )
        * (
            iv(0, iv_arg)
            + iv(1, iv_arg) * tau / np.sqrt(Rs * (1 - Fs) * (T - tau) * tau / (beta * R))
        )
    )


def _kinetic_kernel_sorbed(  # noqa: PLR0913
    T: float,
    R: float,
    tau: NDArray[np.float64],
    Rs: float,
    Fs: float,
    beta: float,
    betas: float,
    ws: float,
) -> NDArray[np.float64]:
    """Bessel function convolution kernel for sorbed phase kinetic sorption.

    Computes the kernel function used in the convolution integral for the
    sorbed phase concentration under kinetic sorption conditions.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    tau : ndarray
        Integration time variable.
    Rs : float
        Solid phase retardation factor.
    Fs : float
        Fraction of sorption sites kinetically controlled (0 to 1).
    beta : float
        Total kinetic sorption retardation factor.
    betas : float
        Kinetic sorption retardation factor for solid phase.
    ws : float
        Damköhler number for kinetic sorption.

    Returns
    -------
    ndarray
        Kernel values over tau for numerical integration.
    """
    iv_arg = (
        2 * ws / (1 - betas) / (1 + Rs)
        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
        / (beta * R)
    )

    return (
        np.exp(
            -ws * (T - tau) / (1 - betas) / (1 + Rs)
            - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs)
        )
        * (
            iv(0, iv_arg)
            + np.sqrt(Rs * (1 - Fs) * (T - tau) / (beta * R) / tau) * iv(1, iv_arg)
        )
    )