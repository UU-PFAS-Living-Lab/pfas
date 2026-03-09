"""High-level analytical solvers for contaminant transport.

This module provides the public solver interface. All mathematical primitives
(BVP/IVP helpers, kinetic kernels, dimensionless parameter computation) live
in ``solver_utils.py``. The solvers here are purely orchestration logic.

Main functions
--------------
- :func:`equilibrium_solver` — ADE with instantaneous sorption equilibrium
- :func:`kinetic_solver` — ADE with first-order kinetic sorption
- :func:`analytical_soln` — top-level entry point dispatching to the above
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, iv

from pfas import utils
from pfas.solver_utils import (
    DimensionlessParams,
    _BVP_FUNCTIONS,
    _ivp_eq,
    _ivp_neq,
    _kinetic_kernel_aqueous,
    _kinetic_kernel_sorbed,
    compute_dimensionless_params,
)


def equilibrium_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C10: float,
    Ci: NDArray[np.float64],
    theta: float,
    bc: str = "flux",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve advection-dispersion equation with equilibrium sorption.

    Computes aqueous and total concentrations for contaminant transport
    through porous media assuming instantaneous sorption equilibrium, using
    analytical solutions to the ADE. The solution combines contributions from
    the boundary value problem (BVP) and, when non-zero initial conditions are
    present, the initial value problem (IVP).

    Parameters
    ----------
    R : float
        Retardation factor accounting for sorption equilibrium.
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, and `.T0`.
    C10 : float
        Normalized constant boundary concentration during pulse.
    Ci : ndarray of shape (n_depth,)
        Normalized initial concentration profile with depth.
        Pass an array of zeros if there is no initial contamination.
    theta : float
        Volumetric water content (-).
    bc : str, optional
        Upper boundary condition type. Must be a key in ``_BVP_FUNCTIONS``
        in ``solver_utils.py``. Options: ``'flux'`` (default) or
        ``'resident'``. Default is ``'flux'``.

    Returns
    -------
    C1 : ndarray of shape (len(Z), len(T))
        Aqueous phase concentration (mg/L).
    C_tot : ndarray of shape (len(Z), len(T))
        Total concentration (mg/L bulk volume).

    Raises
    ------
    ValueError
        If `bc` is not a recognised boundary condition type.
    """
    if bc not in _BVP_FUNCTIONS:
        raise ValueError(
            f"Unknown boundary condition '{bc}'. "
            f"Available options: {list(_BVP_FUNCTIONS.keys())}"
        )

    bvp_func = _BVP_FUNCTIONS[bc]
    Z, T, T0, P = dim.Z, dim.T, dim.T0, dim.P

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))

    for i in range(len(T)):
        if T[i] <= T0:
            C1_bvp[:, i] = C10 * bvp_func(T[i], R, Z, P)
        else:
            C1_bvp[:, i] = (
                C10 * bvp_func(T[i], R, Z, P)
                - C10 * bvp_func(T[i] - T0, R, Z, P)
            )

    if max(Ci) != 0:
        xi = np.linspace(0, 1, len(Ci))
        for ti in range(len(T)):
            for zi in range(len(Z)):
                C1_ivp[zi, ti] = np.trapz(
                    _ivp_eq(T[ti], R, Z[zi], P, xi) * Ci, xi
                )

    C1 = C1_bvp + C1_ivp
    C_tot = C1 * R * theta

    return C1, C_tot


def kinetic_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C10: float,
    Ci: NDArray[np.float64],
    beta_s: float,
    beta: float,
    cflag: bool,
    R_s: float,
    f: float,
    Kd: float,
    theta: float,
    rhob: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve advection-dispersion equation with kinetic (time-dependent) sorption.

    Computes aqueous and sorbed phase concentrations for contaminant transport
    with first-order kinetic sorption, using modified Bessel function solutions.
    Handles both the boundary value problem (BVP) and the initial value problem
    (IVP) for non-zero initial conditions.

    Parameters
    ----------
    R : float
        Retardation factor for aqueous phase.
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, `.T0`, and `.omega`.
    C10 : float
        Normalized constant boundary concentration during pulse.
    Ci : ndarray of shape (n_depth,)
        Normalized initial concentration profile with depth.
    beta_s : float
        Kinetic sorption retardation factor for solid phase.
    beta : float
        Total kinetic sorption retardation factor.
    cflag : bool
        If True, return volume-averaged concentrations.
    R_s : float
        Solid phase retardation factor.
    f : float
        Fraction of sorption sites kinetically controlled (0 to 1).
    Kd : float
        Distribution coefficient for sorption (L/kg).
    theta : float
        Volumetric water content (-).
    rhob : float
        Bulk density of soil (kg/L).

    Returns
    -------
    C1 : ndarray of shape (len(Z), len(T))
        Aqueous phase concentration (mg/L).
    C2 : ndarray of shape (len(Z), len(T))
        Sorbed phase concentration (mg/kg).
    C_tot : ndarray of shape (len(Z), len(T))
        Total concentration (mg/L bulk volume).
    """
    Z, T, T0, P, omega = dim.Z, dim.T, dim.T0, dim.P, dim.omega
    m = 30  # number of modified Bessel function terms

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    C2_bvp = np.zeros((len(Z), len(T)))
    C2_ivp = np.zeros((len(Z), len(T)))

    for i in range(len(Z)):
        for j in range(len(T)):
            if T[j] <= T0:
                C1_bvp[i, j], C2_bvp[i, j] = utils.ABfunc(
                    Z[i], T[j], omega, beta_s, beta, P, R, R_s, m, cflag
                )
            else:
                C1_bvp[i, j], C2_bvp[i, j] = utils.ABfunc(
                    Z[i], T[j], omega, beta_s, beta, P, R, R_s, m, cflag
                )
                A, B = utils.ABfunc(Z[i], T[j] - T0, omega, beta_s, beta, P, R, R_s, m, cflag)
                C1_bvp[i, j] -= A
                C2_bvp[i, j] -= B

            if max(Ci) != 0:
                xi = np.linspace(0, 1, len(Ci))
                tau = np.linspace(0, T[j], 100)

                GfuncT = np.trapz(_ivp_neq(T[j], R, Z[i], P, xi, beta) * Ci, xi)

                if beta_s == 1:
                    C1_ivp[i, j] = GfuncT
                else:
                    C1_ivp[i, j] = (
                        np.exp(-omega * T[j] * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s))
                        * GfuncT
                    )
                    C2_ivp[i, j] = (
                        (1 - f) * Kd * Ci[i]
                        * np.exp(-omega * T[j] / (1 - beta_s) / (1 + R_s))
                    )
                    Gfunctau = np.zeros((len(tau), 1))
                    for k in range(1, len(tau) - 1):
                        Gfunctau[k] = np.trapz(
                            _ivp_neq(tau[k], R, Z[i], P, xi, beta) * Ci, xi
                        )
                    C1_ivp[i, j] += omega / (1 - beta_s) / (1 + R_s) * np.trapz(
                        _kinetic_kernel_aqueous(T[j], R, tau[1:-1], R_s, f, beta, beta_s, omega)
                        * Gfunctau[1:-1],
                        tau[1:-1],
                    )
                    C2_ivp[i, j] += omega / (1 - beta_s) / (1 + R_s) * (1 - f) * Kd * np.trapz(
                        _kinetic_kernel_sorbed(T[j], R, tau[1:-1], R_s, f, beta, beta_s, omega)
                        * Gfunctau[1:-1],
                        tau[1:-1],
                    )

    C1_bvp = C10 * C1_bvp
    C2_bvp = (1 - f) * Kd * C10 * C2_bvp
    C1 = C1_bvp + C1_ivp
    C2 = C2_bvp + C2_ivp
    C_tot = C1 * beta * R * theta + rhob * C2

    return C1, C2, C_tot


def analytical_soln(  # noqa: PLR0913
    grid,
    bulk_density: float,
    boundary_conditions,
    initial_contaminant_concentration: NDArray[np.float64],
    hydro_properties,
    adsorption,
    kinetic: bool = False,
    volume_averaged: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64]]:
    """Solve contaminant transport using analytical solutions.

    Computes aqueous and sorbed phase concentrations for PFAS transport through
    the vadose zone using analytical solutions to the advection-dispersion equation
    (ADE) with retardation. Dimensionless parameters are computed via
    :func:`compute_dimensionless_params` and passed to the appropriate solver.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization grid. Must have `.depth` (m)
        and `.time` (s) arrays.
    bulk_density : float
        Bulk density of the porous medium (kg/L).
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions. Must have `.pulse_time` (s)
        and `.contaminant_release_rate`.
    initial_contaminant_concentration : ndarray of shape (n_depth,)
        Initial aqueous concentration distribution in the domain (mg/L).
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` (m/s),
        `.dispersion_coefficient` (m²/s), and `.water_content` (-).
    adsorption : Adsorption
        Adsorption parameters. Must have `.total_retardation`, `.Kd`,
        `.sp_retardation`, `.frac_int`, `.beta`, and `.beta_s`. When
        kinetic=True, also requires `.rate_const`.
    kinetic : bool, optional
        If True, use the kinetic sorption model (:func:`kinetic_solver`),
        which returns a separate sorbed phase C2. If False (default), use
        the equilibrium model (:func:`equilibrium_solver`), and C2 is None.
    volume_averaged : bool, optional
        If True, return volume-averaged concentrations. Only used by the
        kinetic solver. Default is False.

    Returns
    -------
    C1 : ndarray
        Aqueous phase concentration (mg/L).
    C2 : ndarray or None
        Sorbed phase concentration (mg/kg). None when kinetic=False.
    C_tot : ndarray
        Total concentration (mg/L bulk volume).

    Raises
    ------
    ValueError
        If pore_velocity or dispersion_coefficient is zero.
    """
    dim = compute_dimensionless_params(
        grid,
        boundary_conditions,
        hydro_properties,
        adsorption=adsorption,
        kinetic=kinetic,
    )

    C2 = None

    if kinetic:
        C1, C2, C_tot = kinetic_solver(
            adsorption.total_retardation,
            dim,
            boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration,
            adsorption.beta_s,
            adsorption.beta,
            volume_averaged,
            adsorption.sp_retardation,
            adsorption.frac_int,
            adsorption.Kd,
            hydro_properties.water_content,
            bulk_density,
        )
    else:
        C1, C_tot = equilibrium_solver(
            adsorption.total_retardation,
            dim,
            boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration,
            hydro_properties.water_content,
        )

    return C1, C2, C_tot