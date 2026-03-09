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
    present, the initial value problem (IVP):

        C(Z,T) = C^B(Z,T) + C^I(Z,T)

    The BVP term is constructed by superimposing step inputs for each pulse
    interval in ``dim.pulses`` (CXTFIT eq. 2.20). For each interval
    (T_start, T_end), a forward step is switched on at T_start and a backward
    step is subtracted at T_end:

        C^B(Z,T) = sum over intervals of:
            H(T - T_start) * A1(Z, T - T_start)
          - H(T - T_end)   * A1(Z, T - T_end)

    where A1 is the step BVP solution (see ``_BVP_FUNCTIONS``) and H is the
    Heaviside function. This naturally handles:
    - Step input:       ``pulses = [(0, inf)]``
    - Pulse from zero:  ``pulses = [(0, T0)]``
    - Delayed pulse:    ``pulses = [(T_start, T_end)]``
    - Multiple pulses:  ``pulses = [(T1s, T1e), (T2s, T2e), ...]``

    The IVP term integrates the Green's function kernel over the initial
    concentration profile Ci(xi) (CXTFIT Table 2.2).

    Parameters
    ----------
    R : float
        Retardation factor, R = 1 + rho_b * Kd / theta.
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, and `.pulses`.
    C10 : float
        Normalized inlet concentration during active pulse periods
        (C0 in CXTFIT notation).
    Ci : ndarray of shape (n_depth,)
        Normalized initial concentration profile Ci(Z) (mg/L).
        Pass an array of zeros if there is no initial contamination.
    theta : float
        Volumetric water content (-).
    bc : str, optional
        Upper boundary condition type. Must be a key in ``_BVP_FUNCTIONS``
        in ``solver_utils.py``. Options: ``'flux'`` (default, third-type BC)
        or ``'resident'`` (first-type BC). Default is ``'flux'``.

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

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Section 2, eq. (2.14) for superposition of BVP and IVP;
    eq. (2.20) for pulse input via step superposition.
    """
    if bc not in _BVP_FUNCTIONS:
        raise ValueError(
            f"Unknown boundary condition '{bc}'. "
            f"Available options: {list(_BVP_FUNCTIONS.keys())}"
        )

    bvp_func = _BVP_FUNCTIONS[bc]
    Z, T, pulses, P = dim.Z, dim.T, dim.pulses, dim.P

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))

    for i, Ti in enumerate(T):
        # Pulse superposition over all intervals (CXTFIT eq. 2.20):
        # For each (T_start, T_end), add a forward step at T_start and
        # subtract a backward step at T_end (Heaviside superposition).
        for T_start, T_end in pulses:
            if Ti > T_start:
                C1_bvp[:, i] += C10 * bvp_func(Ti - T_start, R, Z, P)
            if T_end != np.inf and Ti > T_end:
                C1_bvp[:, i] -= C10 * bvp_func(Ti - T_end, R, Z, P)

    if max(Ci) != 0:
        xi = np.linspace(0, 1, len(Ci))
        for ti, Ti in enumerate(T):
            for zi, Zi in enumerate(Z):
                C1_ivp[zi, ti] = np.trapz(_ivp_eq(Ti, R, Zi, P, xi) * Ci, xi)

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
    (IVP) for non-zero initial conditions. Pulse superposition over multiple
    intervals follows the same Heaviside approach as :func:`equilibrium_solver`.

    Parameters
    ----------
    R : float
        Retardation factor for aqueous phase.
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, `.pulses`, and `.omega`.
    C10 : float
        Normalized inlet concentration during active pulse periods.
    Ci : ndarray of shape (n_depth,)
        Normalized initial concentration profile with depth.
    beta_s : float
        Partitioning coefficient for the solid phase (β_s, CXTFIT Table 3.1).
    beta : float
        Dimensionless partitioning coefficient (β, CXTFIT Table 3.1).
    cflag : bool
        If True, return volume-averaged concentrations.
    R_s : float
        Retardation factor for kinetic sorption sites (R_s, CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at equilibrium (f, CXTFIT Table 3.1).
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
    Z, T, pulses, P, omega = dim.Z, dim.T, dim.pulses, dim.P, dim.omega
    m = 30  # number of modified Bessel function terms

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    C2_bvp = np.zeros((len(Z), len(T)))
    C2_ivp = np.zeros((len(Z), len(T)))

    for i, Zi in enumerate(Z):
        for j, Tj in enumerate(T):

            # Pulse superposition over all intervals
            for T_start, T_end in pulses:
                if Tj > T_start:
                    A, B = utils.ABfunc(Zi, Tj - T_start, omega, beta_s, beta, P, R, R_s, m, cflag)
                    C1_bvp[i, j] += A
                    C2_bvp[i, j] += B
                if T_end != np.inf and Tj > T_end:
                    A, B = utils.ABfunc(Zi, Tj - T_end, omega, beta_s, beta, P, R, R_s, m, cflag)
                    C1_bvp[i, j] -= A
                    C2_bvp[i, j] -= B

            if max(Ci) != 0:
                xi = np.linspace(0, 1, len(Ci))
                tau = np.linspace(0, Tj, 100)

                GfuncT = np.trapz(_ivp_neq(Tj, R, Zi, P, xi, beta) * Ci, xi)

                if beta_s == 1:
                    C1_ivp[i, j] = GfuncT
                else:
                    C1_ivp[i, j] = (
                        np.exp(-omega * Tj * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s))
                        * GfuncT
                    )
                    C2_ivp[i, j] = (
                        (1 - f) * Kd * Ci[i]
                        * np.exp(-omega * Tj / (1 - beta_s) / (1 + R_s))
                    )
                    Gfunctau = np.zeros((len(tau), 1))
                    for k in range(1, len(tau) - 1):
                        Gfunctau[k] = np.trapz(
                            _ivp_neq(tau[k], R, Zi, P, xi, beta) * Ci, xi
                        )
                    C1_ivp[i, j] += omega / (1 - beta_s) / (1 + R_s) * np.trapz(
                        _kinetic_kernel_aqueous(Tj, R, tau[1:-1], R_s, f, beta, beta_s, omega)
                        * Gfunctau[1:-1],
                        tau[1:-1],
                    )
                    C2_ivp[i, j] += omega / (1 - beta_s) / (1 + R_s) * (1 - f) * Kd * np.trapz(
                        _kinetic_kernel_sorbed(Tj, R, tau[1:-1], R_s, f, beta, beta_s, omega)
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
    pulse_intervals: list[tuple[float, float]],
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
        Contaminant source boundary conditions. Must have
        `.contaminant_release_rate`.
    initial_contaminant_concentration : ndarray of shape (n_depth,)
        Initial aqueous concentration distribution in the domain (mg/L).
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` (m/s),
        `.dispersion_coefficient` (m²/s), and `.water_content` (-).
    adsorption : Adsorption
        Adsorption parameters. Must have `.total_retardation`, `.Kd`,
        `.sp_retardation`, `.frac_int`, `.beta`, and `.beta_s`. When
        kinetic=True, also requires `.rate_const`.
    pulse_intervals : list of (float, float)
        Inlet concentration on/off periods in physical time (s). Examples:
        - Continuous step:   ``[(0, np.inf)]``
        - Pulse from t=0:    ``[(0, 5000)]``
        - Delayed pulse:     ``[(2000, 5000)]``
        - Multiple pulses:   ``[(0, 1000), (3000, 5000)]``
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
    ValueError
        If any pulse interval has t_start >= t_end.
    """
    dim = compute_dimensionless_params(
        grid,
        boundary_conditions,
        hydro_properties,
        pulse_intervals=pulse_intervals,
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