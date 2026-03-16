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

from typing import cast

import numpy as np
from numpy.typing import NDArray

from pfas.solver_utils import (
    _BVP_FUNCTIONS,
    _H0,
    DimensionlessParams,
    _bvp_neq,
    _Hs,
    _ivp_eq,
    _ivp_neq,
    compute_dimensionless_params,
)


def equilibrium_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C0: float,
    Ci: NDArray[np.float64],
    theta: float,
    bc: str = "flux",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve advection-dispersion equation with equilibrium sorption.

    Computes aqueous and total concentrations for PFAS transport
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
    C0 : float
        Normalised inlet concentration during active pulse periods (mg/L).
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
                C1_bvp[:, i] += C0 * bvp_func(Ti - T_start, R, Z, P)
            if T_end != np.inf and Ti > T_end:
                C1_bvp[:, i] -= C0 * bvp_func(Ti - T_end, R, Z, P)

    if max(Ci) != 0:
        # Use dtype=np.float64 so xi is NDArray[np.float64], not NDArray[floating[Any]]
        xi: NDArray[np.float64] = np.linspace(0, 1, len(Ci), dtype=np.float64)
        for ti, Ti in enumerate(T):
            for zi, Zi in enumerate(Z):
                integrand = cast(
                    NDArray[np.float64],
                    _ivp_eq(Ti, R, Zi, P, xi) * Ci,
                )
                C1_ivp[zi, ti] = np.trapezoid(integrand, xi)

    C1 = C1_bvp + C1_ivp
    C_tot = C1 * R * theta

    return C1, C_tot


def kinetic_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C0: float,
    Ci: NDArray[np.float64],
    beta_s: float,
    beta: float,
    volume_averaged: bool,
    R_s: float,
    f: float,
    Kd: float,
    theta: float,
    rho_b: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve advection-dispersion equation with kinetic (time-dependent) sorption.

    Computes aqueous (C₁) and sorbed phase (C₂) concentrations for contaminant
    transport with first-order kinetic sorption. The total solution combines
    boundary value problem (BVP) and initial value problem (IVP) contributions:

        C(Z,T) = C^B(Z,T) + C^I(Z,T)

    **BVP term — CXTFIT eq. 3.20**

    Pulse superposition using the Heaviside approach over each interval
    (T_start, T_end) in ``dim.pulses``:

        C^B(Z,T) = Σₖ [ H(T - Tₛ,ₖ)·A₁(Z, T-Tₛ,ₖ) - H(T - Tₑ,ₖ)·A₁(Z, T-Tₑ,ₖ) ]

    where A₁ (aqueous, ``C1_bvp``) and A₂ (sorbed, ``C2_bvp``) are evaluated
    via :func:`_bvp_neq` (CXTFIT eqs. 3.21–3.22). Dimensional scaling applied
    after the loop (CXTFIT eq. 3.20):

        C1_bvp ← C0 · C1_bvp
        C2_bvp ← (1-f) · Kd · C0 · C2_bvp

    **IVP term — CXTFIT eqs. 3.31, 3.32 / Table 3.4**

    When ``beta_s == 1`` (no kinetic sites), the IVP reduces to the equilibrium
    Green's function at current time T:

        C1_ivp = G(Z, T)

    When ``beta_s != 1``, the IVP splits into three contributions:

    1. Initial aqueous concentration contribution, modified by inter-phase mass
       transfer (CXTFIT eq. 3.23, first term). The exponential prefactor
       represents the fraction of the initial aqueous concentration remaining
       in the aqueous phase at time T as mass transfers to kinetic sorption sites:

        C1_ivp = exp( -ω·T·(1-f)·Rₛ / ((1-βₛ)·β·R·(1+Rₛ)) ) · G(Z, T)

    2. Initial sorbed concentration contribution, modified by inter-phase mass
       transfer (CXTFIT eq. 3.24, first term). The exponential prefactor
       represents the fraction of the initial sorbed concentration remaining
       at kinetic sorption sites at time T as mass transfers back to the
       aqueous phase:

        C2_ivp = (1-f)·Kd·Cᵢ · exp( -ω·T / ((1-βₛ)·(1+Rₛ)) )

    3. Convolution integrals over intermediate times τ ∈ (0, T), using the
       H₀ and Hₛ kernels from CXTFIT Table 3.4 (eqs. 3.31-3.32, second terms;
       derived from van Genuchten (1981) Appendix B):

        C1_ivp += ω/((1-βₛ)·(1+Rₛ)) · ∫₀ᵀ H₀(T,τ) · G(Z,τ) dτ
        C2_ivp += ω/((1-βₛ)·(1+Rₛ)) · (1-f)·Kd · ∫₀ᵀ Hₛ(T,τ) · G(Z,τ) dτ

       where G(Z,τ) = ∫₀¹ G_neq(Z, τ, ξ)·Cᵢ(ξ) dξ is pre-evaluated at 100
       equally spaced τ points via :func:`_ivp_neq` and the trapezoidal rule,
       with solutions for G(Z,τ) from Table 3.3.

    **Total concentration — CXTFIT eq. 3.6 / van Genuchten (1981) eq. 2**

        C_tot = θ·β·R·C₁ + ρ_b·C₂

    reflecting partitioning between the mobile aqueous phase (scaled by β·R·θ)
    and the kinetic sorbed phase (scaled by ρ_b).

    Parameters
    ----------
    R : float
        Overall retardation factor, R = 1 + ρ_b·Kd/θ (CXTFIT Table 3.1).
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, `.pulses`, and `.omega` (ω).
    C0 : float
        Normalised inlet concentration during active pulse periods (mg/L).
    Ci : ndarray of shape (n_depth,)
        Normalised initial aqueous concentration profile with depth (mg/L).
        Pass an array of zeros if there is no initial contamination.
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
        Set to 1 for the fully equilibrium case (no kinetic sites).
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
        Ratio of equilibrium sorption capacity to total sorption capacity.
    volume_averaged : bool
        If True, use volume-averaged (resident) concentrations in the BVP
        kernel :func:`_FT`. If False, use flux-averaged concentrations.
    R_s : float
        Retardation factor for kinetic sorption sites, R_s (CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at instantaneous equilibrium (CXTFIT
        Table 3.1). f=1 reduces to full equilibrium; f=0 to fully kinetic.
    Kd : float
        Linear distribution coefficient (L/kg).
    theta : float
        Volumetric water content (-).
    rho_b : float
        Bulk density of the porous medium ρ_b (kg/L).

    Returns
    -------
    C1 : ndarray of shape (len(Z), len(T))
        Aqueous phase concentration (mg/L).
    C2 : ndarray of shape (len(Z), len(T))
        Sorbed phase concentration (mg/kg).
    C_tot : ndarray of shape (len(Z), len(T))
        Total concentration (mg/L bulk volume),
        C_tot = θ·β·R·C₁ + ρ_b·C₂  (CXTFIT eq. 3.6).


    References
    ----------
    van Genuchten, M. Th. (1981). Non-Equilibrium Transport Parameters from
    Miscible Displacement Experiments. Research Report No. 119, USDA-ARS.
    Eqs. 2, 14; Appendix B (H₀, Hₛ kernel derivation).

    Toride, Leij & van Genuchten (1995). CXTFIT Version 2.0. Research Report
    No. 137, USDA-ARS. Eqs. 3.6, 3.20–3.24; Tables 3.1, 3.4.

    Lindstrom, F.T. and Stone, W.J. (1974). Soil Sci. Soc. Am. Proc.
    """
    Z, T, pulses, P = dim.Z, dim.T, dim.pulses, dim.P

    # omega is only defined for kinetic (non-equilibrium) sorption
    omega = dim.omega
    assert omega is not None, "omega must be set for kinetic sorption"

    C1_bvp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C1_ivp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C2_bvp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C2_ivp: NDArray[np.float64] = np.zeros((len(Z), len(T)))

    for i, Zi in enumerate(Z):
        for j, Tj in enumerate(T):

            # Pulse superposition over all intervals (CXTFIT eq. 3.20):
            # A₁ (equilibrium phase) and A₂ (nonequilibrium phase) BVP solutions
            # are evaluated via _bvp_neq and superimposed using the Heaviside approach.
            for T_start, T_end in pulses:
                if Tj > T_start:
                    A_eq, A_neq = _bvp_neq(
                        Zi, Tj - T_start, omega, beta_s, beta, P, R, R_s,
                        volume_averaged=volume_averaged,
                    )
                    C1_bvp[i, j] += A_eq
                    C2_bvp[i, j] += A_neq
                if T_end != np.inf and Tj > T_end:
                    A_eq, A_neq = _bvp_neq(
                        Zi, Tj - T_end, omega, beta_s, beta, P, R, R_s,
                        volume_averaged=volume_averaged,
                    )
                    C1_bvp[i, j] -= A_eq
                    C2_bvp[i, j] -= A_neq

            if max(Ci) != 0:
                # Use dtype=np.float64 so xi is NDArray[np.float64], not
                # NDArray[floating[Any]], satisfying _ivp_neq's parameter type.
                xi: NDArray[np.float64] = np.linspace(0, 1, len(Ci), dtype=np.float64)
                tau = np.linspace(0, Tj, 100)

                # G(Z, T): Green's function integral at current time T (CXTFIT Table 3.2)
                integrand_ZT = cast(
                    NDArray[np.float64],
                    _ivp_neq(Tj, R, Zi, P, xi, beta) * Ci,
                )
                G_ZT = np.trapezoid(integrand_ZT, xi)

                if beta_s == 1:
                    C1_ivp[i, j] = G_ZT
                else:
                    C1_ivp[i, j] = (
                        np.exp(-omega * Tj * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s))
                        * G_ZT
                    )
                    C2_ivp[i, j] = (
                        (1 - f) * Kd * Ci[i]
                        * np.exp(-omega * Tj / (1 - beta_s) / (1 + R_s))
                    )
                    # G(Z, τ): Green's function at intermediate times for convolution
                    G_Ztau = np.zeros(len(tau))
                    for k in range(1, len(tau) - 1):
                        integrand_tau = cast(
                            NDArray[np.float64],
                            _ivp_neq(tau[k], R, Zi, P, xi, beta) * Ci,
                        )
                        G_Ztau[k] = np.trapezoid(integrand_tau, xi)

                    C1_ivp[i, j] += omega / (1 - beta_s) / (1 + R_s) * np.trapezoid(
                        _H0(Tj, R, tau[1:-1], R_s, f, beta, beta_s, omega)
                        * G_Ztau[1:-1],
                        tau[1:-1],
                    )
                    C2_ivp[i, j] += (
                        omega / (1 - beta_s) / (1 + R_s) * (1 - f) * Kd * np.trapezoid(
                            _Hs(Tj, R, tau[1:-1], R_s, f, beta, beta_s, omega)
                            * G_Ztau[1:-1],
                            tau[1:-1],
                        )
                    )

    C1_bvp = cast(NDArray[np.float64], C0 * C1_bvp)
    C2_bvp = cast(NDArray[np.float64], (1 - f) * Kd * C0 * C2_bvp)
    C1 = C1_bvp + C1_ivp
    C2 = C2_bvp + C2_ivp
    C_tot = C1 * beta * R * theta + rho_b * C2

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
        If True, use volume-averaged (resident) concentrations in the kinetic
        BVP kernel. If False (default), use flux-averaged concentrations.
        Only used by the kinetic solver.

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
