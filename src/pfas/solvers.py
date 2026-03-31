"""High-level analytical solvers for contaminant transport.

This module provides the public solver interface. All mathematical primitives
(BVP/IVP helpers, kinetic kernels, dimensionless parameter computation) live
in ``solver_utils.py``. The solvers here are purely orchestration logic.

Main functions
--------------
- :func:`equilibrium_solver` — ADE with instantaneous sorption equilibrium
- :func:`kinetic_solver` — ADE with first-order kinetic sorption
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from pfas.solver_utils import (
    _BVP_FUNCTIONS,
    _H0,
    _IVP_FUNCTIONS,
    DimensionlessParams,
    _bvp_neq,
    _Hs,
    _ivp_eq_flux,
    _ivp_neq,
)


def equilibrium_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C_list: list[float],
    Ci: NDArray[np.float64],
    theta: float,
    bc: str = "resident",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve advection-dispersion equation with equilibrium sorption.

    Computes aqueous and total concentrations for PFAS transport
    through porous media assuming instantaneous sorption equilibrium, using
    analytical solutions to the ADE. The solution combines contributions from
    the boundary value problem (BVP) and, when non-zero initial conditions are
    present, the initial value problem (IVP):

        C(Z,T) = C^B(Z,T) + C^I(Z,T)

    The BVP term implements CXTFIT eq. 2.20 directly. Given a series of
    rectangular pulses with concentrations ``C_list = [f1, f2, ..., fn]``
    switching at times ``dim.T_list = [T1, T2, ..., Tn]`` (dimensionless),
    the concentration increments are:

        deltaC = np.diff([0] + C_list)   →  [f1-f0, f2-f1, ..., fn-f_{n-1}]

    and eq. 2.20 becomes:

        C^B(Z,T) = sum_{j=1}^{i} deltaC[j] * G1^E(Z, T - T_j; mu^E)

    where each term is only added when T > T_j (Heaviside). This naturally
    handles:
    - Step input:        C_list=[C0],  dim.T_list=[0]
    - Single pulse:      C_list=[C0, 0],     dim.T_list=[0, T_pulse_end]
    - Multiple pulses:   C_list=[f1,f2,...], dim.T_list=[T1, T2, ...]

    The IVP term integrates the Green's function kernel over the initial
    concentration profile Ci(xi) (CXTFIT Table 2.2).

    Parameters
    ----------
    R : float
        Retardation factor, R = 1 + rho_b * Kd / theta.
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, and `.T_list` (dimensionless switching times
        corresponding to each entry in ``C_list``).
    C_list : list of float
        Inlet concentrations for each interval (mg/L).
        ``C_list[j]`` is the concentration active from ``dim.T_list[j]``
        until ``dim.T_list[j+1]``. The last interval extends to infinity.
        Must have the same length as ``dim.T_list``.
    Ci : ndarray of shape (n_depth,)
        Normalised initial concentration profile Ci(Z) (mg/L).
        Pass an array of zeros if there is no initial contamination.
    theta : float
        Volumetric water content (-).
    bc : str, optional
        Upper boundary condition type. Must be a key in ``_BVP_FUNCTIONS``.
        Options: ``'flux'`` (third-type BC) or ``'resident'`` (first-type BC).
        Default is ``'resident'``.

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
    ValueError
        If ``len(C_list) != len(dim.T_list)``.

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Section 2, eq. (2.20) for multiple rectangular pulses
    via step superposition.
    """
    if bc not in _BVP_FUNCTIONS:
        raise ValueError(
            f"Unknown boundary condition '{bc}'. "
            f"Available options: {list(_BVP_FUNCTIONS.keys())}"
        )

    bvp_func = _BVP_FUNCTIONS[bc]
    ivp_func = _IVP_FUNCTIONS[bc]
    Z, T, P, T_list = dim.Z, dim.T, dim.P, dim.T_list

    if len(C_list) != len(T_list):
        raise ValueError(
            f"C_list (len={len(C_list)}) and dim.T_list (len={len(T_list)}) "
            "must have the same length."
        )

    # ------------------------------------------------------------------
    # BVP term (eq. 2.20)
    # ------------------------------------------------------------------
    # deltaC[j] = f_j - f_{j-1}  (prepend f_0 = 0, CXTFIT eq. 2.20)
    deltaC: NDArray[np.float64] = np.diff([0.0] + C_list)

    C1_bvp = np.zeros((len(Z), len(T)))

    # eq. 2.20:  C^B(Z,T) = sum_j  deltaC[j] * G1^E(Z, T-T_j)
    #            only for T > T_j  (Heaviside)
    for i, Ti in enumerate(T):
        for delta, Tj in zip(deltaC, T_list):
            if Ti > Tj:
                C1_bvp[:, i] += delta * bvp_func(Ti - Tj, R, Z, P)

    # ------------------------------------------------------------------
    # IVP term
    # ------------------------------------------------------------------
    C1_ivp = np.zeros((len(Z), len(T)))

    if max(Ci) != 0:
        xi: NDArray[np.float64] = np.linspace(0, 1, len(Ci), dtype=np.float64)
        for ti, Ti in enumerate(T):
            for zi, Zi in enumerate(Z):
                integrand = cast(
                    NDArray[np.float64],
                    ivp_func(Ti, R, Zi, P, xi) * Ci,
                )
                C1_ivp[zi, ti] = np.trapezoid(integrand, xi)

    C1 = C1_bvp + C1_ivp
    C_tot = C1 * R * theta

    return C1, C_tot

def kinetic_solver(  # noqa: PLR0913
    R: float,
    dim: DimensionlessParams,
    C_list: list[float],
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

    When beta_s == 1 (no kinetic sites), delegates to equilibrium_solver.
    Otherwise computes aqueous (C₁) and sorbed phase (C₂) concentrations for contaminant
    transport with first-order kinetic sorption. The total solution combines
    boundary value problem (BVP) and initial value problem (IVP) contributions:

        C(Z,T) = C^B(Z,T) + C^I(Z,T)

    **BVP term — CXTFIT eq. 3.20**

    Pulse superposition using eq. 2.20 over switching times ``dim.T_list``
    with concentration increments:

        deltaC = np.diff([0] + C_list)   →  [f1-f0, f2-f1, ..., fn-f_{n-1}]

        C^B_k(Z,T) = Σⱼ deltaC[j] · Aₖ(Z, T - T_list[j])   for T > T_list[j]

    where A₁ (aqueous, ``C1_bvp``) and A₂ (sorbed, ``C2_bvp``) are evaluated
    via :func:`_bvp_neq` (CXTFIT eqs. 3.21–3.22). The sorbed phase BVP is
    additionally scaled by ``(1-f) * Kd`` after the loop to convert from
    dimensionless to mg/kg units.

    **IVP term — CXTFIT eqs. 3.31, 3.32 / Table 3.4**

    When ``beta_s == 1`` (no kinetic sites), the IVP reduces to the equilibrium
    Green's function at current time T:

        C1_ivp = G(Z, T)

    When ``beta_s != 1``, the IVP splits into three contributions:

    1. Initial aqueous concentration contribution, modified by inter-phase mass
       transfer (CXTFIT eq. 3.23, first term):

        C1_ivp = exp( -ω·T·(1-f)·Rₛ / ((1-βₛ)·β·R·(1+Rₛ)) ) · G(Z, T)

    2. Initial sorbed concentration contribution (CXTFIT eq. 3.24, first term):

        C2_ivp = (1-f)·Kd·Cᵢ · exp( -ω·T / ((1-βₛ)·(1+Rₛ)) )

    3. Convolution integrals over intermediate times τ ∈ (0, T), using the
       H₀ and Hₛ kernels from CXTFIT Table 3.4 (eqs. 3.31-3.32, second terms):

        C1_ivp += ω/((1-βₛ)·(1+Rₛ)) · ∫₀ᵀ H₀(T,τ) · G(Z,τ) dτ
        C2_ivp += ω/((1-βₛ)·(1+Rₛ)) · (1-f)·Kd · ∫₀ᵀ Hₛ(T,τ) · G(Z,τ) dτ

    **Total concentration — CXTFIT eq. 3.6**

        C_tot = θ·β·R·C₁ + ρ_b·C₂

    Parameters
    ----------
    R : float
        Overall retardation factor, R = 1 + ρ_b·Kd/θ (CXTFIT Table 3.1).
    dim : DimensionlessParams
        Dimensionless parameters from :func:`compute_dimensionless_params`.
        Uses `.Z`, `.T`, `.P`, `.T_list` (dimensionless switching times),
        and `.omega` (ω).
    C_list : list of float
        Inlet concentrations for each interval (mg/L).
        ``C_list[j]`` is the concentration active from ``dim.T_list[j]``
        until ``dim.T_list[j+1]``. The last interval extends to infinity.
        Must have the same length as ``dim.T_list``.
    Ci : ndarray of shape (n_depth,)
        Normalised initial aqueous concentration profile with depth (mg/L).
        Pass an array of zeros if there is no initial contamination.
    beta_s : float
        Solid-phase partitioning coefficient β_s (CXTFIT Table 3.1).
    beta : float
        Dimensionless partitioning coefficient β (CXTFIT Table 3.1).
    volume_averaged : bool
        If True, use volume-averaged (resident) concentrations in the BVP
        kernel :func:`_FT`. If False, use flux-averaged concentrations.
    R_s : float
        Retardation factor for kinetic sorption sites (CXTFIT Table 3.1).
    f : float
        Fraction of sorption sites at instantaneous equilibrium (CXTFIT
        Table 3.1).
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

    Toride, Leij & van Genuchten (1995). CXTFIT Version 2.0. Research Report
    No. 137, USDA-ARS. Eqs. 3.6, 3.20–3.24; Tables 3.1, 3.4.

    Lindstrom, F.T. and Stone, W.J. (1974). Soil Sci. Soc. Am. Proc.
    """
    # When there are no kinetic sites (beta_s == 1), use equilibrium solver
    if beta_s == 1:
        bc = "resident" if volume_averaged else "flux"
        C1, C_tot = equilibrium_solver(R, dim, C_list, Ci, theta, bc)
        C2 = np.zeros_like(C1)  # No sorbed phase in kinetic sites
        return C1, C2, C_tot

    # Original kinetic solver logic for beta_s < 1
    Z, T, P, T_list = dim.Z, dim.T, dim.P, dim.T_list
    omega = dim.omega
    assert omega is not None, "omega must be set for kinetic sorption"

    # deltaC[j] = f_j - f_{j-1}  (prepend f_0 = 0, CXTFIT eq. 3.20)
    deltaC: NDArray[np.float64] = np.diff([0.0] + C_list)

    C1_bvp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C1_ivp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C2_bvp: NDArray[np.float64] = np.zeros((len(Z), len(T)))
    C2_ivp: NDArray[np.float64] = np.zeros((len(Z), len(T)))

    for i, Zi in enumerate(Z):
        for j, Tj in enumerate(T):

            # Pulse superposition (CXTFIT eq. 3.20):
            # C^B_k(Z,T) = sum_j deltaC[j] * A_k(Z, T - T_list[j])
            # only for T > T_list[j]  (Heaviside)
            for delta, Tk in zip(deltaC, T_list):
                if Tj > Tk:
                    A_eq, A_neq = _bvp_neq(
                        Zi, Tj - Tk, omega, beta_s, beta, P, R, R_s,
                        volume_averaged=volume_averaged,
                    )
                    C1_bvp[i, j] += delta * A_eq
                    C2_bvp[i, j] += delta * A_neq

            if max(Ci) != 0:
                xi: NDArray[np.float64] = np.linspace(0, 1, len(Ci), dtype=np.float64)
                tau = np.linspace(0, Tj, 100)

                integrand_ZT = cast(
                    NDArray[np.float64],
                    _ivp_neq(Tj, R, Zi, P, xi, beta) * Ci,
                )
                G_ZT = np.trapezoid(integrand_ZT, xi)

                if beta_s == 1:
                    integrand = cast(
                        NDArray[np.float64],
                        _ivp_eq_flux(Tj, R, Zi, P, xi) * Ci,
                    )
                    C1_ivp[i, j] = np.trapezoid(integrand, xi)

                    # no kinetic sorption phase
                    C2_ivp[i, j] = 0.0
                else:
                    C1_ivp[i, j] = (
                        np.exp(-omega * Tj * (1 - f) * R_s / (1 - beta_s) / (beta * R) / (1 + R_s))
                        * G_ZT
                    )
                    C2_ivp[i, j] = (
                        (1 - f) * Kd * Ci[i]
                        * np.exp(-omega * Tj / (1 - beta_s) / (1 + R_s))
                    )
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

    C1 = C1_bvp + C1_ivp
    C2 = cast(NDArray[np.float64], (1 - f) * Kd * C2_bvp) + C2_ivp
    C_tot = C1 * beta * R * theta + rho_b * C2

    return C1, C2, C_tot
