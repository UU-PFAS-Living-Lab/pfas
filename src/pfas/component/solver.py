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
import warnings
import numpy as np
from numpy.typing import NDArray
from pfas.solver_utils import compute_dimensionless_params
from pfas.solver_utils import (
    _BVP_FUNCTIONS,
    _H0,
    _IVP_FUNCTIONS,
    _bvp_neq,
    _Hs,
    _ivp_eq_flux,
    _ivp_neq,
)
from typing import Annotated
from pfas.data_structure import Adsorption, HydrologicalProperties
from annotated_types import Gt
from pydantic import BaseModel, field_validator, model_validator

class EquilibriumSolver(
    BaseModel, validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
):
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

    References
    ----------
    Toride, Leij & van Genuchten (1995), CXTFIT Version 2.0, Research Report
    No. 137, USDA-ARS. Section 2, eq. (2.20) for multiple rectangular pulses
    via step superposition.
    """

    grid: object
    hydro_properties: HydrologicalProperties
    adsorption: Adsorption
    boundary_conditions: object
    initial_contaminant_concentration: NDArray[np.float64] | None = None
    bc: str = "resident"

    @field_validator("bc")
    @classmethod
    def _validate_bc(cls, value: str) -> str:
        if value not in _BVP_FUNCTIONS:
            raise ValueError(
                f"Unknown boundary condition '{value}'. "
                f"Available options: {list(_BVP_FUNCTIONS.keys())}"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_initial_concentration(cls, values):
        initial_contaminant_concentration = values.get(
            "initial_contaminant_concentration"
        )
        grid = values.get("grid")
        hydro_properties = values.get("hydro_properties")

        if initial_contaminant_concentration is not None and grid is not None:
            if len(initial_contaminant_concentration) != len(grid.depth):
                raise ValueError(
                    "initial_contaminant_concentration must have the same "
                    "length as grid.depth"
                )

        if hydro_properties is not None:
            if hydro_properties.water_content < 0:
                raise ValueError(
                    "hydro_properties.water_content must not be negative"
                )

        return values

    def compute(self) -> dict[str, NDArray[np.float64]]:
        """Compute aqueous and total concentrations.

        Returns
        -------
        dict
            ``{"C1": C1, "C_tot": C_tot}``, each of shape ``(len(Z), len(T))``.
            ``C1`` is the aqueous phase concentration (mg/L); ``C_tot`` is the
            total concentration (mg/L bulk volume).

        Raises
        ------
        ValueError
            If ``len(boundary_conditions.C_list) != len(dim.T_list)``.
        """
        bvp_func = _BVP_FUNCTIONS[self.bc]
        ivp_func = _IVP_FUNCTIONS[self.bc]

        R = self.adsorption.total_retardation
        theta = self.hydro_properties.water_content
        C_list = self.boundary_conditions.C_list

        Ci = (
            self.initial_contaminant_concentration
            if self.initial_contaminant_concentration is not None
            else np.zeros(len(self.grid.depth))
        )

        dim = compute_dimensionless_params(
            self.grid,
            self.hydro_properties,
            T_list=self.boundary_conditions.T_list,
            adsorption=self.adsorption,
            kinetic=False,
        )
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

        return {"C1": C1, "C_tot": C_tot}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["C1", "C_tot"]


class KineticSolver(
    BaseModel, validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
):
    """Solve advection-dispersion equation with kinetic (time-dependent) sorption.

    When beta_s == 1 (no kinetic sites), delegates to an internal
    :class:`EquilibriumSolver`. Otherwise computes aqueous (C₁) and sorbed
    phase (C₂) concentrations for contaminant transport with first-order
    kinetic sorption. The total solution combines boundary value problem
    (BVP) and initial value problem (IVP) contributions:

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

    References
    ----------
    van Genuchten, M. Th. (1981). Non-Equilibrium Transport Parameters from
    Miscible Displacement Experiments. Research Report No. 119, USDA-ARS.

    Toride, Leij & van Genuchten (1995). CXTFIT Version 2.0. Research Report
    No. 137, USDA-ARS. Eqs. 3.6, 3.20–3.24; Tables 3.1, 3.4.

    Lindstrom, F.T. and Stone, W.J. (1974). Soil Sci. Soc. Am. Proc.
    """

    grid: object
    hydro_properties: HydrologicalProperties
    adsorption: Adsorption
    boundary_conditions: object
    bulk_density: Annotated[float, Gt(0)]
    initial_contaminant_concentration: NDArray[np.float64] | None = None
    volume_averaged: bool = True
    
    @model_validator(mode="before")
    @classmethod
    def validate_initial_concentration(cls, values):
        initial_contaminant_concentration = values.get(
            "initial_contaminant_concentration"
        )
        grid = values.get("grid")
        hydro_properties = values.get("hydro_properties")

        if initial_contaminant_concentration is not None and grid is not None:
            if len(initial_contaminant_concentration) != len(grid.depth):
                raise ValueError(
                    "initial_contaminant_concentration must have the same "
                    "length as grid.depth"
                )

        if hydro_properties is not None:
            if hydro_properties.water_content < 0:
                raise ValueError(
                    "hydro_properties.water_content must not be negative"
                )

        return values
    
    def compute(self) -> dict[str, NDArray[np.float64]]:
        """Compute aqueous, sorbed, and total concentrations.

        Returns
        -------
        dict
            ``{"C1": C1, "C2": C2, "C_tot": C_tot}``, each of shape
            ``(len(Z), len(T))``. ``C1`` is aqueous phase concentration
            (mg/L), ``C2`` is sorbed phase concentration (mg/kg), and
            ``C_tot`` is total concentration (mg/L bulk volume),
            C_tot = θ·β·R·C₁ + ρ_b·C₂ (CXTFIT eq. 3.6).
        """
        R = self.adsorption.total_retardation
        beta_s = self.adsorption.beta_s
        beta = self.adsorption.beta
        volume_averaged = self.volume_averaged
        R_s = self.adsorption.sp_retardation
        f = self.adsorption.frac_int
        Kd = self.adsorption.Kd
        theta = self.hydro_properties.water_content
        rho_b = self.bulk_density
        C_list = self.boundary_conditions.C_list

        Ci = (
            self.initial_contaminant_concentration
            if self.initial_contaminant_concentration is not None
            else np.zeros(len(self.grid.depth))
        )

        # When there are no kinetic sites (beta_s == 1), delegate to the
        # equilibrium solver.
        if beta_s == 1:
            warnings.warn(
                "beta_s == 1 (no kinetic sites). You have effectively selected the equilibrium sorption model." \
                "Using the EquilibriumSolver will be more efficient. " 
            )

        dim = compute_dimensionless_params(
            self.grid,
            self.hydro_properties,
            T_list=self.boundary_conditions.T_list,
            adsorption=self.adsorption,
            kinetic=True,
        )

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

        return {"C1": C1, "C2": C2, "C_tot": C_tot}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["C1", "C2", "C_tot"]

    