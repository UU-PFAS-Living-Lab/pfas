"""
Analytical solvers for advection-dispersion equation with sorption.

This module provides analytical solutions for contaminant transport through porous media
using the advection-dispersion equation (ADE). It supports both equilibrium and kinetic
sorption models.

The module includes:

- **Equilibrium sorption**: Instantaneous sorption equilibrium using complementary error
  functions and exponential solutions.
- **Kinetic sorption**: Time-dependent sorption kinetics using modified Bessel function
  convolution kernels.
- **Two problem types**:
  - Boundary value problems (BVP): Constant boundary concentration input (e.g., pulse or step)
  - Initial value problems (IVP): Non-zero initial conditions in the domain

Key dimensionless parameters used throughout:
    - T: Dimensionless time
    - R: Retardation factor (accounts for sorption)
    - Z: Dimensionless depth
    - pec: Peclet number (ratio of advection to dispersion)

Main solver functions:
    - equilibrium_solver(): ADE with equilibrium sorption
    - kinetic_solver(): ADE with kinetic sorption
"""

import numpy as np
from scipy.special import erfc, iv

from pfas import utils
from pfas.solver_utils import DimensionlessParams

def eqbvpfunc(T, R, Z, pec):
    """
    Complementary error function solution for equilibrium boundary value problem.

    Computes the dimensionless concentration profile for the boundary value problem
    with constant boundary concentration using complementary error functions and
    exponential terms.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    Z : float or ndarray
        Dimensionless depth.
    pec : float
        Peclet number (ratio of advection to dispersion).

    Returns
    -------
    float or ndarray
        Dimensionless concentration at specified T, R, Z, pec.
    """
    return (
        0.5 * erfc((R * Z - T) / (2 * (T * R / pec) ** (1 / 2)))
        + ((T * pec) / (np.pi * R)) ** (1 / 2) * np.exp(-((R * Z - T) ** 2) / (4 * T * R / pec))
        - (1/2) * (1 + pec * Z + pec * T / R)
        * np.exp(pec * Z)
        * erfc((R * Z + T) / (2 * (T * R / pec) ** (1 / 2)))
    )

def eqivpfunc(T, R, Z, pec, kesi):
    """
    Complementary error function solution for equilibrium initial value problem.

    Computes the dimensionless concentration profile for the initial value problem
    with distributed initial concentration using complementary error functions.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    Z : float or ndarray
        Dimensionless depth.
    pec : float
        Peclet number.
    kesi : float or ndarray
        Dimensionless depth coordinate for initial concentration profile.

    Returns
    -------
    float or ndarray
        Dimensionless concentration at specified parameters.
    """
    return (
        (np.exp(-((R * Z - R * kesi - T) ** 2) / (4 * T * R / pec))
            + np.exp(-pec * kesi - (R * Z + R * kesi - T) ** 2 / (4 * T * R / pec)))
        / (2 * np.sqrt(np.pi * T / pec / R))
        - pec / 2 * np.exp(pec * Z)
        * erfc((R * Z + R * kesi + T) / (2 * np.sqrt(T * R / pec)))
    )


def neqivpfunc(T, R, Z, pec, kesi, beta): #noqa: PLR0913
    """
    Solution for non-equilibrium initial value problem with kinetic sorption.

    Compute the dimensionless concentration profile for the initial value problem
    accounting for kinetic sorption effects with a retardation coefficient beta.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor for aqueous phase.
    Z : float or ndarray
        Dimensionless depth.
    pec : float
        Peclet number.
    kesi : float or ndarray
        Dimensionless depth coordinate for initial concentration profile.
    beta : float
        Retardation factor related to kinetic sorption kinetics.

    Returns
    -------
    float or ndarray
        Dimensionless concentration at specified parameters.
    """
    return (
        (np.exp(-pec * beta * R * (Z - kesi - T / (beta * R)) ** 2 / (4 * T))
            + np.exp(-kesi * pec - pec * beta * R * (Z + kesi - T / (beta * R)) ** 2 / (4 * T)))
        / (2 * np.sqrt(np.pi * T / (beta * R * pec)))
        - pec / 2 * np.exp(pec * Z)
        * erfc((Z + kesi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / pec)))
    )

def hfunc(T, R, tau, Rs, Fs, beta, betas, ws): #noqa: PLR0913
    """
    Compute modified Bessel function based kernel for kinetic sorption convolution.

    Computes the kernel function using modified Bessel functions for the convolution
    integral in kinetic sorption calculations.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    tau : float or ndarray
        Integration time variable.
    Rs : float
        Surface retardation factor.
    Fs : float
        Fraction of sorption sites kinetically controlled.
    beta : float
        Retardation factor related to kinetic sorption.
    betas : float
        Surface retardation factor.
    ws : float
        Damköhler number for kinetic sorption.

    Returns
    -------
    float or ndarray
        Kernel value for the convolution integral.
    """
    iv_arg_2 = (
        2 * ws / (1 - betas) / (1 + Rs)
        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
        / (beta * R))

    return (
        Rs * (1 - Fs) / (beta * R)
        * np.exp(
            -ws * (T - tau) / (1 - betas) / (1 + Rs)
            - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs))
        * (
            iv(0, iv_arg_2) + iv(1, iv_arg_2) * tau
            / np.sqrt(Rs * (1 - Fs) * (T - tau) * tau / (beta * R))
        )
    )


def hs2func(T, R, tau, Rs, Fs, beta, betas, ws): #noqa: PLR0913
    """
    Compute modified Bessel function kernel for sorbed phase kinetic convolution.

    Computes the kernel function for the sorbed phase concentration convolution
    integral in kinetic sorption calculations, related to hfunc but for sorbed phase.

    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor.
    tau : float or ndarray
        Integration time variable.
    Rs : float
        Surface retardation factor.
    Fs : float
        Fraction of sorption sites kinetically controlled.
    beta : float
        Retardation factor related to kinetic sorption.
    betas : float
        Surface retardation factor.
    ws : float
        Kinetic sorption rate coefficient.

    Returns
    -------
    float or ndarray
        Kernel value for the sorbed phase convolution integral.
    """
    iv_arg_2 = (
        2 * ws / (1 - betas) / (1 + Rs)
        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
        / (beta * R))

    return (
        np.exp(
            -ws * (T - tau) / (1 - betas) / (1 + Rs)
            - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs))
        * (
            iv(0, iv_arg_2)
            + np.sqrt(Rs * (1 - Fs) * (T - tau) / (beta * R) / tau)
            * iv(1, iv_arg_2)
        )
    )
def equilibrium_solver(R, dim: DimensionlessParams, C10, Ci, theta):  # noqa: PLR0913
    """
    Solve advection-dispersion equation with equilibrium sorption.

    Computes the aqueous and total concentrations for contaminant transport
    through porous media assuming instantaneous sorption equilibrium using
    analytical solutions to the advection-dispersion equation.

    The solution includes contributions from:
    - Boundary value problem (BVP): response to constant boundary condition
    - Initial value problem (IVP): response to non-zero initial conditions

    Parameters
    ----------
    R : float
        Retardation factor accounting for sorption equilibrium.
    dim : DimensionlessParams
        Dimensionless parameters computed by :func:`compute_dimensionless_params`.
        Uses `.Z` (dimensionless depth), `.T` (dimensionless time),
        `.P` (Péclet number), and `.T0` (dimensionless pulse duration).
    C10 : float
        Normalized constant boundary concentration during pulse.
    Ci : ndarray
        Normalized initial concentration profile with depth (length must equal Z).
    theta : float
        Volumetric water content.

    Returns
    -------
    C1 : ndarray
        Aqueous phase concentration (mg/L) with shape (len(Z), len(T)).
    C_tot : ndarray
        Total concentration (mg/L bulk volume) with shape (len(Z), len(T)).
    """
    Z, T, T0, P = dim.Z, dim.T, dim.T0, dim.P

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))

    for i in range(len(T)):
        if T[i] <= T0:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i], R, Z, P)
        else:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i], R, Z, P) - C10 * eqbvpfunc(T[i] - T0, R, Z, P)
        if max(Ci) != 0:
            for ti in range(len(T)):
                for zi in range(len(Z)):
                    kesi = np.linspace(0, 1, len(Ci))
                    C1_ivp[zi, ti] = np.trapz(eqivpfunc(T[ti], R, Z[zi], P, kesi) * Ci, kesi)

    C1 = C1_bvp + C1_ivp
    C_tot = C1 * R * theta

    return C1, C_tot


def kinetic_solver(R, dim: DimensionlessParams, C10, Ci, betas, beta, cflag, Rs, Fs, Kd, theta, rhob):  # noqa: PLR0913
    """
    Solve advection-dispersion equation with kinetic (time-dependent) sorption.

    Computes aqueous and sorbed phase concentrations for contaminant transport
    with time-dependent sorption kinetics using modified Bessel function solutions.
    Handles both boundary value problem (constant boundary concentration) and
    initial value problem (non-zero initial conditions).

    The solution accounts for:
    - Kinetic sorption with first-order kinetics
    - Multiple sorption domains (equilibrium and kinetic)
    - Distributed sorption sites

    Parameters
    ----------
    R : float
        Retardation factor for aqueous phase.
    dim : DimensionlessParams
        Dimensionless parameters computed by :func:`compute_dimensionless_params`.
        Uses `.Z` (dimensionless depth), `.T` (dimensionless time),
        `.P` (Péclet number), `.T0` (dimensionless pulse duration),
        and `.ws` (Damköhler number for kinetic sorption).
    C10 : float
        Normalized constant boundary concentration during pulse.
    Ci : ndarray
        Normalized initial concentration profile with depth.
    betas : float
        Kinetic sorption retardation factor for solid phase.
    beta : float
        Total kinetic sorption retardation factor.
    cflag : int
        Configuration flag for volume-averaged (1) concentrations.
    Rs : float
        Solid phase retardation factor.
    Fs : float
        Fraction of sorption sites that are kinetically controlled (0 to 1).
    Kd : float
        Distribution coefficient for sorption (L/kg).
    theta : float
        Volumetric water content.
    rhob : float
        Bulk density of soil (kg/L).

    Returns
    -------
    C1 : ndarray
        Aqueous phase concentration (mg/L) with shape (len(Z), len(T)).
    C2 : ndarray
        Sorbed phase concentration (mg/kg) with shape (len(Z), len(T)).
    C_tot : ndarray
        Total concentration (mg/L bulk volume) with shape (len(Z), len(T)).
    """
    Z, T, T0, P, ws = dim.Z, dim.T, dim.T0, dim.P, dim.ws

    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    C2_bvp = np.zeros((len(Z), len(T)))
    C2_ivp = np.zeros((len(Z), len(T)))
    m = 30  # number of modified Bessel function terms used

    for i in range(len(Z)):
        for j in range(len(T)):
            if T[j] <= T0:
                C1_bvp[i, j], C2_bvp[i, j] = utils.ABfunc(
                    Z[i], T[j], ws, betas, beta, P, R, Rs, m, cflag
                )
            elif T[j] > T0:
                C1_bvp[i, j], C2_bvp[i, j] = utils.ABfunc(
                    Z[i], T[j], ws, betas, beta, P, R, Rs, m, cflag
                )
                A, B = utils.ABfunc(Z[i], T[j] - T0, ws, betas, beta, P, R, Rs, m, cflag)
                C1_bvp[i, j] = C1_bvp[i, j] - A
                C2_bvp[i, j] = C2_bvp[i, j] - B

            if max(Ci) != 0:
                kesi = np.linspace(0, 1, len(Ci))
                tau = np.linspace(0, T[j], 100)

                GfuncT = np.trapz(neqivpfunc(T[j], R, Z[i], P, kesi, beta) * Ci, kesi)
                if betas == 1:
                    C1_ivp[i, j] = GfuncT
                else:
                    C1_ivp[i, j] = (
                        np.exp(-ws * T[j] * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs))
                        * GfuncT
                    )
                    C2_ivp[i, j] = (
                        (1 - Fs) * Kd * Ci[i] * np.exp(-ws * T[j] / (1 - betas) / (1 + Rs))
                    )
                    Gfunctau = np.zeros((len(tau), 1))
                    for k in range(1, len(tau) - 1):
                        Gfunctau[k] = np.trapz(
                            neqivpfunc(tau[k], R, Z[i], P, kesi, beta) * Ci, kesi
                        )
                    C1_ivp[i, j] = C1_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * np.trapz(
                        hfunc(T[j], R, tau[1:-1], Rs, Fs, beta, betas, ws) * Gfunctau[1:-1],
                        tau[1:-1],
                    )
                    C2_ivp[i, j] = C2_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * (
                        1 - Fs
                    ) * Kd * np.trapz(
                        hs2func(T[j], R, tau[1:-1], Rs, Fs, beta, betas, ws) * Gfunctau[1:-1],
                        tau[1:-1],
                    )

    C1_bvp = C10 * C1_bvp
    C2_bvp = (1 - Fs) * Kd * C10 * C2_bvp
    C1 = C1_bvp + C1_ivp
    C2 = C2_bvp + C2_ivp
    C_tot = C1 * beta * R * theta + rhob * C2

    return C1, C2, C_tot