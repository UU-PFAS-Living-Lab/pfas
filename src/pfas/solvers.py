import numpy as np
from pfas import utils
from scipy.special import erfc, iv


def eqbvpfunc(T, R, Z, P):
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
    P : float
        Peclet number (ratio of advection to dispersion).
    
    Returns
    -------
    float or ndarray
        Dimensionless concentration at specified T, R, Z, P.
    """
    return (
        0.5 * erfc((R * Z - T) / (2 * (T * R / P) ** (1 / 2)))
        + ((T * P) / (np.pi * R)) ** (1 / 2) * np.exp(-((R * Z - T) ** 2) / (4 * T * R / P))
        - (1/2) * (1 + P * Z + P * T / R)
        * np.exp(P * Z)
        * erfc((R * Z + T) / (2 * (T * R / P) ** (1 / 2)))
    )

def eqivpfunc(T, R, Z, P, kesi):
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
    P : float
        Peclet number.
    kesi : float or ndarray
        Dimensionless depth coordinate for initial concentration profile.
    
    Returns
    -------
    float or ndarray
        Dimensionless concentration at specified parameters.
    """
    return (
        (np.exp(-((R * Z - R * kesi - T) ** 2) / (4 * T * R / P))
            + np.exp(-P * kesi - (R * Z + R * kesi - T) ** 2 / (4 * T * R / P)))
        / (2 * np.sqrt(np.pi * T / P / R))
        - P / 2 * np.exp(P * Z)
        * erfc((R * Z + R * kesi + T) / (2 * np.sqrt(T * R / P)))
    )


def neqivpfunc(T, R, Z, P, kesi, beta):
    """
    Solution for non-equilibrium initial value problem with kinetic sorption.
    
    Computes the dimensionless concentration profile for the initial value problem
    accounting for kinetic sorption effects with a retardation coefficient beta.
    
    Parameters
    ----------
    T : float
        Dimensionless time.
    R : float
        Retardation factor for aqueous phase.
    Z : float or ndarray
        Dimensionless depth.
    P : float
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
        (np.exp(-P * beta * R * (Z - kesi - T / (beta * R)) ** 2 / (4 * T))
            + np.exp(-kesi * P - P * beta * R * (Z + kesi - T / (beta * R)) ** 2 / (4 * T)))
        / (2 * np.sqrt(np.pi * T / (beta * R * P)))
        - P / 2 * np.exp(P * Z)
        * erfc((Z + kesi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / P)))
    )

def Hfunc(T, R, tau, Rs, Fs, beta, betas, ws):
    """
    Modified Bessel function based kernel for kinetic sorption convolution.
    
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
        Kinetic sorption rate coefficient.
    
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


def Hs2func(T, R, tau, Rs, Fs, beta, betas, ws):
    """
    Modified Bessel function kernel for sorbed phase kinetic convolution.
    
    Computes the kernel function for the sorbed phase concentration convolution
    integral in kinetic sorption calculations, related to Hfunc but for sorbed phase.
    
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

def equilibrium_solver(R, Z, T, P, T0, C10, Ci, theta):
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
    Z : ndarray
        Dimensionless depth nodes (0 to 1).
    T : ndarray
        Dimensionless time points.
    P : float
        Peclet number (ratio of advection to dispersion).
    T0 : float
        Dimensionless pulse duration of contaminant input.
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

    # Solution for the boundary value problem
    # Define the solution for a constant boundary condition as a function
    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    for i in range(len(T)):
        if T[i] <= T0:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i], R, Z, P)
        else:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i], R, Z, P) - C10 * eqbvpfunc(T[i] - T0, R, Z, P)
        if max(Ci) != 0:
            # Solution for the initial value problem
            for i in range(len(T)):
                for j in range(len(Z)):
                    kesi = np.linspace(0, 1, len(Ci))
                    C1_ivp[j, i] = np.trapz(eqivpfunc(T[i], R, Z[j], P, kesi) * Ci, kesi)
        C1 = C1_bvp + C1_ivp
        #C2 = C2_bvp + C2_ivp
        C_tot = C1*R*theta #+ rhob*C2 #TODO
    return C1, C_tot


def kinetic_solver(R, Z, T, P, T0, C10, Ci, ws, betas, beta, cflag, Rs, Fs, Kd, theta, rhob):
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
    Z : ndarray
        Dimensionless depth nodes (0 to 1).
    T : ndarray
        Dimensionless time points.
    P : float
        Peclet number.
    T0 : float
        Dimensionless pulse duration of contaminant input.
    C10 : float
        Normalized constant boundary concentration during pulse.
    Ci : ndarray
        Normalized initial concentration profile with depth.
    ws : float
        Kinetic sorption rate coefficient.
    betas : float
        Kinetic sorption retardation factor for solid phase.
    beta : float
        Total kinetic sorption retardation factor
    cflag : int
        Configuration flag for volume-averaged (1) concentrations?
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
    # Initialize solutions for the aqueous concentration for BVP and IVP problems
    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    # Initialize solutions for adsorbed concentration at the kinetic sorption domain
    C2_bvp = np.zeros((len(Z), len(T)))
    C2_ivp = np.zeros((len(Z), len(T)))
    m = 30  # number of modified bessel function terms used
    for i in range(len(Z)):
        for j in range(len(T)):
            # Solution for the boundary value problem
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
                # Solution for the initial value problem
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
                        Gfunctau[k] = np.trapz(neqivpfunc(tau[k], R, Z[i], P, kesi, beta) * Ci, kesi)
                    C1_ivp[i, j] = C1_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * np.trapz(
                        Hfunc(T[j], R, tau[1:-1], Rs, Fs, beta, betas, ws) * Gfunctau[1:-1],
                        tau[1:-1]
                    )
                    C2_ivp[i, j] = C2_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * (
                        1 - Fs
                    ) * Kd * np.trapz(Hs2func(T[j], R, tau[1:-1], Rs, Fs, beta, betas, ws)
                                      * Gfunctau[1:-1], tau[1:-1])

    # Convert dimensionless C1_bvp and C2_bvp to original dimensions
    C1_bvp = C10 * C1_bvp
    C2_bvp = (1 - Fs) * Kd * C10 * C2_bvp
    C1 = C1_bvp + C1_ivp
    C2 = C2_bvp + C2_ivp
    C_tot = C1*beta*R*theta + rhob*C2
    return C1, C2, C_tot
