import numpy as np
from scipy.special import erfc, iv

from pfas_leach_screening import utils


def equilibrium_solver(R, Z, T, P, T0, C10, Ci, theta):
    """
    Kinetic solver and parameters.
    """
    # Solution for the boundary value problem
    # Define the solution for a constant boundary condition as a function
    C1_bvp = np.zeros((len(Z), len(T)))
    C1_ivp = np.zeros((len(Z), len(T)))
    eqbvpfunc = (
        lambda T: 0.5 * erfc((R * Z - T) / (2 * (T * R / P) ** (1 / 2)))
        + ((T * P) / (np.pi * R)) ** (1 / 2) * np.exp(-((R * Z - T) ** 2) / (4 * T * R / P))
        - 1
        / 2
        * (1 + P * Z + P * T / R)
        * np.exp(P * Z)
        * erfc((R * Z + T) / (2 * (T * R / P) ** (1 / 2)))
    )
    for i in range(len(T)):
        if T[i] <= T0:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i])
        else:
            C1_bvp[:, i] = C10 * eqbvpfunc(T[i]) - C10 * eqbvpfunc(T[i] - T0)
        if max(Ci) != 0:
            # Solution for the initial value problem
            for i in range(len(T)):
                for j in range(len(Z)):
                    kesi = np.linspace(0, 1, len(Ci))
                    eqivpfunc = lambda Z, T: (
                        np.exp(-((R * Z - R * kesi - T) ** 2) / (4 * T * R / P))
                        + np.exp(-P * kesi - (R * Z + R * kesi - T) ** 2 / (4 * T * R / P))
                    ) / (2 * np.sqrt(np.pi * T / P / R)) - P / 2 * np.exp(P * Z) * erfc(
                        (R * Z + R * kesi + T) / (2 * np.sqrt(T * R / P))
                    )
                    C1_ivp[j, i] = np.trapz(eqivpfunc(Z[j], T[i]) * Ci, kesi)
        C1 = C1_bvp + C1_ivp
        #C2 = C2_bvp + C2_ivp
        C_tot = C1*R*theta #+ rhob*C2 #TODO
    return C1, C_tot


def kinetic_solver(R, Z, T, P, T0, C10, Ci, ws, betas, beta, cflag, Rs, Fs, Kd, theta, rhob):
    """
    Kinetic solver its parameters.
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
                neqivpfunc = lambda Z, T: (
                    np.exp(-P * beta * R * (Z - kesi - T / (beta * R)) ** 2 / (4 * T))
                    + np.exp(-kesi * P - P * beta * R * (Z + kesi - T / (beta * R)) ** 2 / (4 * T))
                ) / (2 * np.sqrt(np.pi * T / (beta * R * P))) - P / 2 * np.exp(P * Z) * erfc(
                    (Z + kesi + T / (beta * R)) / (2 * np.sqrt(T / (beta * R) / P))
                )

                Hfunc = (
                    lambda T, tau: Rs
                    * (1 - Fs)
                    / (beta * R)
                    * np.exp(
                        -ws * (T - tau) / (1 - betas) / (1 + Rs)
                        - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs)
                    )
                    * (
                        iv(
                            0,
                            2
                            * ws
                            / (1 - betas)
                            / (1 + Rs)
                            * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
                            / (beta * R),
                        )
                        + iv(
                            1,
                            2
                            * ws
                            / (1 - betas)
                            / (1 + Rs)
                            * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
                            / (beta * R),
                        )
                        * tau
                        / np.sqrt(Rs * (1 - Fs) * (T - tau) * tau / (beta * R))
                    )
                )

                Hs2func = lambda T, tau: np.exp(
                    -ws * (T - tau) / (1 - betas) / (1 + Rs)
                    - ws * tau * (1 - Fs) * Rs / (1 - betas) / (beta * R) / (1 + Rs)
                ) * (
                    iv(
                        0,
                        2
                        * ws
                        / (1 - betas)
                        / (1 + Rs)
                        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
                        / (beta * R),
                    )
                    + np.sqrt(Rs * (1 - Fs) * (T - tau) / (beta * R) / tau)
                    * iv(
                        1,
                        2
                        * ws
                        / (1 - betas)
                        / (1 + Rs)
                        * np.sqrt(Rs * (1 - Fs) * (T - tau) * tau)
                        / (beta * R),
                    )
                )

                GfuncT = np.trapz(neqivpfunc(Z[i], T[j]) * Ci, kesi)
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
                        Gfunctau[k] = np.trapz(neqivpfunc(Z[i], tau[k]) * Ci, kesi)
                    C1_ivp[i, j] = C1_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * np.trapz(
                        Hfunc(T[j], tau[1:-1]) * Gfunctau[1:-1], tau[1:-1]
                    )
                    C2_ivp[i, j] = C2_ivp[i, j] + ws / (1 - betas) / (1 + Rs) * (
                        1 - Fs
                    ) * Kd * np.trapz(Hs2func(T[j], tau[1:-1]) * Gfunctau[1:-1], tau[1:-1])

    # Convert dimensionless C1_bvp and C2_bvp to original dimensions
    C1_bvp = C10 * C1_bvp
    C2_bvp = (1 - Fs) * Kd * C10 * C2_bvp
    C1 = C1_bvp + C1_ivp
    C2 = C2_bvp + C2_ivp
    C_tot = C1*beta*R*theta + rhob*C2
    return C1, C2, C_tot
