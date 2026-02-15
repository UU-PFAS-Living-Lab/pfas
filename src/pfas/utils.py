

"""Utility functions for PFAS analytical solver.

This module provides helper functions for kinetic sorption calculations,
air-water interface area estimation, and numerical integration support.
"""

import numpy as np
from scipy.special import erfc, iv


def ABfunc(Z, T, ws, betas, beta, P, R, Rs, m, cflag): #noqa: N802, PLR0913, PLR0912
    """Compute A and B functions for kinetic sorption transport equations.

    Calculates the A and B functions appearing in Eqs (16-17) of Guo et al (2022)
    for the advection-dispersion equation with kinetic sorption. Uses approximations
    of Goldstein's J functions with modified Bessel functions for efficiency.

    Parameters
    ----------
    Z : float or ndarray
        Dimensionless depth (Z = z/L).
    T : float or ndarray
        Dimensionless time (T = v*t/L).
    ws : float
        Damköhler number (ws = alphas*(1-betas)*(1+Rs)*L/v), characterizing
        the relative importance of kinetic sorption to advection.
    betas : float
        Dimensionless sorption parameter (betas = (1+Fs*Rs)/(1+Rs)).
    beta : float
        Combined sorption parameter (beta = (betas*(1+Rs)+Raw)/(1+Rs+Raw)).
    P : float
        Péclet number (P = v*L/D), ratio of advection to dispersion.
    R : float
        Total retardation factor (dimensionless).
    Rs : float
        Retardation factor associated with surface or solid-phase adsorption.
    m : int
        Number of modified Bessel function terms used in the approximation.
    cflag : int
        Concentration averaging flag:

        - 0: volume-averaged concentration
        - 1: flux-averaged concentration

    Returns
    -------
    A : float or ndarray
        A function value (Eq 18 of Guo et al 2022), representing the
        contribution from sorption equilibrium.
    B : float or ndarray
        B function value (Eq 19 of Guo et al 2022), representing the
        contribution from kinetic sorption kinetics.

    References
    ----------
    Guo et al. (2022). Advection-dispersion equation with kinetic sorption.
    Advances in Water Resources.
    """
    # number of cells for the numerical integration
    n = 1001
    # tau cannot be zero, so start from a very small number to avoid the boundaries
    tau = np.linspace(1E-6,T-1E-6,n)
    Jab = np.zeros(len(tau))
    Jba = np.zeros(len(tau))

    if cflag == 0:
        # volume-averaged concentration
        g = np.sqrt(P/(np.pi*beta*R*tau))*np.exp(-P*(beta*R*Z-tau)**2/(4*beta*R*tau))
        - 1/2*(P/(beta*R))*np.exp(P*Z)*erfc(np.sqrt(P/(4*beta*R*tau))*(beta*R*Z+tau))
    elif cflag == 1:
        # flux-averaged concentration
        g = Z/tau * np.sqrt(P*beta*R/(4*np.pi*tau)) * np.exp(-P*((beta*R*Z-tau)**2) /(4*beta*R*tau))

    # Approximating the Goldstein's J function
    if betas == 1:
        Jab = Jab*0 + 1
        Jba = Jba*0 + 1
    else:
        a = ws*tau/(beta*R)
        b = ws*(T-tau)/((1-betas)*(Rs+1))
        for i in range(n):
            if a[i] + b[i] > 10:
               Jab[i] = 1/2*erfc(np.sqrt(a[i]) - np.sqrt(b[i])
                                 - 1/8/np.sqrt(a[i]) - 1/8/np.sqrt(b[i]))
               Jba[i] = 1/2*erfc(np.sqrt(b[i])
                                 - np.sqrt(a[i]) - 1/8/np.sqrt(b[i]) - 1/8/np.sqrt(a[i]))
            else:
               Iab_sum = 0
               Iba_sum = 0
               if a[i] >= b[i]:
                   for j in range(m):
                       Iab_sum = Iab_sum + (b[i]/a[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   for j in range(1,m+1):
                       Iba_sum = Iba_sum + (b[i]/a[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   Jab[i] = np.exp(-a[i]-b[i])*Iab_sum
                   Jba[i] = 1 - np.exp(-a[i]-b[i])*Iba_sum
               else:
                   for j in range(1,m+1):
                       Iab_sum = Iab_sum + (a[i]/b[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   for j in range(m):
                       Iba_sum = Iba_sum + (a[i]/b[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   Jab[i] = 1 - np.exp(-a[i]-b[i])*Iab_sum
                   Jba[i] = np.exp(-a[i]-b[i])*Iba_sum
    A = np.trapezoid(g*Jab,tau)
    B = np.trapezoid(g*(1-Jba),tau)
    return A, B


def aaw_func_thermo(sigma0, poro, alpha, n, th, thr, ths, sf): #noqa: PLR0913
    """Compute air-water interfacial area using thermodynamic approach.

    Estimates the air-water interfacial area per unit volume of porous medium
    using thermodynamic relations based on capillary pressure and water content,
    following van Genuchten soil water retention characteristics.

    Parameters
    ----------
    sigma0 : float
        Surface tension of water (dyn/cm).
    poro : float
        Porosity of the porous medium (dimensionless, 0-1).
    alpha : float
        van Genuchten parameter (cm⁻¹), related to the air-entry pressure.
    n : float
        van Genuchten parameter (dimensionless), related to pore size distribution.
    th : float
        Current water content (dimensionless).
    thr : float
        Residual water content (dimensionless).
    ths : float
        Saturated water content (dimensionless).
    sf : float
        Scaling factor to correct the thermodynamic-based estimate (dimensionless).

    Returns
    -------
    Aaw : float
        Air-water interfacial area per unit volume (cm²/cm³).

    Notes
    -----
    The function integrates the capillary pressure curve over saturation range
    to estimate the interfacial area. Water properties are assumed:

    - Water density: 1000 kg/m³
    - Gravitational acceleration: 9.81 m/s²
    """
    rhow = 1000
    g = 9.81
    m = 1 - 1/n
    sr = thr/ths
    sw = np.linspace(th/ths,1,1000)
    def pc(sw):
        return (((1-sr)/(sw-sr))**(1/m) - 1)**(1/n)/alpha/100*rhow*g

    aaw = 10*np.trapezoid(poro/sigma0*pc(sw),sw)
    aaw = aaw*sf

    return aaw

def aaw_func_tracer(sw, x2, x1, x0):
    """Compute air-water interfacial area using empirical polynomial model.

    Estimates air-water interfacial area per unit volume using polynomial
    fitting coefficients derived from tracer experiments or pore-scale imaging.
    This approach provides a simplified, computationally efficient alternative
    to thermodynamic calculations.

    Parameters
    ----------
    sw : float or ndarray
        Water saturation (dimensionless, 0-1).
    x2 : float
        Quadratic coefficient of polynomial fit.
    x1 : float
        Linear coefficient of polynomial fit.
    x0 : float
        Constant coefficient (intercept) of polynomial fit.

    Returns
    -------
    Aaw : float or ndarray
        Air-water interfacial area per unit volume (cm²/cm³).

    Notes
    -----
    The polynomial model is: Aaw = x2*sw² + x1*sw + x0

    This function is useful when empirical coefficients have been determined
    from experimental data for a specific porous medium.
    """
    aaw = x2*sw**2 + x1*sw + x0
    return aaw


def kd_fabregat_palau(n_CFx, f_oc, f_silt_clay): #noqa: N802
    """Calculate distribution coefficient using Fabregat-Palau (2021) model.

    Computes the soil-water distribution coefficient (Kd) for PFAS compounds
    based on the number of perfluorinated carbons and soil organic carbon
    and silt-clay content.

    Parameters
    ----------
    n_CFx : int
        Number of perfluorinated carbons (CF2 groups) in the PFAS molecule.
    f_oc : float
        Fraction of organic carbon in soil (dimensionless, 0-1).
    f_silt_clay : float
        Fraction of silt and clay in soil (dimensionless, 0-1).

    Returns
    -------
    Kd : float
        Distribution coefficient (L/kg).

    References
    ----------
    Fabregat-Palau et al. (2021). Modelling the sorption behaviour of
    perfluoroalkyl acids in soils.
    """
    k_oc = k_oc_fabregat_palau2021(n_CFx)
    k_silt_clay = k_sc_fabregat_palau2021(n_CFx)
    Kd = k_oc * f_oc + k_silt_clay * f_silt_clay
    return Kd


def k_sc_fabregat_palau2021(n_CFx):
    """Calculate silt-clay sorption coefficient (Fabregat-Palau 2021).

    Parameters
    ----------
    n_CFx : int
        Number of perfluorinated carbons (CF2 groups).

    Returns
    -------
    k_sc : float
        Silt-clay sorption coefficient (L/kg).

    References
    ----------
    Fabregat-Palau et al. (2021). Modelling the sorption behaviour of
    perfluoroalkyl acids in soils.
    """
    k_sc = 10 ** (0.32 * n_CFx - 1.7)
    return k_sc


def k_oc_fabregat_palau2021(n_CFx):
    """Calculate organic carbon sorption coefficient (Fabregat-Palau 2021).

    Parameters
    ----------
    n_CFx : int
        Number of perfluorinated carbons (CF2 groups).

    Returns
    -------
    k_oc : float
        Organic carbon sorption coefficient (L/kg).

    References
    ----------
    Fabregat-Palau et al. (2021). Modelling the sorption behaviour of
    perfluoroalkyl acids in soils.
    """
    k_oc = 10 ** (0.41 * n_CFx - 0.7)
    return k_oc
