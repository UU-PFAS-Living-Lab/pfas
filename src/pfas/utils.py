

"""Utility functions for PFAS analytical solver.

This module provides helper functions for kinetic sorption calculations,
air-water interface area estimation, and numerical integration support.
"""

import numpy as np


def aaw_func_thermo(sigma0, poro, alpha, n, th, thr, ths, sf):  # noqa: PLR0913, PLR0917
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

def kd_freundlich(C_rep, K_freund, n_freund):  # noqa: N802
    """Calculate distribution coefficient using the Freundlich sorption model.

    Computes the soil-water distribution coefficient (Kd) from a Freundlich
    isotherm, which describes non-linear sorption onto soil.

    The Freundlich isotherm is defined as:
        S = K_freund * C^n_freund
    which gives a concentration-dependent Kd:
        Kd = K_freund * C_rep^(n_freund - 1)

    For C_rep = 0, Kd reduces to K_freund (equivalent to C_rep = 1).

    Parameters
    ----------
    C_rep : float
        Representative aqueous-phase concentration [mg/L].
        If zero, Kd is returned as K_freund (C_rep = 1 assumed).
    K_freund : float
        Freundlich capacity coefficient [(mg/kg) / (mg/L)^n_freund].
    n_freund : float
        Freundlich exponent [-]. n_freund < 1 indicates favourable
        (concave) sorption; n_freund = 1 recovers linear (Kd) sorption.

    Returns
    -------
    Kd : float
        Concentration-dependent distribution coefficient [L/kg].

    """
    if C_rep == 0:
        return K_freund
    Kd = K_freund * C_rep ** (n_freund - 1)
    return Kd
