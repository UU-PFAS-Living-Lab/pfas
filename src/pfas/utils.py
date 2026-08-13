

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

def aaw_func_GSSA(d50, poro, th=None, ths=None, sw=None):
    """Compute air-water interfacial area using the GSSA-based linear model.

    Estimates the air-water interfacial area per unit bulk volume as a
    linear function of water saturation, assuming that the geometric
    smooth-surface specific solid surface area (GSSA) represents the
    maximum possible interfacial area.

    Parameters
    ----------
    th : float or ndarray
        Volumetric water content.
    ths : float
        Saturated volumetric water content.
    poro : float
        Porosity of the porous medium (dimensionless, 0-1).
    d50 : float
        Median grain diameter (cm).

    Returns
    -------
    Aaw : float or ndarray
        Air-water interfacial area per unit volume (cm²/cm³).

    Notes
    -----
    N/A

    """
    if sw is None:
        sw = th / ths

    aaw = (1 - sw) * (6 * (1 - poro) / d50)

    return aaw


def aaw_func_d50(d50, th=None, ths=None, sw=None):
    """ Compute air-water interfacial area using the d50 correlation.

    Estimates the air-water interfacial area per unit bulk volume as a
    linear function of water saturation, with the maximum interfacial
    area estimated from median grain diameter.

    Parameters
    ----------
    sw : float or ndarray
        Water saturation (dimensionless, 0-1).
    d50  : float
        Median grain diameter (cm)

    Returns
    -------
    Aaw : float or ndarray
        Air-water interfacial area per unit volume (cm²/cm³).

    Notes
    -----
    N/A
    
    """
    if sw is None:
        sw = th / ths

    aaw = (1 - sw) * 3.9 * d50**-1.2

    return aaw


def aaw_func_nonlinear_d50(d50, th=None, ths=None, sw=None):
    """Compute air-water interfacial area using a nonlinear grain-diameter
    approximation based on saturation.

    Parameters
    ----------
    sw : float or ndarray
        Water saturation (dimensionless, 0-1).
    d50  : float
        Median grain diameter (cm)

    Returns
    -------
    Aaw : float or ndarray
        Air-water interfacial area per unit volume (cm²/cm³).

    Notes
    -----
    N/A
    
    """
    if sw is None:
        sw = th / ths

    aaw = (-2.85 * sw + 3.6) * ((1 - sw) * 3.9 * d50**-1.2)

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

#Kaw formule van Le et al. (2021): 
def Kaw_0_Le2021(structural_properties):
    """Calculate low concentration air-water partitioning coefficient using Le et al. (2021) model.

    Parameters
    ----------
    structural_properties : dict
        Dictionary with the PFAS structural group counts.
    """
    n_CFx = structural_properties["n_CFx"]
    n_CHx = structural_properties["n_CHx"]
    n_COO = structural_properties["n_COO"]
    n_COOH = structural_properties["n_COOH"]
    n_SO3 = structural_properties["n_SO3"]
    n_R4N = structural_properties["n_R4N"]
    n_OH = structural_properties["n_OH"]
    n_OSO3 = structural_properties["n_OSO3"]
    n__O_ = structural_properties["n__O_"]
    n__S_ = structural_properties["n__S_"]
    n_N_CH3_2_CH2_COO = structural_properties["n_N_CH3_2_CH2_COO"]

    Intercept       = -5.19
    CFx             =  0.60
    CHx             =  0.36
    COO             = -2.42
    COOH            = -0.47
    SO3             = -2.35
    R4N             = -4.30
    OH              = -0.79
    OSO3            = -2.39
    _O_             = -0.41
    _S_             = -0.21
    N_CH3_2_CH2_COO = -1.07
    
    log10_Kaw_0 = (
        Intercept
        + CFx * n_CFx
        + CHx * n_CHx
        + COO * n_COO
        + COOH * n_COOH
        + SO3 * n_SO3
        + R4N * n_R4N
        + OH * n_OH
        + OSO3 * n_OSO3
        + _O_ * n__O_
        + _S_ * n__S_
        + N_CH3_2_CH2_COO * n_N_CH3_2_CH2_COO
    )
    Kaw_0 = 10 ** log10_Kaw_0
    return Kaw_0

#dG0 formule van Le et al. (2021): 
def dG0_Le2021(structural_properties):
    """Calculates the Gibbs free energy change of adsorption using Le et al. (2021) model.

    Parameters
    ----------
    structural_properties : dict
        Dictionary with the PFAS structural group counts.
    """
    n_CFx = structural_properties["n_CFx"]
    n_CHx = structural_properties["n_CHx"]
    n_COO = structural_properties["n_COO"]
    n_COOH = structural_properties["n_COOH"]
    n_SO3 = structural_properties["n_SO3"]
    n_R4N = structural_properties["n_R4N"]
    n_OH = structural_properties["n_OH"]
    n_OSO3 = structural_properties["n_OSO3"]
    n__O_ = structural_properties["n__O_"]
    n__S_ = structural_properties["n__S_"]
    n_N_CH3_2_CH2_COO = structural_properties["n_N_CH3_2_CH2_COO"]

    Intercept       = -14.29
    CFx             = -3.57
    CHx             = -2.07
    COO             =  11.56
    COOH            =  0.34
    SO3             =  11.48
    R4N             =  22.06
    OH              =  4.22
    OSO3            =  10.78
    _O_             =  1.91
    _S_             =  1.79
    N_CH3_2_CH2_COO =  3.42
    
    dG0 = (
        Intercept
        + CFx * n_CFx
        + CHx * n_CHx
        + COO * n_COO
        + COOH * n_COOH
        + SO3 * n_SO3
        + R4N * n_R4N
        + OH * n_OH
        + OSO3 * n_OSO3
        + _O_ * n__O_
        + _S_ * n__S_
        + N_CH3_2_CH2_COO * n_N_CH3_2_CH2_COO
    )
    return dG0


#dG0 formule van Le et al. (2021): 
def Kaw_langmuir_Le2021(Kaw_0, dG0, Cw):
    """Calculates the Gibbs free energy change of adsorption using Le et al. (2021) model.

    Computes the Gibbs free energy change of adsorption for PFAS compounds
    based on the number of perfluorinated carbons and the specific headgroup.

    Parameters
    ----------
    Gamma_max
        maximum surface excess
    Keq
        equilibrium adsorption constant
    Cw
        concentration 
    
    Returns
    -------
    dG0 : float
        Distribution coefficient ().

    References
    ----------
    Le et al. (2021). A group-contribution model for predicting the physicochemical
    behavior of PFAS components for understanding environmental fate.
    """

    omega = 55.3      # water molar concentration (mol/L) at 298K
    R     = 0.008314  # gas constant (kJ/mol/K)
    T     = 298       # temperature (K)
    
    Keq = (1/omega) * np.exp(-dG0/(R*T))
    
    Kaw = (Kaw_0)/(1 + Keq*Cw)

    return Kaw

def Kaw_Szyszkowski(sigma0, a, b, Cw, chi=2, T=298):
    """Calculate air-water partitioning coefficient using the Szyszkowski equation.

    Parameters
    ----------
    sigma0 : float
        Surface tension of PFAS-free water (dyn/cm).
    a : float
        Szyszkowski fitting parameter (mol/L).
    b : float
        Szyszkowski fitting parameter (dimensionless).
    Cw : float
        Aqueous PFAS concentration (mol/L).
    chi : int, optional
        Ionisation coefficient. Use 1 for nonionic PFAS or ionic PFAS
        with swamping electrolyte, and 2 for ionic PFAS without
        swamping electrolyte. Default is 2.
    T : float, optional
        Temperature (K). Default is 298 K.

    Returns
    -------
    Kaw : float
        Air-water interfacial adsorption coefficient (cm3/cm2), equivalent to cm.
    """

    R = 8.314e7  # dyn cm / mol / K

    # Convert mol/L to mol/cm3
    a_mol_cm3 = a / 1000
    Cw_mol_cm3 = Cw / 1000  

    Kaw = (sigma0 * b) / (
        chi * R * T * (a_mol_cm3 + Cw_mol_cm3)
    )

    return Kaw