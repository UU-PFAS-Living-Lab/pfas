

"""Utility functions for PFAS analytical solver.

This module provides helper functions for kinetic sorption calculations,
air-water interface area estimation, and numerical integration support.
"""

import numpy as np
from pfas import ureg 
from pint import Quantity

def _uses_units(**named) -> bool:
    """True if all given inputs are Quantities, False if none are.
 
    Raises if they are mixed, because a half-unitless call is almost
    certainly a mistake.
    """
    flags = {isinstance(v, Quantity) for v in named.values()}
    if len(flags) > 1:
        with_units = [k for k, v in named.items() if isinstance(v, Quantity)]
        without = [k for k, v in named.items() if not isinstance(v, Quantity)]
        raise ValueError(
            f"Mixed inputs: {with_units} have units but {without} do not. "
            "Pass either all Quantities or all plain numbers."
        )
    return flags.pop()

def aaw_func_thermo(  # noqa: PLR0913, PLR0917
    sigma0: Quantity | float,
    poro: float,
    alpha: Quantity | float,
    n: float,
    th: float,
    thr: float,
    ths: float,
    sf: float,
    water_density: Quantity | float,
    gravity: Quantity | float,
) -> Quantity | float:
    """Compute air-water interfacial area using the thermodynamic approach.
 
    Aaw(Sw) = (poro / sigma0) * integral from Sw to 1 of Pc(S) dS, with Pc from
    the van Genuchten retention curve.
 
    Parameters
    ----------
    sigma0, alpha, water_density, gravity
        Either all pint Quantities or all plain numbers. Plain numbers must be
        in one consistent unit system (e.g. SI), since nothing can be checked.
    poro, n, th, thr, ths, sf : float
        Dimensionless.
 
    Returns
    -------
    Quantity or float
        A Quantity in SI base units (1/m) if the inputs had units, otherwise a
        float in the inverse of the length unit implied by the inputs.
    """
    with_units = _uses_units(
        sigma0=sigma0, alpha=alpha
    )
    m = 1 - 1 / n
    sr = thr / ths

    sw = np.linspace(th/ths,1,1000)

    def pc(saturation: np.ndarray) -> Quantity:
        return (
            (
                ((1 - sr) / (saturation - sr)) ** (1 / m) - 1
            ) ** (1 / n)
            / alpha
            * water_density
            * gravity
        )
    
    aaw = poro / sigma0 * np.trapezoid(pc(sw), sw) * sf
    return aaw.to_base_units() if with_units else aaw

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
        Air-water interfacial area per unit volume. 

    Notes
    -----
    The polynomial model is: Aaw = x2*sw² + x1*sw + x0

    This function is useful when empirical coefficients have been determined
    from experimental data for a specific porous medium.
    """
    aaw = x2*sw**2 + x1*sw + x0
    return aaw

def aaw_func_GSSA(d50, poro, th=None, ths=None, sw=None): # noqa: N802
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
        Air-water interfacial area per unit volume. 
    Notes
    -----
    N/A

    """
    if sw is None:
        sw = th / ths

    aaw = (1 - sw) * (6 * (1 - poro) / d50)

    return aaw


def aaw_func_d50(d50, th=None, ths=None, sw=None):
    """Compute air-water interfacial area using the d50 correlation.

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
    """Compute air-water interfacial area from grain diameter and water saturation.

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
    return Kd * ureg.liter / ureg.kilogram


def k_sc_fabregat_palau2021(n_CFx):
    """Calculate silt-clay sorption coefficient (Fabregat-Palau 2021).

    Parameters
    ----------
    n_CFx : int
        Number of perfluorinated carbons (CF2 groups).

    Returns
    -------
    k_sc : float
        Silt-clay sorption coefficient (L/kg silt + clay).

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
        Organic carbon sorption coefficient (L/kg organic carbon).

    References
    ----------
    Fabregat-Palau et al. (2021). Modelling the sorption behaviour of
    perfluoroalkyl acids in soils.
    """
    k_oc = 10 ** (0.41 * n_CFx - 0.7)
    return k_oc


def kd_freundlich(
    C_rep,
    K_freund,
    n_freund,
):  # noqa: N802
    """Calculate Kd at a representative concentration.

    Kd = K_freund * C_rep**(n_freund - 1)

    The function accepts either:
    - Pint quantities for unit-aware calculations; or
    - plain numerical values, assuming consistent units.

    Parameters
    ----------
    C_rep : float or pint.Quantity
        Representative aqueous concentration.
    K_freund : float or pint.Quantity
        Freundlich capacity coefficient.
    n_freund : float
        Freundlich exponent.

    Returns
    -------
    float or pint.Quantity
        Distribution coefficient Kd.
    """
    if isinstance(n_freund, Quantity):
        n_freund = n_freund.to("").magnitude

    n_freund = float(n_freund)

    if isinstance(C_rep, Quantity) != isinstance(K_freund, Quantity):
        raise TypeError(
            "C_rep and K_freund must either both be Pint quantities "
            "or both be plain numerical values."
        )

    if isinstance(C_rep, Quantity):
        if C_rep.magnitude <= 0:
            raise ValueError("C_rep must be greater than zero.")

        Kd = K_freund * C_rep ** (n_freund - 1)
        return Kd.to("liter / kilogram")

    C_rep = float(C_rep)
    K_freund = float(K_freund)

    if C_rep <= 0:
        raise ValueError("C_rep must be greater than zero.")

    return K_freund * C_rep ** (n_freund - 1)

#Kaw formule van Le et al. (2021):
def Kaw_0_Le2021(  # noqa: N802
    structural_properties: dict[str, float | int],
) -> float:
    """
    Calculate the dilute-limit air-water partition coefficient using Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary containing PFAS structural-group counts.

    Returns
    -------
    float
        Dilute-limit air-water partition coefficient in the implicit unit
        specified by ``_KAW_0_LE2021_UNIT``.
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

    intercept = -5.19
    cfx = 0.60
    chx = 0.36
    coo = -2.42
    cooh = -0.47
    so3 = -2.35
    r4n = -4.30
    oh = -0.79
    oso3 = -2.39
    oxygen = -0.41
    sulfur = -0.21
    n_ch3_2_ch2_coo = -1.07

    log10_kaw_0 = (
        intercept
        + cfx * n_CFx
        + chx * n_CHx
        + coo * n_COO
        + cooh * n_COOH
        + so3 * n_SO3
        + r4n * n_R4N
        + oh * n_OH
        + oso3 * n_OSO3
        + oxygen * n__O_
        + sulfur * n__S_
        + n_ch3_2_ch2_coo * n_N_CH3_2_CH2_COO
    )

    return 10**log10_kaw_0


# dG0 formula from Le et al. (2021).
def dG0_Le2021(  # noqa: N802
    structural_properties: dict[str, float | int],
) -> float:
    """
    Calculate the Gibbs free energy of adsorption using Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary containing PFAS structural-group counts.

    Returns
    -------
    float
        Gibbs free energy of adsorption in kJ/mol.
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

    intercept = -14.29
    cfx = -3.57
    chx = -2.07
    coo = 11.56
    cooh = 0.34
    so3 = 11.48
    r4n = 22.06
    oh = 4.22
    oso3 = 10.78
    oxygen = 1.91
    sulfur = 1.79
    n_ch3_2_ch2_coo = 3.42

    return (
        intercept
        + cfx * n_CFx
        + chx * n_CHx
        + coo * n_COO
        + cooh * n_COOH
        + so3 * n_SO3
        + r4n * n_R4N
        + oh * n_OH
        + oso3 * n_OSO3
        + oxygen * n__O_
        + sulfur * n__S_
        + n_ch3_2_ch2_coo * n_N_CH3_2_CH2_COO
    )


def Kaw_langmuir_Le2021(  # noqa: N802
    *,
    Kaw_0: float,
    dG0: float,
    Cw: Quantity | float,
    omega: Quantity | float,
    T: Quantity | float,
) -> Quantity | float:
    """
    Calculate the concentration-dependent air-water partition coefficient.

    Implements the Langmuir-based approach of Le et al. (2021).

    Parameters
    ----------
    Kaw_0 : float
        Dilute-limit air-water partition coefficient calculated using
        ``Kaw_0_Le2021``. Its unit is defined internally by
        ``_KAW_0_LE2021_UNIT``.

    dG0 : float
        Gibbs free energy of adsorption from ``dG0_Le2021``, in kJ/mol.

    Cw : float or pint.Quantity
        Aqueous PFAS concentration. Plain floats are interpreted as mol/L.

    omega : float or pint.Quantity
        Water molar concentration. Plain floats are interpreted as mol/L.

    T : float or pint.Quantity
        Temperature. Plain floats are interpreted as K.

    Returns
    -------
    float or pint.Quantity
        Concentration-dependent air-water partition coefficient.

    Notes
    -----
    ``Cw``, ``omega``, and ``T`` must either all be Pint quantities or all
    be plain floats. Mixed input is rejected by ``_uses_units``.
    """
    with_units = _uses_units(
        Cw=Cw,
        omega=omega,
        T=T,
    )

    if with_units:
        Cw_q = Cw.to("mole / liter")
        omega_q = omega.to("mole / liter")
        T_q = T.to("kelvin")

        kaw_0_q = Kaw_0 * ureg.centimeter
        dG0_q = dG0 * ureg.kilojoule / ureg.mole

        gas_constant = (
            0.008314462618
            * ureg.kilojoule
            / ureg.mole
            / ureg.kelvin
        )

        exponent = (-dG0_q / (gas_constant * T_q)).to("").magnitude
        Keq = np.exp(exponent) / omega_q

        return (kaw_0_q / (1 + Keq * Cw_q)).to(
            ureg.centimeter

        )

    # Float convention:
    # Cw and omega: mol/L; T: K; dG0: kJ/mol.
    gas_constant = 0.008314462618  # kJ/(mol K)
    Keq = np.exp(-dG0 / (gas_constant * T)) / omega

    return Kaw_0 / (1 + Keq * Cw)


def Kaw_Szyszkowski(  # noqa: N802, PLR0913, PLR0917
    *,
    sigma0: Quantity | float,
    a: Quantity | float,
    b: float,
    Cw: Quantity | float,
    chi: float,
    T: Quantity | float,
) -> Quantity | float:
    """
    Calculate the air-water partition coefficient using the Szyszkowski equation.

    Parameters
    ----------
    sigma0 : float or pint.Quantity
        Surface tension of PFAS-free water. Plain floats are interpreted
        as N/m.

    a : float or pint.Quantity
        Szyszkowski fitting parameter. Plain floats are interpreted as mol/L.

    b : float
        Dimensionless Szyszkowski fitting parameter.

    Cw : float or pint.Quantity
        Aqueous PFAS concentration. Plain floats are interpreted as mol/L.

    chi : float
        Dimensionless ionisation coefficient. Use 1 for nonionic PFAS or
        ionic PFAS with swamping electrolyte, and 2 for ionic PFAS without
        swamping electrolyte.

    T : float or pint.Quantity
        Temperature. Plain floats are interpreted as K.

    Returns
    -------
    float or pint.Quantity
        Air-water partition coefficient. Float calculations return a value
        in m; Pint calculations return a Quantity convertible to m.

    Notes
    -----
    ``sigma0``, ``a``, ``Cw``, and ``T`` must either all be Pint quantities
    or all be plain floats. Mixed input is rejected by ``_uses_units``.
    """
    with_units = _uses_units(
        sigma0=sigma0,
        a=a,
        Cw=Cw,
        T=T,
    )

    if with_units:
        sigma0_q = sigma0.to("newton / meter")
        a_q = a.to("mole / meter**3")
        Cw_q = Cw.to("mole / meter**3")
        T_q = T.to("kelvin")

        gas_constant = (
            8.314462618
            * ureg.joule
            / ureg.mole
            / ureg.kelvin
        )

        kaw = (sigma0_q * b) / (
            chi * gas_constant * T_q * (a_q + Cw_q)
        )

        return kaw.to("centimeter")

    # Float convention:
    # sigma0: N/m; a and Cw: mol/L; T: K; result: m.
    gas_constant = 8.314462618  # J/(mol K)

    a_mol_m3 = a * 1000.0
    Cw_mol_m3 = Cw * 1000.0

    return (sigma0 * b) / (
        chi * gas_constant * T * (a_mol_m3 + Cw_mol_m3)
    )