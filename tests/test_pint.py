import pytest
from pint import Quantity
from pint.errors import DimensionalityError

from pfas import ureg
from pfas.utils import (
    Kaw_langmuir_Le2021,
    Kaw_Szyszkowski,
    aaw_func_d50_Lyu,
    aaw_func_nonlinear_d50,
    aaw_func_thermo,
    kd_freundlich,
)


def test_thermodynamic_awi_returns_equivalent_values_with_and_without_units():
    unitless = aaw_func_thermo(
        sigma0=0.072,
        poro=0.4,
        alpha=0.8,
        n=1.6,
        th=0.3,
        thr=0.05,
        ths=0.4,
        sf=1.0,
        water_density=1000.0,
        gravity=9.81,
    )
    with_units = aaw_func_thermo(
        sigma0=72 * ureg.dyne / ureg.centimeter,
        poro=0.4,
        alpha=0.008 / ureg.centimeter,
        n=1.6,
        th=0.3,
        thr=0.05,
        ths=0.4,
        sf=1.0,
        water_density=1.0 * ureg.gram / ureg.centimeter**3,
        gravity=981 * ureg.centimeter / ureg.second**2,
    )

    assert isinstance(with_units, Quantity)
    assert with_units.check("1 / [length]")
    assert with_units.to("1 / meter").magnitude == pytest.approx(unitless)


@pytest.mark.parametrize(
    ("sigma0", "alpha"),
    [
        (0.072 * ureg.newton / ureg.meter, 0.8),
        (0.072, 0.8 / ureg.meter),
    ],
)
def test_thermodynamic_awi_rejects_mixed_unit_inputs(sigma0, alpha):
    with pytest.raises(ValueError, match="Mixed inputs"):
        aaw_func_thermo(
            sigma0=sigma0,
            poro=0.4,
            alpha=alpha,
            n=1.6,
            th=0.3,
            thr=0.05,
            ths=0.4,
            sf=1.0,
            water_density=1000.0,
            gravity=9.81,
        )


@pytest.mark.parametrize(
    "awi_function",
    [aaw_func_d50_Lyu, aaw_func_nonlinear_d50],
    ids=["linear-d50", "nonlinear-d50"],
)
def test_d50_awi_correlations_convert_quantity_units(awi_function):
    unitless = awi_function(d50=0.02, th=0.3, ths=0.4)
    with_units = awi_function(
        d50=0.2 * ureg.millimeter,
        th=0.3 * ureg.dimensionless,
        ths=0.4 * ureg.dimensionless,
    )

    assert isinstance(with_units, Quantity)
    assert with_units.check("1 / [length]")
    assert with_units.to("1 / centimeter").magnitude == pytest.approx(unitless)


@pytest.mark.parametrize(
    "awi_function",
    [aaw_func_d50_Lyu, aaw_func_nonlinear_d50],
    ids=["linear-d50", "nonlinear-d50"],
)
def test_d50_awi_correlations_reject_incompatible_d50_units(awi_function):
    with pytest.raises(DimensionalityError):
        awi_function(d50=1 * ureg.second, th=0.3, ths=0.4)


def test_freundlich_kd_returns_equivalent_values_with_and_without_units():
    unitless = kd_freundlich(C_rep=1.0, K_freund=0.8, n_freund=0.7)
    concentration_unit = ureg.mole / ureg.liter
    with_units = kd_freundlich(
        C_rep=1.0 * concentration_unit,
        K_freund=0.8 * ureg.liter / ureg.kilogram * concentration_unit**0.3,
        n_freund=0.7 * ureg.dimensionless,
    )

    assert isinstance(with_units, Quantity)
    assert with_units.check("[length] ** 3 / [mass]")
    assert with_units.to("liter / kilogram").magnitude == pytest.approx(unitless)


@pytest.mark.parametrize(
    ("c_rep", "k_freund"),
    [
        (1.0 * ureg.mole / ureg.liter, 0.8),
        (1.0, 0.8 * ureg.liter / ureg.kilogram),
    ],
)
def test_freundlich_kd_rejects_mixed_unit_inputs(c_rep, k_freund):
    with pytest.raises(TypeError, match="must either both be Pint quantities"):
        kd_freundlich(C_rep=c_rep, K_freund=k_freund, n_freund=0.7)



def test_le2021_langmuir_returns_equivalent_values_with_and_without_units():
    unitless = Kaw_langmuir_Le2021(
        Kaw_0=2.5,
        dG0=-10.0,
        Cw=1e-4,
        omega=55.3,
        T=298.0,
    )
    with_units = Kaw_langmuir_Le2021(
        Kaw_0=2.5,
        dG0=-10.0,
        Cw=0.1 * ureg.mole / ureg.meter**3,
        omega=55_300 * ureg.mole / ureg.meter**3,
        T=298.0 * ureg.kelvin,
    )

    assert isinstance(with_units, Quantity)
    assert with_units.check("[length]")
    assert with_units.to("centimeter").magnitude == pytest.approx(unitless)


@pytest.mark.parametrize(
    ("cw", "omega", "temperature"),
    [
        (1e-4 * ureg.mole / ureg.liter, 55.3, 298.0),
        (1e-4, 55.3 * ureg.mole / ureg.liter, 298.0),
        (1e-4, 55.3, 298.0 * ureg.kelvin),
    ],
)
def test_le2021_langmuir_rejects_mixed_unit_inputs(cw, omega, temperature):
    with pytest.raises(ValueError, match="Mixed inputs"):
        Kaw_langmuir_Le2021(
            Kaw_0=2.5,
            dG0=-10.0,
            Cw=cw,
            omega=omega,
            T=temperature,
        )


def test_szyszkowski_returns_equivalent_values_with_and_without_units():
    unitless = Kaw_Szyszkowski(
        sigma0=0.072,
        a=5e-4,
        b=1.2,
        Cw=1e-4,
        chi=2.0,
        T=298.0,
    )
    with_units = Kaw_Szyszkowski(
        sigma0=72 * ureg.dyne / ureg.centimeter,
        a=0.5 * ureg.mole / ureg.meter**3,
        b=1.2,
        Cw=0.1 * ureg.mole / ureg.meter**3,
        chi=2.0,
        T=298.0 * ureg.kelvin,
    )

    assert isinstance(with_units, Quantity)
    assert with_units.check("[length]")
    assert with_units.to("meter").magnitude == pytest.approx(unitless)


@pytest.mark.parametrize(
    ("sigma0", "a", "cw", "temperature"),
    [
        (0.072 * ureg.newton / ureg.meter, 5e-4, 1e-4, 298.0),
        (0.072, 5e-4 * ureg.mole / ureg.liter, 1e-4, 298.0),
        (0.072, 5e-4, 1e-4 * ureg.mole / ureg.liter, 298.0),
        (0.072, 5e-4, 1e-4, 298.0 * ureg.kelvin),
    ],
)
def test_szyszkowski_rejects_mixed_unit_inputs(sigma0, a, cw, temperature):
    with pytest.raises(ValueError, match="Mixed inputs"):
        Kaw_Szyszkowski(
            sigma0=sigma0,
            a=a,
            b=1.2,
            Cw=cw,
            chi=2.0,
            T=temperature,
        )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            Kaw_langmuir_Le2021,
            {
                "Kaw_0": 2.5,
                "dG0": -10.0,
                "Cw": 1.0 * ureg.meter,
                "omega": 55.3 * ureg.mole / ureg.liter,
                "T": 298.0 * ureg.kelvin,
            },
        ),
        (
            Kaw_Szyszkowski,
            {
                "sigma0": 0.072 * ureg.newton / ureg.meter,
                "a": 5e-4 * ureg.mole / ureg.liter,
                "b": 1.2,
                "Cw": 1e-4 * ureg.mole / ureg.liter,
                "chi": 2.0,
                "T": 298.0 * ureg.meter,
            },
        ),
    ],
    ids=["langmuir-concentration", "szyszkowski-temperature"],
)
def test_kaw_correlations_reject_incompatible_units(function, kwargs):
    with pytest.raises(DimensionalityError):
        function(**kwargs)
