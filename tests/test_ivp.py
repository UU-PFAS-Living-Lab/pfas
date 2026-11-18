"""Test the initial value problem with equilibrium and kinetic solvers.

Exercise both solver pathways with unitless inputs and Pint quantities.
"""

import numpy as np
import pytest
from pint import Quantity

from pfas import ureg
from pfas.component import (
    BoundaryPreprocessor,
    EquilibriumSolver,
    GridGenerator,
    KineticSolver,
    LinearSPsorption,
    Retardation,
    SWCsorption,
    WaterPreprocessor,
)
from pfas.model import Model

BULK_DENS = 1.6
CONCENTRATION_UNIT = ureg.milligram / ureg.liter

SOLVER_CONFIGS = {
    "equilibrium": {
        "solver_cls": EquilibriumSolver,
        "kinetic_sorption": False,
        "frac_int": 0.8,
        "rate_const": 0.1,
        "retardation_range": (39, 50),
        "solver_kwargs": {"bc": "flux"},
        "retardation_kwargs": {},
        "grid_kwargs": {
            "domain_length": 60,
            "spatial_resolution": 0.5,
            "time_resolution": 20,
            "time_total": 10000,
        },
    },
    "kinetic": {
        "solver_cls": KineticSolver,
        "kinetic_sorption": True,
        "frac_int": 0.8,
        "rate_const": 0.1,
        "retardation_range": (39, 50),
        "solver_kwargs": {},
        "retardation_kwargs": {
            "kinetic": True,
            "kin_params": {"frac_int": 0.8, "rate_const": 0.1},
        },
        "grid_kwargs": {
            "domain_length": 10,
            "spatial_resolution": 1,
            "time_resolution": 200,
            "time_total": 1000,
        },
    },
}


@pytest.fixture(scope="module", params=list(SOLVER_CONFIGS), ids=list(SOLVER_CONFIGS))
def solver_config(request):
    return SOLVER_CONFIGS[request.param]


@pytest.fixture(scope="module", params=[False, True], ids=["unitless", "pint"])
def with_units(request):
    return request.param


@pytest.fixture(scope="module")
def simulation(solver_config, with_units):
    """Build and run a simulation for one solver configuration.

    Return the model and inputs needed by the tests.
    """
    model = Model()

    model.compute(
        GridGenerator,
        **solver_config["grid_kwargs"],
    )

    n_nodes = len(model.grid.depth)
    cutoff = n_nodes // 2

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 0.5
    if with_units:
        initial_concentration = initial_concentration * CONCENTRATION_UNIT

    van_genuchten_alpha = 0.019
    sigma0 = 71
    kaw = 0.1
    kd = 8
    bulk_density = BULK_DENS
    if with_units:
        van_genuchten_alpha /= ureg.meter
        sigma0 *= ureg.newton / ureg.meter
        kaw *= ureg.centimeter
        kd *= ureg.liter / ureg.kilogram
        bulk_density *= ureg.kilogram / ureg.liter

    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    model.compute(
        BoundaryPreprocessor,
        C_list=[0 * CONCENTRATION_UNIT if with_units else 0],
        T_list=[0],
    )

    sorption_solid = {
        "kinetic_sorption": solver_config["kinetic_sorption"],
        "sorption_isotherm": "linear",
        "kinetic": {
            "frac_int": solver_config["frac_int"],
            "rate_const": solver_config["rate_const"],
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": kd,
        },
    }

    model.compute(
        LinearSPsorption,
        sorption_solid=sorption_solid,
    )

    model.compute(
        SWCsorption,
        sigma0=sigma0,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=van_genuchten_alpha,
        water_density=1000 * ureg.kilogram / ureg.meter**3 if with_units else 1000,
        gravity=9.81 * ureg.meter / ureg.second**2 if with_units else 9.81,
    )

    model.compute(
        Retardation,
        Kaw=kaw,
        bulk_density=bulk_density,
        **solver_config["retardation_kwargs"],
    )

    model.compute(
        solver_config["solver_cls"],
        initial_contaminant_concentration=initial_concentration,
        **solver_config["solver_kwargs"],
    )

    return {
        "model": model,
        "initial_concentration": initial_concentration,
        "cutoff": cutoff,
    }


@pytest.fixture(scope="module")
def model(simulation):
    return simulation["model"]


@pytest.fixture(scope="module")
def initial_concentration(simulation):
    return simulation["initial_concentration"]


@pytest.fixture(scope="module")
def cutoff(simulation):
    return simulation["cutoff"]


@pytest.fixture(scope="module")
def z(model):
    return np.asarray(model.grid.depth)


@pytest.fixture(scope="module")
def t(model):
    return np.asarray(model.grid.time)


def test_generated_data(model):
    assert model.generated_data, "model.generated_data should not be empty"

    for key, value in model.generated_data.items():
        values = _magnitudes(value)
        if values.ndim > 0:
            assert values.size > 0, f"{key} array is empty"
            assert not np.isnan(values).all(), f"{key} array is entirely NaN"


def test_physical_grid(z, t):
    assert np.all(np.diff(z) > 0), "depth values must be strictly increasing"
    assert np.all(np.diff(t) >= 0), "time values must be non-decreasing"


def test_water_properties(model):
    hydro = model.generated_data["hydro_properties"]

    theta = hydro.water_content
    v = hydro.pore_velocity
    dispersion = hydro.dispersion_coefficient

    assert 0 < theta <= 0.34
    assert v > 0
    assert dispersion > 0

    # Independent calculation from WaterPreprocessor inputs
    expected_v = 1.5 / theta
    expected_dispersion = expected_v * 1.5

    assert np.isclose(v, expected_v)
    assert np.isclose(dispersion, expected_dispersion)


def test_retardation(model, solver_config, with_units):
    adsorption = model.generated_data["adsorption"]
    retardation = adsorption.total_retardation
    if with_units:
        assert isinstance(retardation, Quantity)
        assert retardation.check("")
    else:
        assert isinstance(retardation, (int, float))
    retardation_value = float(_magnitudes(retardation))

    lo, hi = solver_config["retardation_range"]
    solver_name = solver_config["solver_cls"].__name__
    assert lo <= retardation_value <= hi, (
        f"R={retardation_value} is outside the expected range [{lo}, {hi}] "
        f"for {solver_name}"
    )


def test_initial_condition(initial_concentration, z):
    ci = _magnitudes(initial_concentration)

    assert ci.shape == z.shape
    assert np.isclose(ci.min(), 0.0)
    assert np.isclose(ci.max(), 0.5)

    contaminated = np.where(ci > 0)[0]
    assert len(contaminated) > 0, "expected a contaminated region in the initial condition"


def test_solution_shapes(model, z, t, with_units):
    c1 = model.generated_data["C1"]
    c_tot = model.generated_data["C_tot"]

    assert c1.shape == (len(z), len(t))
    assert c_tot.shape == c1.shape
    if with_units:
        assert isinstance(c1, Quantity)
        assert isinstance(c_tot, Quantity)
        assert c1.check("[mass] / [length] ** 3")
        assert c_tot.check("[mass] / [length] ** 3")
    else:
        assert isinstance(c1, np.ndarray)
        assert isinstance(c_tot, np.ndarray)


def test_negative_concentrations(model, z, t):
    """Report significantly negative concentrations without failing.
    """
    c1 = _magnitudes(model.generated_data["C1"])

    minimum = np.min(c1)
    print("\nminimum C =", minimum)

    negative_locations = np.where(c1 < -1e-12)
    n_negative = len(negative_locations[0])
    print("number of significantly negative values =", n_negative)

    if n_negative > 0:
        first = negative_locations[0][0]
        second = negative_locations[1][0]

        print("first negative value:", c1[first, second])
        print("at depth =", z[first], "cm")
        print("at time =", t[second], "s")


MASS_BALANCE_TOLERANCE = 1e-2


def test_mass_diagnostic(model, initial_concentration, z, t):
    c1 = _magnitudes(model.generated_data["C1"])
    ci = _magnitudes(initial_concentration)

    initial_mass = np.trapezoid(ci, z)
    assert initial_mass > 0, "initial mass must be positive for this test to be meaningful"

    print("\nInitial concentration integral:")
    print(" Ci dz =", initial_mass)

    masses = []
    for j in range(len(t)):
        mass = np.trapezoid(c1[:, j], z)
        masses.append(mass)

        assert np.isfinite(mass), f"Non-finite total mass at t={t[j]}"

        assert mass <= initial_mass * (1 + MASS_BALANCE_TOLERANCE), (
            f"Total mass at t={t[j]} ({mass}) exceeds the initial mass "
            f"({initial_mass}) beyond the allowed tolerance"
        )
        assert mass >= -initial_mass * MASS_BALANCE_TOLERANCE, (
            f"Total mass at t={t[j]} is significantly negative: {mass}"
        )

    for j in [0, 1, 2, 5, 10, len(t) // 2, -1]:
        if j >= len(t):
            continue

        print(
            f"t = {t[j]:8.1f} s | "
            f"\u222bC_tot dz = {masses[j]:.8g} | "
            f"ratio = {masses[j] / initial_mass:.8g}"
        )


def _magnitudes(value):
    """Return numeric values for assertions, whether unit-bearing or plain."""
    if isinstance(value, Quantity):
        return np.asarray(value.magnitude)
    return np.asarray(value)
