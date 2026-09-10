"""
Tests for the initial value problem for both
the EquilibriumSolver and KineticSolver 
sorption pathways.

Run with:
    pytest test_pfas_model.py -v -s
"""

import numpy as np
import pytest

from pfas.model import Model
from pfas.component import (
    LinearSPsorption,
    SWCsorption,
    Retardation,
    WaterPreprocessor,
    BoundaryPreprocessor,
    GridGenerator,
    EquilibriumSolver,
    KineticSolver,
)


BULK_DENS = 1.6

SOLVER_CONFIGS = {
    "equilibrium": {
        "solver_cls": EquilibriumSolver,
        "kinetic_sorption": False,
        "frac_int": 0.8,
        "rate_const": 0.1,
        "retardation_range": (40, 50),
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
        "retardation_range": (40, 50),
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


@pytest.fixture(scope="module")
def simulation(solver_config):
    """Build the grid, run the full model pipeline for the given solver
    configuration, and return the pieces needed by the tests below."""

    model = Model()

    model.compute(
        GridGenerator,
        **solver_config["grid_kwargs"],
    )

    n_nodes = len(model.grid.depth)
    cutoff = n_nodes // 2

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 0.5

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
        C_list=[0],
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
            "Kd": 8,
        },
    }

    model.compute(
        LinearSPsorption,
        sorption_solid=sorption_solid,
    )

    model.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019,
    )

    model.compute(
        Retardation,
        Kaw=10,
        bulk_density=BULK_DENS,
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
        if isinstance(value, np.ndarray):
            assert value.size > 0, f"{key} array is empty"
            assert not np.isnan(value).all(), f"{key} array is entirely NaN"


def test_physical_grid(z, t):
    assert np.all(np.diff(z) > 0), "depth values must be strictly increasing"
    assert np.all(np.diff(t) >= 0), "time values must be non-decreasing"


def test_water_properties(model):
    hydro = model.generated_data["hydro_properties"]

    theta = hydro.water_content
    v = hydro.pore_velocity
    D = hydro.dispersion_coefficient

    assert 0 < theta <= 0.34
    assert v > 0
    assert D > 0

    # Independent calculation from WaterPreprocessor inputs
    expected_v = 1.5 / theta
    expected_D = expected_v * 1.5

    assert np.isclose(v, expected_v)
    assert np.isclose(D, expected_D)


def test_retardation(model, solver_config):
    adsorption = model.generated_data["adsorption"]
    R = adsorption.total_retardation

    lo, hi = solver_config["retardation_range"]
    solver_name = solver_config["solver_cls"].__name__
    assert lo <= R <= hi, (
        f"R={R} is outside the expected range [{lo}, {hi}] for {solver_name}"
    )


def test_initial_condition(initial_concentration, z):
    Ci = np.asarray(initial_concentration)

    assert Ci.shape == z.shape
    assert np.isclose(Ci.min(), 0.0)
    assert np.isclose(Ci.max(), 0.5)

    contaminated = np.where(Ci > 0)[0]
    assert len(contaminated) > 0, "expected a contaminated region in the initial condition"


def test_solution_shapes(model, z, t):
    C1 = model.generated_data["C1"]
    C_tot = model.generated_data["C_tot"]

    assert C1.shape == (len(z), len(t))
    assert C_tot.shape == C1.shape


def test_negative_concentrations(model, z, t):
    """
    Reports any significantly negative concentration values without
    failing the test, mirroring the original exploratory diagnostic.
    """
    C1 = model.generated_data["C1"]

    minimum = np.min(C1)
    print("\nminimum C =", minimum)

    negative_locations = np.where(C1 < -1e-12)
    n_negative = len(negative_locations[0])
    print("number of significantly negative values =", n_negative)

    if n_negative > 0:
        first = negative_locations[0][0]
        second = negative_locations[1][0]

        print("first negative value:", C1[first, second])
        print("at depth =", z[first], "cm")
        print("at time =", t[second], "s")


MASS_BALANCE_TOLERANCE = 1e-2


def test_mass_diagnostic(model, initial_concentration, z, t):
    C1 = model.generated_data["C1"]
    Ci = np.asarray(initial_concentration)

    initial_mass = np.trapezoid(Ci, z)
    assert initial_mass > 0, "initial mass must be positive for this test to be meaningful"

    print("\nInitial concentration integral:")
    print(" Ci dz =", initial_mass)

    masses = []
    for j in range(len(t)):
        mass = np.trapezoid(C1[:, j], z)
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