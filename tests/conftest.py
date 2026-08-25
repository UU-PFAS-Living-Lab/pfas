import pytest
import numpy as np
from pfas.analytical_soln import (
    SimulationGrid,
    HydrologicalProperties,
    BoundaryConditions,
    Adsorption,
)
from pfas.solver_utils import compute_dimensionless_params


@pytest.fixture
def grid():
    """Simple spatial and temporal grid for solver tests."""
    depth = np.linspace(0, 1, 5)
    time = np.linspace(0, 10, 6)
    return SimulationGrid(depth, time)


@pytest.fixture
def hydro():
    """Hydrological properties with small but realistic values."""
    return HydrologicalProperties(
        water_content=0.3,
        pore_velocity=1e-4,
        dispersion_coefficient=1e-6,
    )


@pytest.fixture
def bc_constant():
    """Constant inlet concentration boundary condition."""
    return BoundaryConditions(C_list=[1.0], T_list=[0.0])


@pytest.fixture
def adsorption_equilibrium():
    """Adsorption parameters for equilibrium solver."""
    return Adsorption(
        Kd=0.0,
        rate_const=0.0,
        frac_int=1.0,
        sp_retardation=0.0,
        awi_retardation=0.0,
    )


@pytest.fixture
def adsorption_kinetic():
    """Adsorption parameters for kinetic solver."""
    return Adsorption(
        Kd=0.5,
        rate_const=0.1,
        frac_int=0.3,
        sp_retardation=1.0,
        awi_retardation=0.0,
    )


@pytest.fixture
def initial(grid):
    """Zero initial concentration profile."""
    return np.zeros_like(grid.depth)


@pytest.fixture
def dim_equilibrium(grid, hydro, bc_constant, adsorption_equilibrium):
    """Dimensionless parameters for equilibrium solver."""
    return compute_dimensionless_params(
        grid,
        hydro,
        T_list=bc_constant.T_list,
        adsorption=adsorption_equilibrium,
        kinetic=False,
    )


@pytest.fixture
def dim_kinetic(grid, hydro, bc_constant, adsorption_kinetic):
    """Dimensionless parameters for kinetic solver."""
    return compute_dimensionless_params(
        grid,
        hydro,
        T_list=bc_constant.T_list,
        adsorption=adsorption_kinetic,
        kinetic=True,
    )
