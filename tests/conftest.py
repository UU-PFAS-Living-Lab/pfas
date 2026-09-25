import pytest
import numpy as np
from pfas.data_structure import (
    SimulationGrid,
    HydrologicalProperties,
    BoundaryConditions,
    Adsorption,
)
from pfas.solver_utils import DimensionlessParams


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
    return np.arange(0, 1.01, 0.25)


@pytest.fixture
def dim_equilibrium(grid, hydro, bc_constant, adsorption_equilibrium):
    """Dimensionless parameters for equilibrium solver."""
    return DimensionlessParams(
        Z=np.array([0.  , 0.25, 0.5 , 0.75, 1.  ]),
        T=np.array([0.    , 0.0002, 0.0004, 0.0006, 0.0008, 0.001 ]),
        T_list=[np.float64(0.0)],
        P=np.float64(100.00000000000001),
        omega=None
    )

@pytest.fixture
def dim_kinetic(grid, hydro, bc_constant, adsorption_kinetic):
    """Dimensionless parameters for kinetic solver."""
    return DimensionlessParams(
            Z=np.array([0.  , 0.25, 0.5 , 0.75, 1.  ]),
            T=np.array([0.    , 0.0002, 0.0004, 0.0006, 0.0008, 0.001 ]),
            T_list=[np.float64(0.0)],
            P=np.float64(100.00000000000001),
            omega=np.float64(699.9999999999999)
    )
