import numpy as np
from pfas.component import GridGenerator
from pfas.data_structure import SimulationGrid


def test_grid_generator_basic():
    """
    Test that GridGenerator.compute() returns a SimulationGrid with correct
    depth and time arrays.

    This verifies:
    - Correct midpoint-centered depth grid
    - Correct uniform time grid
    - Correct number of grid points based on resolution
    """
    gg = GridGenerator(
        domain_length=10.0,
        spatial_resolution=1.0,
        time_resolution=2.0,
        time_total=10.0,
    )

    out = gg.compute()
    assert "grid" in out
    grid = out["grid"]
    assert isinstance(grid, SimulationGrid)

    expected_depth = np.linspace(0.5, 9.5, 10)
    assert np.allclose(grid.depth, expected_depth)

    expected_time = np.linspace(0, 10.0, int(10.0 / 2.0 + 0.5))
    assert np.allclose(grid.time, expected_time)


def test_grid_generator_resolution_effect():
    """
    Test that changing spatial and temporal resolution changes the number
    of grid points accordingly.

    This verifies:
    - Correct discretization logic
    """
    gg = GridGenerator(
        domain_length=5.0,
        spatial_resolution=0.5,
        time_resolution=1.0,
        time_total=4.0,
    )
    grid = gg.compute()["grid"]

    assert len(grid.depth) == int(5.0 / 0.5)
    assert len(grid.time) == int(4.0 / 1.0 + 0.5)


def test_grid_generator_outputs_property():
    """
    Test that the outputs property returns the correct output key.

    This verifies:
    - Consistency with Model.compute() orchestration
    """
    gg = GridGenerator(
        domain_length=10.0,
        spatial_resolution=1.0,
        time_resolution=2.0,
        time_total=10.0,
    )
    assert gg.outputs == ["grid"]
