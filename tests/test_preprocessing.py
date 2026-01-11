import numpy as np

from pfas.analytical_soln import SimulationGrid
from pfas.preprocessing import preprocess_configuration, run_simulation


def test_simulation(configuration):
    domain_length = configuration["experimental_conditions"]["domain_length"]
    time_length = round(configuration["experimental_conditions"]["time_total"]/
                        configuration["experimental_conditions"]["time_resolution"])
    grid_shape = (domain_length, time_length)
    params = preprocess_configuration(configuration)
    C1, C2, C_tot, grid = run_simulation(params)
    assert isinstance(C1, np.ndarray)
    assert C1.shape == grid_shape
    assert isinstance(C2, np.ndarray)
    assert C2.shape == grid_shape
    assert isinstance(C_tot, np.ndarray)
    assert C_tot.shape == grid_shape
    assert isinstance(grid, SimulationGrid)
    assert len(grid.depth), len(grid.time) == grid_shape

