import numpy as np

from pfas.analytical_soln import SimulationGrid
from pfas.model import Model
from pfas.preprocessing import (
    WaterPreprocessor, BoundaryPreprocessor, GridGenerator,
    SpRetardationPreprocessor, SWCAdsorptionPreprocessor,
    SorptionKawiDirectInput, SimulationRunner
)


def test_simulation(configuration):
    # Create the model with the configuration object
    model = Model(configuration)
    
    # Add each preprocessor in order
    model.add(GridGenerator)
    model.add(WaterPreprocessor)
    model.add(BoundaryPreprocessor)
    model.add(SpRetardationPreprocessor)
    model.add(SWCAdsorptionPreprocessor)
    model.add(SorptionKawiDirectInput)
    model.add(SimulationRunner)
    
    # Extract results from generated_data
    C1 = model.generated_data["C1"]
    C2 = model.generated_data["C2"]
    C_tot = model.generated_data["C_tot"]
    grid = model.generated_data["grid"]
    
    # Calculate expected grid dimensions from configuration
    # Access the underlying dict
    exp_cond = configuration.config_dict["experimental_conditions"]
    domain_length = exp_cond["domain_length"]
    spatial_resolution = exp_cond["spatial_resolution"]
    time_total = exp_cond["time_total"]
    time_resolution = exp_cond["time_resolution"]
    
    expected_depth_length = int(domain_length / spatial_resolution)
    expected_time_length = int(time_total / time_resolution + 0.5)
    grid_shape = (expected_depth_length, expected_time_length)
    
    # Assertions
    assert isinstance(C1, np.ndarray)
    assert C1.shape == grid_shape
    assert isinstance(C2, np.ndarray)
    assert C2.shape == grid_shape
    assert isinstance(C_tot, np.ndarray)
    assert C_tot.shape == grid_shape
    assert isinstance(grid, SimulationGrid)
    assert (len(grid.depth), len(grid.time)) == grid_shape