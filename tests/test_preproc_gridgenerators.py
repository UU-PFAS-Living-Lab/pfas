import pytest
from pydantic import ValidationError

from pfas.preprocessing import GridGenerator, SimulationGrid


def test_outputs_property(valid_grid_generator):
    assert valid_grid_generator.outputs == ["grid"]

def test_compute_returns_grid(valid_grid_generator):
    result = valid_grid_generator.compute()
    assert "grid" in result
    assert isinstance(result["grid"], SimulationGrid)

def test_grid_shapes(valid_grid_generator):
    grid = valid_grid_generator.compute()["grid"]

    assert len(grid.depth) == 10
    assert len(grid.time) == 5

def test_validation_error():
    with pytest.raises(ValidationError):
        GridGenerator(
            domain_length=-1.0,
            spatial_resolution=1.0,
            time_resolution=1.0,
            time_total=1.0,
        )

