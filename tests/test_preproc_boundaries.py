import pytest
from pydantic import ValidationError

from pfas.preprocessing import BoundaryConditions, BoundaryPreprocessor


def test_outputs_property(valid_boundary_preprocessor):
    assert valid_boundary_preprocessor.outputs == ["boundary_conditions"]

def test_compute_returns_expected_key(result_boundary):
    assert "boundary_conditions" in result_boundary

def test_compute_returns_boundary_conditions(result_boundary):
    bc = result_boundary["boundary_conditions"]
    assert isinstance(bc, BoundaryConditions)

# Test pydantic

@pytest.mark.parametrize(
    "field,value",
    [
        ("average_infiltration_rate", 0.0),
        ("solute_concentration_influx", -1.0),
        ("pulse_duration", 0.0),
    ],
)
def test_positive_constraints(field, value):
    kwargs = dict(
        average_infiltration_rate=1e-8,
        solute_concentration_influx=100.0,
        pulse_duration=3600.0,
    )
    kwargs[field] = value

    with pytest.raises(ValidationError):
        BoundaryPreprocessor(**kwargs)

