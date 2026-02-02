import pytest
from pydantic import ValidationError

from pfas.preprocessing import BoundaryPreprocessor, BoundaryConditions

@pytest.fixture
def valid_preprocessor():
    """A valid BoundaryPreprocessor instance."""
    return BoundaryPreprocessor(
        average_infiltration_rate=1e-8,
        solute_concentration_influx=100.0,  # mg/L
        pulse_duration=3600.0,               # s
    )


@pytest.fixture
def result(valid_preprocessor):
    """Computed result dictionary."""
    return valid_preprocessor.compute()

def test_outputs_property(valid_preprocessor):
    assert valid_preprocessor.outputs == ["boundary_conditions"]

def test_compute_returns_expected_key(result):
    assert "boundary_conditions" in result

def test_compute_returns_boundary_conditions(result):
    bc = result["boundary_conditions"]
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

