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

# Test pydantic validators for simple scalar constraints

@pytest.mark.parametrize(
    "field,value",
    [
        ("average_infiltration_rate", 0.0),
        ("average_infiltration_rate", -1e-9),
        ("solute_concentration_influx", -1.0),
    ],
)
def test_positive_constraints(field, value):
    # only one field is made invalid at a time; others kept valid
    kwargs = dict(
        average_infiltration_rate=1e-8,
        solute_concentration_influx=100.0,
        pulse_intervals=[(0, 2000)],
    )
    kwargs[field] = value

    with pytest.raises(ValidationError):
        BoundaryPreprocessor(**kwargs)


# pulse_intervals has its own custom validator, so test a variety of
# invalid interval lists separately
@pytest.mark.parametrize("intervals", [
    [],                # empty list
    [(0, 0)],          # zero‑length interval
    [(-1, 1)],         # negative start time
    # overlapping intervals are currently allowed by the validator
])
def test_invalid_pulse_intervals(intervals):
    with pytest.raises(ValidationError):
        BoundaryPreprocessor(
            average_infiltration_rate=1e-8,
            solute_concentration_influx=100.0,
            pulse_intervals=intervals,
        )

