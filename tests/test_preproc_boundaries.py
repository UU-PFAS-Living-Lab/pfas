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

# Test pydantic validators for list constraints

@pytest.mark.parametrize(
    "field,value",
    [
        ("C_list", [-1.0]),  # negative concentration not allowed
    ],
)
def test_positive_constraints(field, value):
    # only one field is made invalid at a time; others kept valid
    kwargs = dict(
        C_list=[100.0, 0],
        T_list=[0, 2000],
    )
    kwargs[field] = value

    with pytest.raises(ValidationError):
        BoundaryPreprocessor(**kwargs)


# T_list has its own custom validators, so test a variety of
# invalid T_list cases separately
@pytest.mark.parametrize("T_list", [
    [],                # empty list
    [1],               # doesn't start with 0
    ([0, 0]),          # non-strictly increasing
    [0, -1],           # negative time
    # overlapping intervals are currently allowed by the validator
])
def test_invalid_T_list(T_list):
    with pytest.raises(ValidationError):
        BoundaryPreprocessor(
            C_list=[100.0, 0],
            T_list=T_list,
        )

