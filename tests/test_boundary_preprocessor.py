import pytest
from pfas.component import BoundaryPreprocessor
from pfas.data_structure import BoundaryConditions


def test_boundary_preprocessor_basic():
    """
    Test that BoundaryPreprocessor.compute() returns a BoundaryConditions object
    with the same C_list and T_list.

    This verifies:
    - Correct pass-through behavior
    - Correct construction of BoundaryConditions
    """
    bp = BoundaryPreprocessor(
        C_list=[10.0, 0.0],
        T_list=[0.0, 200.0],
    )
    out = bp.compute()
    assert "boundary_conditions" in out
    bc = out["boundary_conditions"]
    assert isinstance(bc, BoundaryConditions)
    assert bc.C_list == [10.0, 0.0]
    assert bc.T_list == [0.0, 200.0]


def test_boundary_preprocessor_requires_same_length():
    """
    Test that mismatched C_list and T_list lengths raise an error.

    This verifies:
    - The model_validator enforcing equal-length lists
    """
    with pytest.raises(ValueError):
        BoundaryPreprocessor(C_list=[1.0], T_list=[0.0, 10.0])


def test_boundary_preprocessor_requires_start_at_zero():
    """
    Test that T_list[0] must be zero.

    This verifies:
    - Correct enforcement of the analytical solution's boundary condition format
    """
    with pytest.raises(ValueError):
        BoundaryPreprocessor(C_list=[1.0], T_list=[5.0])


def test_boundary_preprocessor_requires_increasing_times():
    """
    Test that T_list must be strictly increasing.

    This verifies:
    - Prevention of overlapping or invalid time intervals
    """
    with pytest.raises(ValueError):
        BoundaryPreprocessor(C_list=[1.0, 0.0], T_list=[0.0, 0.0])


def test_boundary_preprocessor_outputs_property():
    """
    Test that the outputs property returns the correct output key.

    This verifies:
    - Consistency with Model.compute() orchestration
    """
    bp = BoundaryPreprocessor(C_list=[1.0], T_list=[0.0])
    assert bp.outputs == ["boundary_conditions"]
