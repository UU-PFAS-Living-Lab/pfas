import pytest
from pydantic import ValidationError

from pfas.preprocessing import SpRetardationPreprocessor

def test_outputs_property(valid_sp_retardation_preprocessor):
    assert valid_sp_retardation_preprocessor.outputs == ["sp_retardation"]

def test_compute_returns_keys(valid_sp_retardation_preprocessor):
    result = valid_sp_retardation_preprocessor.compute()

    assert "sp_retardation" in result
    assert "Kd" in result

def test_kd_value(valid_sp_retardation_preprocessor):
    result = valid_sp_retardation_preprocessor.compute()

    assert result["Kd"] == 0.8

