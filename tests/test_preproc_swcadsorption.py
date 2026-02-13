
import pytest

from pfas.preprocessing import SWCAdsorptionPreprocessor


def test_compute_swc_based_returns_dict(valid_swc_adsorption_swc):
    result = valid_swc_adsorption_swc.compute()

    assert isinstance(result, dict)
    assert "aaw" in result


def test_compute_swc_based_positive_value(valid_swc_adsorption_swc):
    result = valid_swc_adsorption_swc.compute()

    assert result["aaw"] > 0.0

def test_compute_guo_returns_dict(valid_swc_adsorption_guo):
    result = valid_swc_adsorption_guo.compute()

    assert isinstance(result, dict)
    assert "aaw" in result

def test_compute_guo_positive_value(valid_swc_adsorption_guo):
    result = valid_swc_adsorption_guo.compute()

    assert result["aaw"] > 0.0

def test_extra_field_forbidden(
    result_water,
    soil_params,
    awi_swc_based,
):
    with pytest.raises(ValueError):
        SWCAdsorptionPreprocessor(
            hydro_properties=result_water["hydro_properties"],
            scaling_factor_awi=1.0,
            AWI=awi_swc_based,
            soil=soil_params,
            extra_field=123,  # forbidden by extra="forbid"
        )

