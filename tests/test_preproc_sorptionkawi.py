import pytest

from pfas.analytical_soln import Adsorption
from pfas.preprocessing import SorptionKawiDirectInput


def test_compute_returns_adsorption(valid_sorption_kawi_direct_input):
    result = valid_sorption_kawi_direct_input.compute()

    assert isinstance(result, dict)
    assert "adsorption" in result
    assert isinstance(result["adsorption"], Adsorption)


def test_awi_retardation_formula(valid_sorption_kawi_direct_input):
    result = valid_sorption_kawi_direct_input.compute()
    adsorption = result["adsorption"]

    theta = valid_sorption_kawi_direct_input.hydro_properties.water_content
    expected = (
        valid_sorption_kawi_direct_input.kaw
        * valid_sorption_kawi_direct_input.aaw
        / theta
    )

    assert adsorption.awi_retardation == pytest.approx(expected)

def test_solid_phase_parameters_propagated(valid_sorption_kawi_direct_input):
    adsorption = valid_sorption_kawi_direct_input.compute()["adsorption"]

    assert adsorption.Kd == valid_sorption_kawi_direct_input.Kd
    assert adsorption.sp_retardation == valid_sorption_kawi_direct_input.sp_retardation
    assert adsorption.rate_const == valid_sorption_kawi_direct_input.sorption_solid.get(
        "rate_const", 0.0
    )
    assert adsorption.frac_int == valid_sorption_kawi_direct_input.sorption_solid.get(
        "fraction_instantaneous", 1.0
    )


def test_defaults_when_optional_sorption_fields_missing(result_water, valid_sp_retardation_preprocessor):
    sp_ret = valid_sp_retardation_preprocessor.compute()["sp_retardation"]
    sorption = SorptionKawiDirectInput(
        kaw=1.0,
        hydro_properties=result_water["hydro_properties"],
        Kd=0.8,
        aaw=10.0,
        sorption_solid={},  # no optional fields
        sp_retardation=sp_ret,
    )

    adsorption = sorption.compute()["adsorption"]

    assert adsorption.rate_const == 0.0
    assert adsorption.frac_int == 1.0

def test_extra_fields_forbidden(
    result_water,
    valid_sp_retardation_preprocessor,
):
    sp_ret = valid_sp_retardation_preprocessor.compute()["sp_retardation"]

    with pytest.raises(ValueError):
        SorptionKawiDirectInput(
            kaw=0.5,
            hydro_properties=result_water["hydro_properties"],
            Kd=0.8,            # direct input
            aaw=100.0,
            sorption_solid={},
            sp_retardation=sp_ret,
            extra_field=123,   # forbidden by Pydantic
        )

def test_outputs_property(result_water):
    obj = SorptionKawiDirectInput(
        kaw=1.0,
        hydro_properties=result_water["hydro_properties"],
        Kd=1.0,
        aaw=1.0,
        sorption_solid={},
        sp_retardation=1.0,
    )

    assert obj.outputs == ["awi_retardation"]
