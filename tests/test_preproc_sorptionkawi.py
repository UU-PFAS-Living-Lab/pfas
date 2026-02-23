import pytest

from pfas.preprocessing import SorptionKawiDirectInput, AdsorptionCollector


def test_compute_returns_awi(valid_sorption_kawi_direct_input):
    result = valid_sorption_kawi_direct_input.compute()

    assert isinstance(result, dict)
    assert "awi_retardation" in result


def test_defaults_when_optional_sorption_fields_missing(result_water, valid_sp_retardation_preprocessor):
    """AdsorptionCollector applies defaults for rate_const and frac_int when sorption_solid is empty."""
    sp_result = valid_sp_retardation_preprocessor.compute()

    awi_retardation = SorptionKawiDirectInput(
        kaw=1.0,
        hydro_properties=result_water["hydro_properties"],
        aaw=10.0,
    ).compute()["awi_retardation"]

    adsorption = AdsorptionCollector(
        Kd=sp_result["Kd"],
        sp_retardation=sp_result["sp_retardation"],
        awi_retardation=awi_retardation,
        sorption_solid={},  # no optional fields — defaults should apply
    ).compute()["adsorption"]

    assert adsorption.rate_const == 0.0
    assert adsorption.frac_int == 1.0


def test_extra_fields_forbidden(result_water):
    with pytest.raises(Exception):
        SorptionKawiDirectInput(
            kaw=0.5,
            hydro_properties=result_water["hydro_properties"],
            aaw=100.0,
            extra_field=123,  # forbidden by Pydantic
        )


def test_outputs_property(result_water):
    obj = SorptionKawiDirectInput(
        kaw=1.0,
        hydro_properties=result_water["hydro_properties"],
        aaw=1.0,
    )

    assert obj.outputs == ["awi_retardation"]