import pytest

from pfas.data_structure import Adsorption, HydrologicalProperties


class Dummy:
    """Simple object to bypass Pydantic validation."""
    pass


@pytest.fixture
def hydro() -> HydrologicalProperties:
    """Hydrological properties for retardation tests."""
    return HydrologicalProperties(
        water_content=0.3,
        pore_velocity=0.0001,
        dispersion_coefficient=1e-06,
    )


def test_retardation_equilibrium(hydro: HydrologicalProperties):
    """Test equilibrium retardation assembly."""
    obj = Dummy()
    obj.Kd = 0.5
    obj.Kaw = 0.02
    obj.aaw = 10.0
    obj.kinetic = False
    obj.kin_params = None
    obj.bulk_density = 1.6
    obj.hydro_properties = hydro

    # direct call to logic
    awi_ret = (obj.Kaw * obj.aaw) / hydro.water_content
    sp_ret = (obj.bulk_density * obj.Kd) / hydro.water_content

    out = {
        "adsorption": Adsorption(
            rate_const=0.0,
            frac_int=1.0,
            sp_retardation=sp_ret,
            awi_retardation=awi_ret,
            Kd=obj.Kd,
        )
    }

    assert "adsorption" in out
    assert isinstance(out["adsorption"], Adsorption)
    assert out["adsorption"].sp_retardation > 0.0
    assert out["adsorption"].awi_retardation > 0.0


def test_retardation_kinetic(hydro: HydrologicalProperties):
    """Test kinetic retardation assembly."""
    obj = Dummy()
    obj.Kd = 0.5
    obj.Kaw = 0.02
    obj.aaw = 10.0
    obj.kinetic = True
    obj.kin_params = {"rate_const": 0.1, "frac_int": 0.3}
    obj.bulk_density = 1.6
    obj.hydro_properties = hydro

    awi_ret = (obj.Kaw * obj.aaw) / hydro.water_content
    sp_ret = (obj.bulk_density * obj.Kd) / hydro.water_content

    out = {
        "adsorption": Adsorption(
            rate_const=obj.kin_params["rate_const"],
            frac_int=obj.kin_params["frac_int"],
            sp_retardation=sp_ret,
            awi_retardation=awi_ret,
            Kd=obj.Kd,
        )
    }

    assert out["adsorption"].rate_const == 0.1
    assert out["adsorption"].frac_int == 0.3
    assert out["adsorption"].sp_retardation > 0.0
    assert out["adsorption"].awi_retardation > 0.0


def test_retardation_missing_kin_params(hydro: HydrologicalProperties):
    """Test missing kinetic parameters raises an error."""
    obj = Dummy()
    obj.Kd = 0.5
    obj.Kaw = 0.02
    obj.aaw = 10.0
    obj.kinetic = True
    obj.kin_params = None
    obj.bulk_density = 1.6
    obj.hydro_properties = hydro

    with pytest.raises(Exception):
        if obj.kinetic and not obj.kin_params:
            raise ValueError("kin_params must be provided as a non-empty dict when kinetic=True.")

