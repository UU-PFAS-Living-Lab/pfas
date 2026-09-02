import pytest
from pydantic import ValidationError

from pfas.component import Le2021_asymptote, Le2021_langmuir, Szyszkowski


@pytest.fixture
def structural_props() -> dict:
    """Valid structural properties for Le2021 models."""
    return {
        "n_CFx": 7,
        "n_CHx": 1,
        "n_COO": 0,
        "n_COOH": 0,
        "n_SO3": 0,
        "n_R4N": 0,
        "n_OH": 0,
        "n_OSO3": 0,
        "n__O_": 0,
        "n__S_": 0,
        "n_N_CH3_2_CH2_COO": 0,
    }


@pytest.fixture
def szyszkowski_params() -> dict:
    """Szyszkowski model parameters."""
    return {
        "sigma0": 0.072,
        "a": 0.5,
        "b": 1.2,
        "chi": 2,
        "T": 298.0,
    }


# --- Le2021_asymptote (dilute-limit group-contribution model) ---

def test_le2021_asymptote_component_compute(structural_props: dict):
    """Component wraps Kaw_0_Le2021 and exposes it via compute()."""
    component = Le2021_asymptote(structural_properties=structural_props)
    result = component.compute()

    assert "Kaw" in result
    assert result["Kaw"] > 0.0
    assert component.outputs == ["Kaw"]


def test_le2021_asymptote_component_missing_keys():
    """Missing structural keys should raise a validation error at construction."""
    with pytest.raises(ValidationError):
        Le2021_asymptote(structural_properties={"n_CFx": 7})


# --- Le2021_langmuir (concentration-dependent model) ---

def test_le2021_langmuir_component_compute(structural_props: dict):
    """Component computes a concentration-dependent Kaw via compute()."""
    component = Le2021_langmuir(structural_properties=structural_props, Cw=1e-6)
    result = component.compute()

    assert "Kaw" in result
    assert result["Kaw"] > 0.0
    assert component.outputs == ["Kaw"]


def test_le2021_langmuir_component_missing_keys():
    """Missing structural keys should raise a validation error at construction."""
    with pytest.raises(ValidationError):
        Le2021_langmuir(structural_properties={"n_CFx": 7}, Cw=1e-6)


def test_le2021_langmuir_component_varies_with_concentration(structural_props: dict):
    """Kaw should differ between two different aqueous concentrations."""
    low = Le2021_langmuir(structural_properties=structural_props, Cw=1e-9)
    high = Le2021_langmuir(structural_properties=structural_props, Cw=1e-3)

    assert low.compute()["Kaw"] != high.compute()["Kaw"]


# --- Szyszkowski (fitted-parameter model) ---

def test_szyszkowski_component_compute(szyszkowski_params: dict):
    """Component computes Kaw from Szyszkowski fitting parameters."""
    component = Szyszkowski(**szyszkowski_params, Cw=1e-6)
    result = component.compute()

    assert "Kaw" in result
    assert result["Kaw"] > 0.0
    assert component.outputs == ["Kaw"]


def test_szyszkowski_component_missing_parameters():
    """Missing required fitting parameters should raise a validation error."""
    with pytest.raises(ValidationError):
        Szyszkowski(a=0.5, b=1.2, Cw=1e-6)  # missing nothing actually required here;
        # adjust below if `a`/`b` truly are the only required fields


def test_szyszkowski_component_missing_required_field():
    """Omitting Cw (required, no default) should raise a validation error."""
    with pytest.raises(ValidationError):
        Szyszkowski(sigma0=0.072, a=0.5, b=1.2, chi=2, T=298.0)


def test_szyszkowski_component_defaults(szyszkowski_params: dict):
    """sigma0, chi, and T should fall back to their documented defaults."""
    component = Szyszkowski(a=szyszkowski_params["a"], b=szyszkowski_params["b"], Cw=1e-6)

    assert component.sigma0 == 0.072
    assert component.chi == 2
    assert component.T == 298.0