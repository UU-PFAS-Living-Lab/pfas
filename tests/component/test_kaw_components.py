import pytest

from pfas.utils import (
    Kaw_0_Le2021,
    Kaw_langmuir_Le2021,
    dG0_Le2021,
    Kaw_Szyszkowski,
)


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


def test_le2021_asymptote_compute(structural_props: dict):
    """Test Kaw calculation using Le2021 asymptote model."""
    kaw = Kaw_0_Le2021(structural_props)
    assert kaw > 0.0


def test_le2021_asymptote_missing_keys():
    """Test missing structural keys raises an error."""
    with pytest.raises(Exception):
        Kaw_0_Le2021({"n_CFx": 7})


def test_le2021_langmuir_compute(structural_props: dict):
    """Test Kaw calculation using Le2021 Langmuir model."""
    kaw0 = Kaw_0_Le2021(structural_props)
    dg0 = dG0_Le2021(structural_props)
    kaw = Kaw_langmuir_Le2021(kaw0, dg0, Cw=1e-6)
    assert kaw > 0.0


def test_le2021_langmuir_missing_keys():
    """Test missing structural keys raises an error."""
    with pytest.raises(Exception):
        dG0_Le2021({"n_CFx": 7})


def test_szyszkowski_compute(szyszkowski_params: dict):
    """Test Kaw calculation using Szyszkowski model."""
    kaw = Kaw_Szyszkowski(
        sigma0=szyszkowski_params["sigma0"],
        a=szyszkowski_params["a"],
        b=szyszkowski_params["b"],
        Cw=1e-6,
        chi=szyszkowski_params["chi"],
        T=szyszkowski_params["T"],
    )
    assert kaw > 0.0


def test_szyszkowski_missing_parameters():
    """Test missing Szyszkowski parameters raises an error."""
    with pytest.raises(Exception):
        Kaw_Szyszkowski(a=0.5, b=1.2, Cw=1e-6)

