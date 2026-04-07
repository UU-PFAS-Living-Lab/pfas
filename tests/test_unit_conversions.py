import pytest
from pfas.unit_converter import UnitConverter


# -----------------------------
# Temperature
# -----------------------------
def test_temperature_celsius():
    val, unit = UnitConverter.to_si(25, "C")
    assert unit == "K"
    assert pytest.approx(val, rel=1e-6) == 298.15


def test_temperature_fahrenheit():
    val, unit = UnitConverter.to_si(77, "F")
    assert unit == "K"
    assert pytest.approx(val, rel=1e-6) == 298.15


# -----------------------------
# Length
# -----------------------------
def test_length_cm():
    val, unit = UnitConverter.to_si(10, "cm")
    assert unit == "m"
    assert val == 0.1


def test_length_inch():
    val, unit = UnitConverter.to_si(2, "inch")
    assert unit == "m"
    assert pytest.approx(val, rel=1e-6) == 0.0508


# -----------------------------
# Mass
# -----------------------------
def test_mass_grams():
    val, unit = UnitConverter.to_si(500, "g")
    assert unit == "kg"
    assert val == 0.5


def test_mass_lb():
    val, unit = UnitConverter.to_si(2, "lb")
    assert unit == "kg"
    assert pytest.approx(val, rel=1e-6) == 0.907184


# -----------------------------
# Molality
# -----------------------------
def test_molality_mol_per_g():
    val, unit = UnitConverter.to_si(5, "mol/g")
    assert unit == "mol/kg"
    assert val == 5000.0


def test_molality_mmol_per_kg():
    val, unit = UnitConverter.to_si(3, "mmol/kg")
    assert unit == "mol/kg"
    assert val == 0.003


# -----------------------------
# Molarity
# -----------------------------
def test_molarity_mol_per_L():
    val, unit = UnitConverter.to_si(1, "mol/L")
    assert unit == "mol/m**3"
    assert val == 1000.0


def test_molarity_mmol_per_mL():
    val, unit = UnitConverter.to_si(2, "mmol/mL")
    assert unit == "mol/m**3"
    assert pytest.approx(val, rel=1e-12) == 2000.0


# -----------------------------
# Volume ratios
# -----------------------------
def test_volume_ratio_L_per_kg():
    val, unit = UnitConverter.to_si(1, "L/kg")
    assert unit == "m**3/kg"
    assert val == 0.001


def test_volume_ratio_mL_per_g():
    val, unit = UnitConverter.to_si(5, "mL/g")
    assert unit == "m**3/kg"
    assert val == pytest.approx(0.005, rel=1e-12)


# -----------------------------
# Flow rates
# -----------------------------
def test_flow_cm2_per_s():
    val, unit = UnitConverter.to_si(10, "cm**2/s")
    assert unit == "m**2/s"
    assert pytest.approx(val, rel=1e-6) == 0.001


def test_flow_cm3_per_s():
    val, unit = UnitConverter.to_si(10, "cm**3/s")
    assert unit == "m**3/s"
    assert pytest.approx(val, rel=1e-6) == 1e-5


def test_flow_mm3_per_min():
    val, unit = UnitConverter.to_si(60, "mm**3/min")
    assert unit == "m**3/min"
    assert pytest.approx(val, rel=1e-12) == 60 * (1e-9)


# -----------------------------
# Mass ↔ mol conversions
# -----------------------------
def test_mass_to_mol():
    val, unit = UnitConverter.to_si(0.1, "kg", molar_mass=0.05844)
    assert unit == "mol"
    assert pytest.approx(val, rel=1e-12) == 0.1 / 0.05844


def test_mol_to_mass():
    val, unit = UnitConverter.to_si(2, "mol", molar_mass=0.05844)
    assert unit == "kg"
    assert pytest.approx(val, rel=1e-12) == 2 * 0.05844


# -----------------------------
# Error handling
# -----------------------------
def test_unsupported_unit():
    with pytest.raises(ValueError):
        UnitConverter.to_si(1, "bananas")


def test_missing_molar_mass():
    # "mol" without molar_mass must raise
    with pytest.raises(ValueError):
        UnitConverter.to_si(1, "mol")

