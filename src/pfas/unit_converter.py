"""Class to convert units."""
import re


class UnitConverter:
    """
    Convert a wide range of physical and chemical units to their SI equivalents.

    This class provides a unified interface for converting:
    - Length units → meters (m)
    - Mass units → kilograms (kg)
    - Temperature → Kelvin (K)
    - Volume ratios (e.g., L/kg, mL/g) → m³/kg
    - Flow rates (e.g., cm**2/s, mm**3/s, mL/min) → SI flow units
    - Molality (mol/kg, mmol/g, …) → mol/kg
    - Molarity (mol/L, mmol/mL, …) → mol/m³
    - Mass–mole conversions (kg ↔ mol) when a molar mass is provided

    Parameters
    ----------
    value : float
        The numeric value to convert.

    unit : str
        The unit string to convert from. Examples:
        - Length: "cm", "mm", "inch"
        - Mass: "g", "mg", "lb"
        - Temperature: "C", "F"
        - Molality: "mol/kg", "mmol/g"
        - Molarity: "mol/L", "mmol/mL"
        - Volume ratios: "L/kg", "mL/g"
        - Flow rates: "cm**2/s", "mm**3/s", "mL/min"
        - Chemical mass/mole: "kg", "mol" (requires molar_mass)

    molar_mass : float, optional
        Molar mass in kg/mol.
        Required only when converting between mass and moles.

    Returns
    -------
    (converted_value, si_unit) : tuple
        A tuple containing:
        - The converted numeric value
        - The SI unit string

    Raises
    ------
    ValueError
        If the unit is unsupported or ambiguous.

    Examples
    --------
    Basic physical units:
    >>> UnitConverter.to_si(10, "cm")
    (0.1, "m")

    >>> UnitConverter.to_si(500, "g")
    (0.5, "kg")

    Temperature:
    >>> UnitConverter.to_si(25, "C")
    (298.15, "K")

    Molality:
    >>> UnitConverter.to_si(5, "mol/g")
    (5000.0, "mol/kg")

    Molarity:
    >>> UnitConverter.to_si(2, "mmol/mL")
    (2000.0, "mol/m**3")

    Volume ratios:
    >>> UnitConverter.to_si(1, "L/kg")
    (0.001, "m**3/kg")

    Flow rates:
    >>> UnitConverter.to_si(10, "cm**2/s")
    (0.001, "m**2/s")

    Mass ↔ mol (requires molar mass):
    >>> UnitConverter.to_si(0.1, "kg", molar_mass=0.05844)
    (1.7111567419575635, "mol")

    >>> UnitConverter.to_si(2, "mol", molar_mass=0.05844)
    (0.11688, "kg")
    """

    # -----------------------------
    # Shared unit tables
    # -----------------------------
    LENGTH_UNITS = {
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'km': 1000,
        'inch': 0.0254,
        'foot': 0.3048,
        'mile': 1609.34,
    }

    MASS_UNITS = {
        'kg': 1,
        'g': 1e-3,
        'mg': 1e-6,
        'lb': 0.453592,
        'oz': 0.0283495,
    }

    MOL_PREFIX = {
        'mol': 1,
        'mmol': 1e-3,
        'µmol': 1e-6,
        'umol': 1e-6,
    }

    VOLUME_UNITS = {
        'l': 1e-3,
        'ml': 1e-6,
        'm**3': 1,
    }

    FLOW_LENGTH_UNITS = ["cm", "mm", "m"]

    # -----------------------------
    # Flow‑rate detection
    # -----------------------------
    @classmethod
    def is_flow_rate_unit(cls, unit):
        # Extract alphabetic tokens only (ignore numbers like "2" or "3")
        bases = re.findall(r"[a-zA-Z]+", unit)
        return all(
            b in cls.FLOW_LENGTH_UNITS or b in ("s", "min")
            for b in bases
        )

    # -----------------------------
    # Molality
    # -----------------------------
    @classmethod
    def molality_to_si(cls, value, unit):
        unit = unit.lower()
        num, denom = unit.split('/')

        if num not in cls.MOL_PREFIX:
            raise ValueError(f"Unsupported mol prefix '{num}'")
        if denom not in ('kg', 'g'):
            raise ValueError(f"Unsupported mass unit '{denom}'")

        value *= cls.MOL_PREFIX[num]
        value /= (1 if denom == 'kg' else 1e-3)

        return value, "mol/kg"

    # -----------------------------
    # Molarity
    # -----------------------------
    @classmethod
    def molarity_to_si(cls, value, unit):
        unit = unit.lower()
        num, denom = unit.split('/')

        if num not in cls.MOL_PREFIX:
            raise ValueError(f"Unsupported mol prefix '{num}'")
        if denom not in cls.VOLUME_UNITS:
            raise ValueError(f"Unsupported volume unit '{denom}'")

        value *= cls.MOL_PREFIX[num]
        value /= cls.VOLUME_UNITS[denom]

        return value, "mol/m**3"

    # -----------------------------
    # Mass ↔ mol
    # -----------------------------
    @staticmethod
    def mass_mole_convert(value, unit, molar_mass):
        if unit == 'kg':
            return value / molar_mass, 'mol'
        if unit == 'mol':
            return value * molar_mass, 'kg'
        raise ValueError("Unit must be 'kg' or 'mol'")

    # -----------------------------
    # Volume ratios
    # -----------------------------
    @classmethod
    def volume_to_si(cls, value, unit):
        num, denom = unit.split('/')
        num = num.lower()
        denom = denom.lower()

        if num not in cls.VOLUME_UNITS:
            raise ValueError(f"Unsupported volume unit '{num}'")
        if denom not in ('kg', 'g'):
            raise ValueError(f"Unsupported mass unit '{denom}'")

        value *= cls.VOLUME_UNITS[num]
        value /= (1 if denom == 'kg' else 1e-3)

        return value, "m**3/kg"

    # -----------------------------
    # Flow rates
    # -----------------------------
    @classmethod
    def flow_rates_to_si(cls, value, unit):
        match = re.findall(r'([a-zA-Z]+)(\*\*(\d+))?', unit)
        si_value = value

        length_units = {'m': 1, 'cm': 0.01, 'mm': 0.001}

        for base, _, power in match:
            exponent = int(power) if power else 1
            if base in length_units:
                si_value *= length_units[base] ** exponent
            elif base in ['s', 'min']:
                continue
            else:
                raise ValueError(f"Unit {base} not supported")

        si_unit = unit
        for base, _, _ in match:
            if base in length_units:
                si_unit = si_unit.replace(base, 'm')

        return si_value, si_unit

    # -----------------------------
    # Dispatcher
    # -----------------------------
    # ruff: noqa: PLR0911
    @classmethod
    def to_si(cls, value, unit, molar_mass=None):
        unit = unit.strip().lower()

        # Temperature
        if unit == 'c':
            return value + 273.15, 'K'
        if unit == 'f':
            return (value - 32) * 5/9 + 273.15, 'K'

        # Length
        if unit in cls.LENGTH_UNITS:
            return value * cls.LENGTH_UNITS[unit], 'm'

        # Mass ↔ mol (priority)
        if unit in ('mol', 'kg') and molar_mass is not None:
            return cls.mass_mole_convert(value, unit, molar_mass)

        # Mass (only if no molar_mass)
        if unit in cls.MASS_UNITS:
            return value * cls.MASS_UNITS[unit], 'kg'

        # Molality
        if unit in ['mol/kg', 'mol/g', 'mmol/kg', 'mmol/g']:
            return cls.molality_to_si(value, unit)

        # Molarity
        if unit in ['mol/l', 'mol/ml', 'mmol/l', 'mmol/ml']:
            return cls.molarity_to_si(value, unit)

        # Volume ratios
        if '/' in unit:
            num, _ = unit.split('/')
            if num in ('l', 'ml'):
                return cls.volume_to_si(value, unit)

        # Flow rates
        if cls.is_flow_rate_unit(unit):
            return cls.flow_rates_to_si(value, unit)

        raise ValueError(f"Unit '{unit}' not supported for SI conversion")
