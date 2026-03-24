"""Convert units to SI."""
import re

def molality_to_si(value, unit):
    """
    Convert molality-like units to SI mol/kg.

    Supported:
    mol/kg, mol/g, mmol/kg, mmol/g
    """
    unit = unit.lower().strip()

    # Prefix factors
    mol_prefix = {
        'mol': 1,
        'mmol': 1e-3,
        'µmol': 1e-6,
        'umol': 1e-6,
    }

    # Mass units
    mass_units = {
        'kg': 1,
        'g': 1e-3,
    }

    if '/' not in unit:
        raise ValueError("Molality must be a ratio like 'mol/kg'")

    num, denom = unit.split('/')

    if num not in mol_prefix:
        raise ValueError(f"Unsupported mol prefix '{num}'")

    if denom not in mass_units:
        raise ValueError(f"Unsupported mass unit '{denom}'")

    # Convert numerator to mol
    value *= mol_prefix[num]

    # Convert denominator to kg
    value /= mass_units[denom]

    return value, "mol/kg"

def molarity_to_si(value, unit):
    """
    Convert molarity-like units to SI mol/m**3.

    Supported:
    mol/L, mol/mL, mmol/L, mmol/mL
    """
    unit = unit.lower().strip()

    mol_prefix = {
        'mol': 1,
        'mmol': 1e-3,
        'µmol': 1e-6,
        'umol': 1e-6,
    }

    volume_units = {
        'l': 1e-3,     # L → m³
        'ml': 1e-6,    # mL → m³
    }

    if '/' not in unit:
        raise ValueError("Molarity must be a ratio like 'mol/L'")

    num, denom = unit.split('/')

    if num not in mol_prefix:
        raise ValueError(f"Unsupported mol prefix '{num}'")

    if denom not in volume_units:
        raise ValueError(f"Unsupported volume unit '{denom}'")

    # Convert numerator to mol
    value *= mol_prefix[num]

    # Convert denominator to m³
    value /= volume_units[denom]

    return value, "mol/m**3"

def mass_mole_convert(value, unit, molar_mass):
    """
    Convert between mass and moles.

    unit = 'kg' or 'mol'
    molar_mass in kg/mol
    """
    if unit == 'kg':
        # mass → mol
        return value / molar_mass, 'mol'
    elif unit == 'mol':
        # mol → mass
        return value * molar_mass, 'kg'
    else:
        raise ValueError("Unit must be 'kg' or 'mol'")



def volume_to_si(value, unit):
    """Convert units like 'L/kg', 'mL/g' to SI."""
    volume_units = {'L': 1e-3, 'mL': 1e-6, 'm**3': 1}  # all to m³
    mass_units = {'kg': 1, 'g': 1e-3}  # all to kg

    if '/' in unit:
        num, denom = unit.split('/')
        # Convert numerator
        if num in volume_units:
            value *= volume_units[num]
            num = 'm**3'
        # Convert denominator
        if denom in mass_units:
            value /= 1  # value already per kg after adjusting numerator
            denom = 'kg'
        return value, f'{num}/{denom}'
    else:
        raise ValueError("Unit not a ratio")

def flow_rates_to_si(value, unit):
    """
    Convert simple compound units to SI.

    Example units: 'cm**2/s', 'mm**3/s', 'mL/min'
    """
    # Match patterns like 'cm**2', 'mm**3'
    match = re.findall(r'([a-zA-Z]+)(\*\*(\d+))?', unit)
    si_value = value

    # Basic length conversion factors
    length_units = {'m': 1, 'cm': 0.01, 'mm': 0.001}

    for base, _, power in match:
        exponent = int(power) if power else 1
        if base in length_units:
            si_value *= length_units[base]**exponent
        elif base == 's':
            continue  # already SI
        else:
            raise ValueError(f"Unit {base} not supported")

    # Determine SI unit string
    si_unit = unit
    for base, _, power in match:
        if base in length_units:
            si_unit = si_unit.replace(base, 'm')

    return si_value, si_unit

def to_si(value, unit, molar_mass=None):
    """
    Convert a value from a given unit to its SI unit.

    Supported:
    - Length: cm, mm, km, inch, foot, mile
    - Mass: g, mg, lb, oz
    - Temperature: C, F
    - Volume ratios: L/kg, mL/g
    - Flow rates: cm**2/s, mm**3/s, mL/min
    - Molality: mol/kg, mol/g, mmol/kg, mmol/g
    - Molarity: mol/L, mol/mL, mmol/L, mmol/mL
    - Mass ↔ mol (requires molar_mass)
    """
    unit = unit.strip().lower()

    # ---------------------------
    # Temperature conversions
    # ---------------------------
    if unit == 'c':
        return value + 273.15, 'K'
    elif unit == 'f':
        return (value - 32) * 5/9 + 273.15, 'K'

    # ---------------------------
    # Length conversions
    # ---------------------------
    length_units = {
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'km': 1000,
        'inch': 0.0254,
        'foot': 0.3048,
        'mile': 1609.34
    }

    if unit in length_units:
        return value * length_units[unit], 'm'

    # ---------------------------
    # Mass conversions
    # ---------------------------
    mass_units = {
        'kg': 1,
        'g': 0.001,
        'mg': 1e-6,
        'lb': 0.453592,
        'oz': 0.0283495
    }

    if unit in mass_units:
        return value * mass_units[unit], 'kg'

    # ---------------------------
    # Volume ratios (L/kg, mL/g)
    # ---------------------------
    if '/' in unit and any(x in unit for x in ['l', 'ml']):
        return volume_to_si(value, unit)

    # ---------------------------
    # Flow rates (cm**2/s, mm**3/s, mL/min)
    # ---------------------------
    if any(x in unit for x in ['cm', 'mm', 'ml']) and ('/' in unit or '**' in unit):
        return flow_rates_to_si(value, unit)

    # ---------------------------
    # Molality (mol/kg)
    # ---------------------------
    molality_units = ['mol/kg', 'mol/g', 'mmol/kg', 'mmol/g']
    if unit in molality_units:
        return molality_to_si(value, unit)

    # ---------------------------
    # Molarity (mol/m³)
    # ---------------------------
    molarity_units = ['mol/l', 'mol/ml', 'mmol/l', 'mmol/ml']
    if unit in molarity_units:
        return molarity_to_si(value, unit)

    # ---------------------------
    # Mass ↔ mol (requires molar mass)
    # ---------------------------
    if unit in ['mol', 'kg'] and molar_mass is not None:
        return mass_mole_convert(value, unit, molar_mass)

    raise ValueError(f"Unit '{unit}' not supported for SI conversion")
