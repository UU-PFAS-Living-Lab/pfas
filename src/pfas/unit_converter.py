import re

def ratio_to_si(value, unit):
    """
    Convert units like 'L/kg', 'mL/g' to SI.
    """
    volume_units = {'L': 1e-3, 'mL': 1e-6, 'm³': 1}  # all to m³
    mass_units = {'kg': 1, 'g': 1e-3}  # all to kg

    if '/' in unit:
        num, denom = unit.split('/')
        # Convert numerator
        if num in volume_units:
            value *= volume_units[num]
            num = 'm³'
        # Convert denominator
        if denom in mass_units:
            value /= 1  # value already per kg after adjusting numerator
            denom = 'kg'
        return value, f'{num}/{denom}'
    else:
        raise ValueError("Unit not a ratio")
    
def compound_to_si(value, unit):
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
        power = int(power) if power else 1
        if base in length_units:
            si_value *= length_units[base]**power
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


def to_si(value, unit):
    """
    Convert a value from a given unit to its SI unit.
    
    Supported units:
    Length: 'cm', 'mm', 'km', 'inch', 'foot', 'mile'
    Mass: 'g', 'mg', 'lb', 'oz'
    Temperature: 'C', 'F'
    
    Returns a tuple: (converted_value, SI_unit)
    """
    
    # Length conversions to meters
    length_units = {
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'km': 1000,
        'inch': 0.0254,
        'foot': 0.3048,
        'mile': 1609.34
    }
    
    # Mass conversions to kilograms
    mass_units = {
        'kg': 1,
        'g': 0.001,
        'mg': 1e-6,
        'lb': 0.453592,
        'oz': 0.0283495
    }
    
    # Temperature conversions to Kelvin
    if unit in ['C', 'c']:
        return (value + 273.15, 'K')
    elif unit in ['F', 'f']:
        return ((value - 32) * 5/9 + 273.15, 'K')
    
    # Check if unit is a length
    if unit in length_units:
        return (value * length_units[unit], 'm')
    
    # Check if unit is a mass
    if unit in mass_units:
        return (value * mass_units[unit], 'kg')
    
    raise ValueError(f"Unit '{unit}' not supported for SI conversion")
