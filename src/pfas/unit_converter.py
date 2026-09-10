"""Class to convert units."""
#ruff: noqa: PLR0912
import re


class UnitConverter:
    """
    Convert a wide range of physical and chemical units to their SI equivalents.

    Supported quantities
    --------------------
    - Length → m
    - Mass → kg
    - Temperature → K
    - Density → kg/m**3
    - Molar mass → kg/mol
    - Inverse length → 1/m
    - Inverse time → 1/s
    - Volume ratios → m**3/kg
    - Flow rates → SI units
    - Molality → mol/kg
    - Molarity → mol/m**3
    - Mass ↔ mol when a molar mass is provided

    Examples
    --------
    >>> UnitConverter.to_si(10, "cm")
    (0.1, "m")

    >>> UnitConverter.to_si(500, "g")
    (0.5, "kg")

    >>> UnitConverter.to_si(25, "C")
    (298.15, "K")

    >>> UnitConverter.to_si(5, "mol/g")
    (5000.0, "mol/kg")

    >>> UnitConverter.to_si(2, "mmol/mL")
    (2000.0, "mol/m**3")

    >>> UnitConverter.to_si(1, "L/kg")
    (0.001, "m**3/kg")

    >>> UnitConverter.to_si(1.65, "g/cm**3")
    (1650.0, "kg/m**3")

    >>> UnitConverter.to_si(5.9, "1/h")
    (0.001638888888888889, "1/s")

    >>> UnitConverter.to_si(0.04479, "1/cm")
    (4.479, "1/m")

    >>> UnitConverter.to_si(414.07, "g/mol")
    (0.41407, "kg/mol")

    >>> UnitConverter.to_si(0.1, "kg", molar_mass=0.05844)
    (1.7111567419575635, "mol")

    >>> UnitConverter.to_si(2, "mol", molar_mass=0.05844)
    (0.11688, "kg")
    """

    # ------------------------------------------------------------------
    # Shared unit tables
    # ------------------------------------------------------------------

    LENGTH_UNITS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "km": 1000.0,
        "inch": 0.0254,
        "foot": 0.3048,
        "mile": 1609.34,
    }

    MASS_UNITS = {
        "kg": 1.0,
        "g": 1e-3,
        "mg": 1e-6,
        "lb": 0.453592,
        "oz": 0.0283495,
    }

    MOL_PREFIX = {
        "mol": 1.0,
        "mmol": 1e-3,
        "µmol": 1e-6,
        "umol": 1e-6,
    }

    VOLUME_UNITS = {
        "l": 1e-3,
        "ml": 1e-6,
        "m**3": 1.0,
    }

    # Density → kg/m**3
    DENSITY_UNITS = {
        "kg/m**3": 1.0,
        "g/cm**3": 1000.0,
        "kg/l": 1000.0,
        "g/l": 1.0,
    }

    # Molar mass → kg/mol
    MOLAR_MASS_UNITS = {
        "kg/mol": 1.0,
        "g/mol": 1e-3,
        "mg/mol": 1e-6,
    }

    # Inverse time → 1/s
    INVERSE_TIME_UNITS = {
        "1/s": 1.0,
        "1/min": 1.0 / 60.0,
        "1/h": 1.0 / 3600.0,
    }

    # Inverse length → 1/m
    INVERSE_LENGTH_UNITS = {
        "1/m": 1.0,
        "1/cm": 100.0,
        "1/mm": 1000.0,
    }

    FLOW_LENGTH_UNITS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
    }

    FLOW_TIME_UNITS = {
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
    }

    # ------------------------------------------------------------------
    # Flow-rate detection
    # ------------------------------------------------------------------

    @classmethod
    def is_flow_rate_unit(cls, unit):
        """
        Check whether a unit has the form of a length/volume rate.

        Examples
        --------
        cm/s
        cm**2/s
        mm**3/min
        m/min
        """
        bases = re.findall(r"[a-zA-Z]+", unit)

        return (
            len(bases) >= 2
            and all(
                b in cls.FLOW_LENGTH_UNITS
                or b in cls.FLOW_TIME_UNITS
                for b in bases
            )
        )

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    @classmethod
    def density_to_si(cls, value, unit):
        """Convert density to kg/m**3."""
        unit = unit.lower()

        if unit not in cls.DENSITY_UNITS:
            raise ValueError(f"Unsupported density unit '{unit}'")

        value *= cls.DENSITY_UNITS[unit]

        return value, "kg/m**3"

    # ------------------------------------------------------------------
    # Molar mass
    # ------------------------------------------------------------------

    @classmethod
    def molar_mass_to_si(cls, value, unit):
        """Convert molar mass to kg/mol."""
        unit = unit.lower()

        if unit not in cls.MOLAR_MASS_UNITS:
            raise ValueError(f"Unsupported molar mass unit '{unit}'")

        value *= cls.MOLAR_MASS_UNITS[unit]

        return value, "kg/mol"

    # ------------------------------------------------------------------
    # Inverse length
    # ------------------------------------------------------------------

    @classmethod
    def inverse_length_to_si(cls, value, unit):
        """Convert inverse length to 1/m."""
        unit = unit.lower()

        if unit not in cls.INVERSE_LENGTH_UNITS:
            raise ValueError(
                f"Unsupported inverse length unit '{unit}'"
            )

        value *= cls.INVERSE_LENGTH_UNITS[unit]

        return value, "1/m"

    # ------------------------------------------------------------------
    # Inverse time
    # ------------------------------------------------------------------

    @classmethod
    def inverse_time_to_si(cls, value, unit):
        """Convert inverse time to 1/s."""
        unit = unit.lower()

        if unit not in cls.INVERSE_TIME_UNITS:
            raise ValueError(
                f"Unsupported inverse time unit '{unit}'"
            )

        value *= cls.INVERSE_TIME_UNITS[unit]

        return value, "1/s"

    # ------------------------------------------------------------------
    # Molality
    # ------------------------------------------------------------------

    @classmethod
    def molality_to_si(cls, value, unit):
        """Convert molality to mol/kg."""
        unit = unit.lower()

        num, denom = unit.split("/")

        if num not in cls.MOL_PREFIX:
            raise ValueError(f"Unsupported mol prefix '{num}'")

        if denom not in ("kg", "g"):
            raise ValueError(f"Unsupported mass unit '{denom}'")

        value *= cls.MOL_PREFIX[num]

        # Convert denominator to kg
        value /= 1.0 if denom == "kg" else 1e-3

        return value, "mol/kg"

    # ------------------------------------------------------------------
    # Molarity
    # ------------------------------------------------------------------

    @classmethod
    def molarity_to_si(cls, value, unit):
        """Convert molarity to mol/m**3."""
        unit = unit.lower()

        num, denom = unit.split("/")

        if num not in cls.MOL_PREFIX:
            raise ValueError(f"Unsupported mol prefix '{num}'")

        if denom not in cls.VOLUME_UNITS:
            raise ValueError(
                f"Unsupported volume unit '{denom}'"
            )

        value *= cls.MOL_PREFIX[num]

        # Convert denominator volume to m**3
        value /= cls.VOLUME_UNITS[denom]

        return value, "mol/m**3"

    # ------------------------------------------------------------------
    # Mass ↔ mol
    # ------------------------------------------------------------------

    @staticmethod
    def mass_mole_convert(value, unit, molar_mass):
        """
        Convert between kg and mol.

        Parameters
        ----------
        value : float
            Numeric value.
        unit : str
            Either "kg" or "mol".
        molar_mass : float
            Molar mass in kg/mol.
        """
        if unit == "kg":
            return value / molar_mass, "mol"

        if unit == "mol":
            return value * molar_mass, "kg"

        raise ValueError("Unit must be 'kg' or 'mol'")

    # ------------------------------------------------------------------
    # Volume ratios
    # ------------------------------------------------------------------

    @classmethod
    def volume_to_si(cls, value, unit):
        """
        Convert volume per mass to m**3/kg.

        Examples
        --------
        L/kg
        mL/kg
        L/g
        mL/g
        """
        num, denom = unit.split("/")

        num = num.lower()
        denom = denom.lower()

        if num not in cls.VOLUME_UNITS:
            raise ValueError(
                f"Unsupported volume unit '{num}'"
            )

        if denom not in ("kg", "g"):
            raise ValueError(
                f"Unsupported mass unit '{denom}'"
            )

        value *= cls.VOLUME_UNITS[num]

        # Convert denominator to kg
        value /= 1.0 if denom == "kg" else 1e-3

        return value, "m**3/kg"

    # ------------------------------------------------------------------
    # Flow rates
    # ------------------------------------------------------------------

    @classmethod
    def flow_rates_to_si(cls, value, unit):
        """
        Convert flow/rate units to SI.

        Examples
        --------
        cm/s      → m/s
        cm**2/s   → m**2/s
        mm**3/min → m**3/s
        """
        # Parse units such as:
        # cm/s
        # cm**2/s
        # mm**3/min
        match = re.findall(
            r"([a-zA-Z]+)(?:\*\*(\d+))?",
            unit,
        )

        if not match:
            raise ValueError(f"Could not parse unit '{unit}'")

        si_value = value

        # Track numerator/denominator position
        numerator, denominator = unit.split("/")

        # ----------------------------
        # Numerator
        # ----------------------------

        num_match = re.fullmatch(
            r"([a-zA-Z]+)(?:\*\*(\d+))?",
            numerator,
        )

        if num_match is None:
            raise ValueError(
                f"Could not parse numerator '{numerator}'"
            )

        base = num_match.group(1).lower()
        power = (
            int(num_match.group(2))
            if num_match.group(2)
            else 1
        )

        if base not in cls.FLOW_LENGTH_UNITS:
            raise ValueError(
                f"Unit '{base}' not supported in flow rate"
            )

        si_value *= cls.FLOW_LENGTH_UNITS[base] ** power

        # ----------------------------
        # Denominator
        # ----------------------------

        denom_match = re.fullmatch(
            r"([a-zA-Z]+)",
            denominator,
        )

        if denom_match is None:
            raise ValueError(
                f"Could not parse denominator '{denominator}'"
            )

        time_base = denom_match.group(1).lower()

        if time_base not in cls.FLOW_TIME_UNITS:
            raise ValueError(
                f"Time unit '{time_base}' not supported"
            )

        # Divide by seconds
        si_value /= cls.FLOW_TIME_UNITS[time_base]

        # Construct SI unit
        if power == 1:
            si_unit = "m"
        else:
            si_unit = f"m**{power}"

        si_unit += "/s"

        return si_value, si_unit

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    # ruff: noqa: PLR0911
    @classmethod
    def to_si(cls, value, unit, molar_mass=None):
        """Convert a value to its SI equivalent."""
        unit = unit.strip().lower()

        # --------------------------------------------------------------
        # Temperature
        # --------------------------------------------------------------

        if unit == "c":
            return value + 273.15, "K"

        if unit == "f":
            return (value - 32) * 5 / 9 + 273.15, "K"

        # --------------------------------------------------------------
        # Length
        # --------------------------------------------------------------

        if unit in cls.LENGTH_UNITS:
            return (
                value * cls.LENGTH_UNITS[unit],
                "m",
            )

        # --------------------------------------------------------------
        # Molar mass
        # --------------------------------------------------------------

        if unit in cls.MOLAR_MASS_UNITS:
            return cls.molar_mass_to_si(value, unit)

        # --------------------------------------------------------------
        # Mass ↔ mol
        # --------------------------------------------------------------

        if unit in ("mol", "kg") and molar_mass is not None:
            return cls.mass_mole_convert(
                value,
                unit,
                molar_mass,
            )

        # --------------------------------------------------------------
        # Mass
        # --------------------------------------------------------------

        if unit in cls.MASS_UNITS:
            return (
                value * cls.MASS_UNITS[unit],
                "kg",
            )

        # --------------------------------------------------------------
        # Density
        # --------------------------------------------------------------

        if unit in cls.DENSITY_UNITS:
            return cls.density_to_si(value, unit)

        # --------------------------------------------------------------
        # Inverse length
        # --------------------------------------------------------------

        if unit in cls.INVERSE_LENGTH_UNITS:
            return cls.inverse_length_to_si(value, unit)

        # --------------------------------------------------------------
        # Inverse time
        # --------------------------------------------------------------

        if unit in cls.INVERSE_TIME_UNITS:
            return cls.inverse_time_to_si(value, unit)

        # --------------------------------------------------------------
        # Molality
        # --------------------------------------------------------------

        if unit in (
            "mol/kg",
            "mol/g",
            "mmol/kg",
            "mmol/g",
        ):
            return cls.molality_to_si(value, unit)

        # --------------------------------------------------------------
        # Molarity
        # --------------------------------------------------------------

        if unit in (
            "mol/l",
            "mol/ml",
            "mmol/l",
            "mmol/ml",
        ):
            return cls.molarity_to_si(value, unit)

        # --------------------------------------------------------------
        # Volume ratios
        # --------------------------------------------------------------

        if "/" in unit:
            num, _ = unit.split("/")

            if num in ("l", "ml"):
                return cls.volume_to_si(value, unit)

        # --------------------------------------------------------------
        # Flow rates
        # --------------------------------------------------------------

        if cls.is_flow_rate_unit(unit):
            return cls.flow_rates_to_si(value, unit)

        # --------------------------------------------------------------
        # Unsupported
        # --------------------------------------------------------------

        raise ValueError(
            f"Unit '{unit}' not supported for SI conversion"
        )
