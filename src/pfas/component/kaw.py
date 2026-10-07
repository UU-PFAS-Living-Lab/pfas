"""Air-water partition coefficient preprocessing components."""

from typing import Annotated

from annotated_types import Gt
from pint import Quantity
from pydantic import BaseModel, model_validator
from pfas import ureg

from pfas.utils import (
    Kaw_0_Le2021,
    Kaw_langmuir_Le2021,
    Kaw_Szyszkowski,
    dG0_Le2021,
)


class Le2021_asymptote(
    BaseModel,
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
):  # noqa: N801
    """
    Compute the dilute-limit air-water partition coefficient.

    Calculates the dilute-limit air-water partition coefficient according
    to the group-contribution approach of Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary of PFAS molecular group counts.

    Attributes
    ----------
    outputs : list of str
        List containing 'Kaw'.
    """

    structural_properties: dict[str, float | int]

    _REQUIRED_STRUCTURAL_KEYS = {
        "n_CFx",
        "n_CHx",
        "n_COO",
        "n_COOH",
        "n_SO3",
        "n_R4N",
        "n_OH",
        "n_OSO3",
        "n__O_",
        "n__S_",
        "n_N_CH3_2_CH2_COO",
    }

    @model_validator(mode="after")
    def validate_structural_properties(self) -> "Le2021_asymptote":
        """Validate that all required structural-group keys are available."""
        missing = self._REQUIRED_STRUCTURAL_KEYS.difference(
            self.structural_properties
        )

        if missing:
            raise ValueError(
                "structural_properties is missing required keys: "
                + ", ".join(sorted(missing))
            )

        return self

    @property
    def kaw(self) -> float:
        """Dilute-limit air-water partition coefficient from Le et al. (2021)."""
        return Kaw_0_Le2021(self.structural_properties)

    def compute(self) -> dict[str, float]:
        """
        Calculate the dilute-limit air-water partition coefficient.

        Returns
        -------
        dict
            Dictionary with key 'Kaw'.
        """
        return {"Kaw": self.kaw}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["Kaw"]


class Le2021_langmuir(
    BaseModel,
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
):  # noqa: N801
    """
    Compute concentration-dependent air-water partitioning using Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary of PFAS molecular group counts.

    Cw : float or pint.Quantity
        Aqueous-phase concentration. Plain floats are interpreted as mol/L.

    omega : float or pint.Quantity, optional
        Water molar concentration. Plain floats are interpreted as mol/L.
        Default is 55.3 mol/L.

    T : float or pint.Quantity, optional
        Temperature. Plain floats are interpreted as K. Default is 298 K.

    Kaw_unit : str, optional
        Unit associated with the Le et al. regression value of $$K_{aw}$$.
        Confirm this against the source publication. Default is centimetre.

    Attributes
    ----------
    outputs : list of str
        List containing 'Kaw'.
    """

    structural_properties: dict[str, float | int]

    # Plain float convention: mol/L.
    Cw: Annotated[Quantity | float, Gt(0)]

    # Plain float convention: mol/L.
    omega: Annotated[Quantity | float, Gt(0)] = (
        55.3 * ureg.mole / ureg.liter
    )
    T: Annotated[Quantity | float, Gt(0)] = 298.0 * ureg.kelvin

    _REQUIRED_STRUCTURAL_KEYS = {
        "n_CFx",
        "n_CHx",
        "n_COO",
        "n_COOH",
        "n_SO3",
        "n_R4N",
        "n_OH",
        "n_OSO3",
        "n__O_",
        "n__S_",
        "n_N_CH3_2_CH2_COO",
    }

    @model_validator(mode="after")
    def validate_structural_properties(self) -> "Le2021_langmuir":
        """Validate that all required structural-group keys are available."""
        missing = self._REQUIRED_STRUCTURAL_KEYS.difference(
            self.structural_properties
        )

        if missing:
            raise ValueError(
                "structural_properties is missing required keys: "
                + ", ".join(sorted(missing))
            )

        return self

    @property
    def Kaw_0(self) -> float:  # noqa: N802
        """Dilute-limit air-water partition coefficient."""
        return Kaw_0_Le2021(self.structural_properties)

    @property
    def dG0(self) -> float:  # noqa: N802
        """Gibbs free energy of adsorption in kJ/mol."""
        return dG0_Le2021(self.structural_properties)

    def Kaw(self) -> Quantity | float:  # noqa: N802
        """Calculate the concentration-dependent air-water partition coefficient."""
        return Kaw_langmuir_Le2021(
            Kaw_0=self.Kaw_0,
            dG0=self.dG0,
            Cw=self.Cw,
            omega=self.omega,
            T=self.T,
        )

    def compute(self) -> dict[str, Quantity | float]:
        """
        Calculate the air-water partition coefficient.

        Returns
        -------
        dict
            Dictionary with key 'Kaw'.
        """
        return {"Kaw": self.Kaw()}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["Kaw"]



class Szyszkowski(
    BaseModel,
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """
    Compute the air-water partition coefficient using the Szyszkowski model.

    Uses a Szyszkowski-based air-water partition coefficient with
    parameters from Guo et al. (2022).

    Parameters
    ----------
    sigma0 : float or pint.Quantity, optional
        Surface tension of water. Plain floats are interpreted as N/m.
        Default is 0.072 N/m.

    a : float or pint.Quantity
        Szyszkowski fitting parameter. Plain floats are interpreted as mol/L.
        Pint quantities must have concentration units compatible with mol/L.

    b : float
        Dimensionless Szyszkowski fitting parameter.

    chi : float, optional
        Ionisation coefficient. Use 1 for nonionic PFAS or ionic PFAS
        with swamping electrolyte, and 2 for ionic PFAS without
        swamping electrolyte. Default is 2.

    T : float or pint.Quantity, optional
        Temperature. Plain floats are interpreted as K. Pint quantities must
        have temperature units compatible with K. Default is 298 K.

    Cw : float or pint.Quantity
        Aqueous-phase concentration. Plain floats are interpreted as mol/L.
        Pint quantities must have concentration units compatible with mol/L.

    Attributes
    ----------
    outputs : list of str
        List containing 'Kaw'.
    """

    # Plain float convention: N/m.
    # Float default ensures fully unitless input remains fully unitless.
    sigma0: Annotated[Quantity | float, Gt(0)] = (
            0.072 * ureg.newton / ureg.meter
        )

    # Plain float convention: mol/L.
    a: Annotated[Quantity | float, Gt(0)]

    # Dimensionless.
    b: Annotated[float, Gt(0)]

    # Dimensionless ionisation coefficient.
    chi: Annotated[float, Gt(0)] = 2.0

    # Default: 298 K.
    # Plain float convention: K.
    T: Annotated[Quantity | float, Gt(0)] = 298.0 * ureg.kelvin

    # Plain float convention: mol/L.
    Cw: Annotated[Quantity | float, Gt(0)]
    def Kaw(self) -> Quantity | float:  # noqa: N802
        """
        Calculate Kaw from aqueous concentration and Szyszkowski parameters.

        Returns
        -------
        float or pint.Quantity
            Air-water partition coefficient at the specified concentration.
        """
        return Kaw_Szyszkowski(
            sigma0=self.sigma0,
            a=self.a,
            b=self.b,
            Cw=self.Cw,
            chi=self.chi,
            T=self.T,
        )

    def compute(self) -> dict[str, Quantity | float]:
        """
        Calculate the air-water partition coefficient at a given concentration.

        Returns
        -------
        dict
            Dictionary with key 'Kaw'.
        """
        return {"Kaw": self.Kaw()}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute() method."""
        return ["Kaw"]