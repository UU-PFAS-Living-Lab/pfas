"""Air-water partition coefficient preprocessing components."""

from typing import Annotated

from pfas.utils import Kaw_0_Le2021, Kaw_langmuir_Le2021, dG0_Le2021, Kaw_Szyszkowski
from pydantic import BaseModel,  model_validator
from annotated_types import Gt
class Le2021_asymptote(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Compute adsorption parameters for air-water interface.

    Calculates retardation factor for sorption at the air-water interface
    using a calculated air-water partition coefficient, according to
    Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary of PFAS molecular group counts, with keys:
        'n_CFx', 'n_CHx', 'n_COO', 'n_COOH', 'n_SO3', 'n_R4N', 'n_OH',
        'n_OSO3', 'n__O_', 'n__S_', 'n_N_CH3_2_CH2_COO'.

    Attributes
    ----------
    outputs : list of str
        List containing 'awi_retardation'.
    """

structural_properties: dict

_REQUIRED_STRUCTURAL_KEYS = {
    "n_CFx", "n_CHx", "n_COO", "n_COOH", "n_SO3", "n_R4N",
    "n_OH", "n_OSO3", "n__O_", "n__S_", "n_N_CH3_2_CH2_COO",
}

@model_validator(mode="after")
def validate_structural_properties(self):
    missing = self._REQUIRED_STRUCTURAL_KEYS.difference(self.structural_properties.keys())
    if missing:
        raise ValueError(
            f"structural_properties is missing required keys: {', '.join(sorted(missing))}"
        )
    return self

@property
def kaw(self) -> float:
    """Air-water partition coefficient (m) from group contributions."""
    return Kaw_0_Le2021(self.structural_properties)

def compute(self) -> dict:
    """
    Calculate the air-water partition coefficient.

    Returns
    -------
    dict
        Dictionary with key 'Kaw'.
    """
    return {"Kaw": self.kaw}

@property
def outputs(self) -> list[str]:
    """List of output keys from compute() method."""
    return ["Kaw"]


from pydantic import BaseModel, ConfigDict, model_validator


class Le2021_langmuir(BaseModel):
    """
    Compute the air-water partition coefficient using the
    Langmuir isotherm from Le et al. (2021).

    Parameters
    ----------
    structural_properties : dict
        Dictionary of PFAS molecular group counts.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    structural_properties: dict

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
    def validate_structural_properties(self):
        missing = (
            self._REQUIRED_STRUCTURAL_KEYS
            - self.structural_properties.keys()
        )

        if missing:
            raise ValueError(
                "structural_properties is missing required keys: "
                + ", ".join(sorted(missing))
            )

        return self

    @property
    def Kaw_0(self) -> float:
        """Dilute-limit air-water partition coefficient."""
        return Kaw_0_Le2021(
            self.structural_properties
        )

    @property
    def dG0(self) -> float:
        """Gibbs free energy of adsorption."""
        return dG0_Le2021(
            self.structural_properties
        )

    def Kaw(self, Cw: float) -> float:
        """
        Calculate the concentration-dependent air-water
        partition coefficient.
        """
        return Kaw_langmuir_Le2021(
            self.Kaw_0,
            self.dG0,
            Cw,
        )

    def compute(self, Cw: float) -> dict:
        """
        Calculate the air-water partition coefficient.

        Parameters
        ----------
        Cw : float
            Aqueous-phase concentration (mol/L).

        Returns
        -------
        dict
            Dictionary containing 'Kaw'.
        """
        kaw = self.Kaw(Cw)

        return {
            "Kaw": kaw
        }

    @property
    def outputs(self) -> list[str]:
        """List of output keys."""
        return ["Kaw"]

class Szyszkowski(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Compute air-water interfacial retardation using a Szyszkowski-based
    air-water partition coefficient with parameters from Guo et al. (2022).

    Parameters
    ----------
    sigma0 : float, optional
        Surface tension of water (N/m). Default is 0.072.
    a : float
        Szyszkowski fitting parameter (mol/L).
    b : float
        Szyszkowski fitting parameter (dimensionless).
    chi : int, optional
        Ionisation coefficient. Use 1 for nonionic PFAS or ionic PFAS
        with swamping electrolyte, and 2 for ionic PFAS without
        swamping electrolyte. Default is 2.
    T : float, optional
        Temperature (K). Default is 298 K.
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    aaw : float
        Air-water interfacial area (cm²/cm³).

    Attributes
    ----------
    outputs : list of str
        List containing 'awi_retardation'.
    """

    sigma0: Annotated[float, Gt(0)] = 0.072
    a: float
    b: float
    chi: int = 2
    T: Annotated[float, Gt(0)] = 298.0

    def Kaw(self, Cw: float) -> float:
        """
        Calculate Kaw from aqueous concentration and Szyszkowski fitting parameters

        Parameters
        ----------
        Cw : float
            Aqueous-phase concentration of the PFAS compound (mol/L).

        Returns
        -------
        float
            Kaw at the given concentration.
        """
        return Kaw_Szyszkowski(
            sigma0=self.sigma0,
            a=self.a,
            b=self.b,
            Cw=Cw,
            chi=self.chi,
            T=self.T,
        )

    def compute(self, Cw: float) -> dict:
        """
        Calculate the air-water partition coefficient at a given concentration.

        Parameters
        ----------
        Cw : float
            Aqueous-phase concentration of the PFAS compound (mol/L),
            typically C_list[j] for the active time interval.

        Returns
        -------
        dict
            Dictionary with key 'Kaw'.
        """
        kaw = self.Kaw(Cw)
        return {"Kaw": kaw}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute() method."""
        return ["Kaw"]
