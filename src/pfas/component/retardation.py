"""Retardation-factor assembly components."""

from typing import Annotated, Optional

from annotated_types import Gt
from pydantic import BaseModel, model_validator

from pfas.analytical_soln import Adsorption, HydrologicalProperties


class Retardation(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Calculate retardation factors from solid-phase and air-water interface sorption data.

    Parameters
    ----------
    Kd : float
        Solid-phase partition coefficient (m³/kg).
    Kaw : float
        Air-water interface partition coefficient (m²/m³).
    aaw : float
        Air-water interface area (m²/m³).
    kinetic : bool
        If True, the retardation factor is calculated for kinetic sorption.
        If False, it is calculated for equilibrium sorption.
    kin_params : dict, optional
        Required only when ``kinetic=True``.
    bulk_density : float
        Soil bulk density (kg/m³). Must be positive.
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    """

    Kd: float  # noqa: N815
    Kaw: float
    aaw: float
    kinetic: bool = False
    kin_params: Optional[dict] = None
    hydro_properties: HydrologicalProperties
    bulk_density: Annotated[float, Gt(0)]

    @model_validator(mode="after")
    def validate_kinetic_inputs(self) -> "Retardation":
        """Require kinetic parameters only when kinetic sorption is enabled."""
        if self.kinetic and (not isinstance(self.kin_params, dict) or not self.kin_params):
            raise ValueError("kin_params must be provided as a non-empty dict when kinetic=True.")
        return self

    def compute(self):
        """
        Assemble the Adsorption object.

        Returns
        -------
        dict
            Dictionary with key 'adsorption' containing an Adsorption instance.
        """
        awi_retardation = (self.Kaw * self.aaw) / self.hydro_properties.water_content
        sp_retardation = (self.bulk_density * self.Kd) / self.hydro_properties.water_content
        kin_params = self.kin_params or {}

        return {
            "adsorption": Adsorption(
                rate_const=kin_params.get("rate_const", 0.0),
                frac_int=kin_params.get("frac_int", 1.0),
                sp_retardation=sp_retardation,
                awi_retardation=awi_retardation,
                Kd=self.Kd,
            )
        }

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["adsorption"]
