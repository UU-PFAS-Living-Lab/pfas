"""Air-water interface area preprocessing components."""

from typing import Annotated

from annotated_types import Gt
from pydantic import BaseModel, model_validator

from pfas.analytical_soln import HydrologicalProperties
from pfas.utils import aaw_func_thermo, aaw_func_tracer


class SWCsorption(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Calculate air-water interface area using thermodynamic relations.

    Uses van Genuchten soil water characteristic curve to estimate
    air-water interfacial area from water saturation.

    Parameters
    ----------
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    sigma0 : float, optional
        Surface tension of water (N/m). Default is 0.072.
    scaling_factor_awi : float
        Scaling factor for air-water interface area (dimensionless).
    AWI : dict
        Dictionary specifying AWI method and associated parameters.
    soil : dict
        Dictionary containing soil parameters: 'porosity', 'van_genuchten_alpha',
        'van_genuchten_n', and 'residual_water_content'.

    Attributes
    ----------
    outputs : list of str
        List containing 'aaw'.
    """

    hydro_properties: HydrologicalProperties
    sigma0: Annotated[float, Gt(0)] = 0.072
    scaling_factor_awi: Annotated[float, Gt(0)]
    AWI: dict
    soil: dict

    def compute(self):
        """
        Calculate air-water interfacial area based on the soil water characteristic curve.

        Returns
        -------
        dict
            Dictionary with key 'aaw' containing the calculated
            air-water interfacial area (m²/m³).
        """
        poro = self.soil["porosity"]
        alpha = self.soil["van_genuchten_alpha"]
        n_vg = self.soil["van_genuchten_n"]
        theta = self.hydro_properties.water_content
        thetar = self.soil["residual_water_content"]
        thetas = poro

        aaw = aaw_func_thermo(
                self.sigma0, poro, alpha, n_vg, theta, thetar, thetas, self.scaling_factor_awi
            )

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]



class GuoTracer(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Calculate air-water interface area using thermodynamic relations.

    Uses the Guo et al. (2022) tracer method to estimate total air-water interfacial area.

    Parameters
    ----------
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    AWI : dict
        Dictionary specifying AWI method and associated parameters. Specifically here
        x0, x1, and x2 for the Guo method.
    soil : dict
        Dictionary containing soil parameters: 'porosity', 'van_genuchten_alpha',
        'van_genuchten_n', and 'residual_water_content'.

    Attributes
    ----------
    outputs : list of str
        List containing 'aaw'.
    """

    hydro_properties: HydrologicalProperties
    AWI: dict
    soil: dict

    @model_validator(mode="after")
    def validate_guo_inputs(self) -> "GuoTracer":
        """Validate the Guo AWI configuration."""
        if not isinstance(self.AWI, dict):
            raise ValueError("AWI must be a dictionary.")
        if "Guo" not in self.AWI:
            raise ValueError("AWI must contain a 'Guo' entry.")

        guo_params = self.AWI["Guo"]
        if not isinstance(guo_params, dict):
            raise ValueError("AWI['Guo'] must be a dictionary.")

        required_keys = {"guo_x0", "guo_x1", "guo_x2"}
        missing_keys = required_keys.difference(guo_params.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"AWI['Guo'] is missing required keys: {missing}")
        return self

    def compute(self):
        """
        Calculate air-water interfacial area based on the soil water characteristic curve.

        Returns
        -------
        dict
            Dictionary with key 'aaw' containing the calculated
            air-water interfacial area (m²/m³).
        """
        theta = self.hydro_properties.water_content

        guo_params = self.AWI["Guo"]
        x0 = guo_params["guo_x0"]
        x1 = guo_params["guo_x1"]
        x2 = guo_params["guo_x2"]
        aaw = aaw_func_tracer(theta, x2, x1, x0)

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]


