"""Air-water interface area preprocessing components."""

from typing import Annotated

from annotated_types import Gt
from pydantic import BaseModel, model_validator

from pfas.analytical_soln import HydrologicalProperties
from pfas.utils import (
    aaw_func_thermo,
    aaw_func_tracer,
    aaw_func_GSSA,
    aaw_func_d50,
    aaw_func_nonlinear_d50,
)


class SWCsorption(BaseModel, validate_assignment=True, extra="forbid"):
    """
    Calculate air-water interface area using thermodynamic relations.

    Uses van Genuchten soil water characteristic curve to estimate
    air-water interfacial area from water saturation.
    """

    hydro_properties: HydrologicalProperties
    sigma0: Annotated[float, Gt(0)] = 0.072
    scaling_factor_awi: Annotated[float, Gt(0)]
    soil: dict

    def compute(self):
        """Calculate air-water interfacial area."""
        poro = self.soil["porosity"]
        alpha = self.soil["van_genuchten_alpha"]
        n_vg = self.soil["van_genuchten_n"]
        theta = self.hydro_properties.water_content
        thetar = self.soil["residual_water_content"]
        thetas = poro

        aaw = aaw_func_thermo(
            self.sigma0,
            poro,
            alpha,
            n_vg,
            theta,
            thetar,
            thetas,
            self.scaling_factor_awi,
        )

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]


class GuoTracer(BaseModel, validate_assignment=True, extra="forbid"):
    """
    Calculate air-water interface area using the Guo et al. (2022)
    tracer relationship.
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
        """Calculate air-water interfacial area."""
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


class GSSAAWI(BaseModel, validate_assignment=True, extra="forbid"):
    """
    Calculate air-water interfacial area using the GSSA-based
    linear model.

    The geometric smooth-surface specific solid surface area (GSSA)
    is calculated from porosity and median grain diameter and is
    assumed to represent the maximum possible air-water interfacial
    area.

    Parameters
    ----------
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    soil : dict
        Dictionary containing soil parameters:
        'porosity' and 'd50'.

    Notes
    -----
    The median grain diameter ``d50`` must be provided in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self):
        """Calculate air-water interfacial area using the GSSA model."""
        theta = self.hydro_properties.water_content
        ths = self.soil["porosity"]
        poro = self.soil["porosity"]
        d50 = self.soil["d50"]

        aaw = aaw_func_GSSA(
            d50=d50,
            poro=poro,
            th=theta,
            ths=ths,
        )

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]


class D50AWI(BaseModel, validate_assignment=True, extra="forbid"):
    """
    Calculate air-water interfacial area using the d50 correlation.

    Estimates the maximum air-water interfacial area from the median
    grain diameter and applies a linear dependence on water saturation.

    Parameters
    ----------
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    soil : dict
        Dictionary containing soil parameter 'd50'.

    Notes
    -----
    The median grain diameter ``d50`` must be provided in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self):
        """Calculate air-water interfacial area using the d50 correlation."""
        theta = self.hydro_properties.water_content
        ths = self.soil["porosity"]
        d50 = self.soil["d50"]

        aaw = aaw_func_d50(
            d50=d50,
            th=theta,
            ths=ths,
        )

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]


class NonlinearD50AWI(BaseModel, validate_assignment=True, extra="forbid"):
    """
    Calculate air-water interfacial area using the nonlinear
    d50 correlation.

    Estimates air-water interfacial area from the median grain
    diameter with an additional nonlinear saturation-dependent
    correction.

    Parameters
    ----------
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    soil : dict
        Dictionary containing soil parameter 'd50'.

    Notes
    -----
    The median grain diameter ``d50`` must be provided in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self):
        """Calculate air-water interfacial area using the nonlinear d50 correlation."""
        theta = self.hydro_properties.water_content
        ths = self.soil["porosity"]
        d50 = self.soil["d50"]

        aaw = aaw_func_nonlinear_d50(
            d50=d50,
            th=theta,
            ths=ths,
        )

        return {"aaw": aaw}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["aaw"]
