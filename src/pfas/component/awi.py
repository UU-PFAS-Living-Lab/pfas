"""Air-water interface area preprocessing components."""

from typing import Annotated

from annotated_types import Gt, Interval
from pydantic import BaseModel, model_validator
from pint import Quantity
from pfas import ureg
from pfas.data_structure import HydrologicalProperties
from pfas.utils import (
    aaw_func_d50,
    aaw_func_GSSA,
    aaw_func_nonlinear_d50,
    aaw_func_thermo,
    aaw_func_tracer,
)

def _aaw_out(aaw):
    """Pass ``aaw`` through; if it has units, check it is an inverse length."""
    if isinstance(aaw, Quantity) and not aaw.check("1/[length]"):
        raise ValueError(f"aaw must have units of 1/length, got {aaw.units}")
    return aaw


class SWCsorption(
    BaseModel,
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Calculate air-water interface area using thermodynamic relations."""

    hydro_properties: HydrologicalProperties

    sigma0: Annotated[Quantity | float, Gt(0)] = 0.072 * ureg.newton / ureg.meter

    scaling_factor_awi: Annotated[float, Gt(0)]
    van_genuchten_n: Annotated[float, Gt(1)]
    van_genuchten_alpha: Annotated[Quantity | float, Gt(0)]

    porosity: Annotated[float, Interval(ge=0, le=1)]
    residual_water_content: Annotated[float, Interval(ge=0, le=1)]

    water_density: Annotated[Quantity | float, Gt(0)] = 1000 * ureg.kg / ureg.meter**3
    gravity: Annotated[Quantity | float, Gt(0)] = 9.81 * ureg.meter / ureg.second**2

    def compute(self) -> dict[str, Quantity | float]:
        """Calculate air-water interfacial area."""
        aaw = aaw_func_thermo(
            sigma0=self.sigma0,
            poro=self.porosity,
            alpha=self.van_genuchten_alpha,
            n=self.van_genuchten_n,
            th=self.hydro_properties.water_content,
            thr=self.residual_water_content,
            ths=self.porosity,
            sf=self.scaling_factor_awi,
            water_density=self.water_density,
            gravity=self.gravity,
        )
        return {"aaw": _aaw_out(aaw)}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["aaw"]


class GuoTracer(BaseModel, validate_assignment=True, extra="forbid"):
    """Calculate air-water interface area using the Guo et al. (2022) tracer relationship."""

    hydro_properties: HydrologicalProperties
    AWI: dict
    soil: dict

    @model_validator(mode="after")
    def validate_guo_inputs(self) -> "GuoTracer":
        """Validate the Guo AWI configuration."""
        if "Guo" not in self.AWI:
            raise ValueError("AWI must contain a 'Guo' entry.")

        guo_params = self.AWI["Guo"]
        if not isinstance(guo_params, dict):
            raise ValueError("AWI['Guo'] must be a dictionary.")

        missing = {"guo_x0", "guo_x1", "guo_x2"}.difference(guo_params)
        if missing:
            raise ValueError(
                f"AWI['Guo'] is missing required keys: {', '.join(sorted(missing))}"
            )
        return self

    def compute(self) -> dict[str, Quantity | float]:
        """Calculate air-water interfacial area."""
        guo = self.AWI["Guo"]
        aaw = aaw_func_tracer(
            self.hydro_properties.water_content,
            guo["guo_x2"],
            guo["guo_x1"],
            guo["guo_x0"],
        )
        return {"aaw": _aaw_out(aaw)}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["aaw"]


class GSSAAWI(BaseModel, validate_assignment=True, extra="forbid"):
    """Calculate air-water interfacial area using the GSSA-based linear model.

    Notes
    -----
    ``soil['d50']`` may be a Quantity (any length unit) or a plain number in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self) -> dict[str, Quantity | float]:
        """Calculate air-water interfacial area using the GSSA model."""
        poro = self.soil["porosity"]
        aaw = aaw_func_GSSA(
            d50=self.soil["d50"],
            poro=poro,
            th=self.hydro_properties.water_content,
            ths=poro,
        )
        return {"aaw": _aaw_out(aaw)}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["aaw"]


class D50AWI(BaseModel, validate_assignment=True, extra="forbid"):
    """Calculate air-water interfacial area using the d50 correlation.

    Notes
    -----
    ``soil['d50']`` may be a Quantity (any length unit) or a plain number in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self) -> dict[str, Quantity | float]:
        """Calculate air-water interfacial area using the d50 correlation."""
        aaw = aaw_func_d50(
            d50=self.soil["d50"],
            th=self.hydro_properties.water_content,
            ths=self.soil["porosity"],
        )
        return {"aaw": _aaw_out(aaw)}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["aaw"]


class NonlinearD50AWI(BaseModel, validate_assignment=True, extra="forbid"):
    """Calculate air-water interfacial area using the nonlinear d50 correlation.

    Notes
    -----
    ``soil['d50']`` may be a Quantity (any length unit) or a plain number in cm.
    """

    hydro_properties: HydrologicalProperties
    soil: dict

    def compute(self) -> dict[str, Quantity | float]:
        """Calculate air-water interfacial area using the nonlinear d50 correlation."""
        aaw = aaw_func_nonlinear_d50(
            d50=self.soil["d50"],
            th=self.hydro_properties.water_content,
            ths=self.soil["porosity"],
        )
        return {"aaw": _aaw_out(aaw)}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute()."""
        return ["aaw"]