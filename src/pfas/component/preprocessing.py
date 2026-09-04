"""
PFAS Preproccesing Module.

==========================================

This module contains preprocessor classes for preparing input parameters
for the PFAS analytical solver.

The preprocessors handle calculations for:
- Water flow properties
- Boundary conditions
- Spatial and temporal grids
Classes
-------
WaterPreprocessor
    Computes hydraulic properties from infiltration and soil parameters.
BoundaryPreprocessor
    Calculates boundary conditions for contaminant input.
GridGenerator
    Generates spatial and temporal discretization grids.
"""

from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt, Interval
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.optimize import brentq

from pfas.data_structure import BoundaryConditions, HydrologicalProperties, SimulationGrid


class WaterPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Compute hydraulic properties from infiltration rate and soil parameters.

    This class calculates volumetric water content, pore water velocity, and
    dispersion coefficient based on van Genuchten soil hydraulic parameters
    and steady-state infiltration conditions.

    Parameters
    ----------
    average_infiltration_rate : float
        Average water infiltration rate (m/s). Must be positive.
    hydraulic_conductivity : float
        Saturated hydraulic conductivity (m/s). Must be positive.
    porosity : float
        Soil porosity (dimensionless). Range: [0, 1].
    dispersivity : float
        Longitudinal dispersivity (m). Must be positive.
    van_genuchten_n : float
        van Genuchten n parameter (dimensionless). Must be positive.
    van_genuchten_l : float, optional
        van Genuchten l parameter (dimensionless). Must be positive.
        If not provided, or if null/None, defaults to 0.5 (the standard
        Mualem assumption).
    residual_water_content : float
        Residual water content (dimensionless). Range: [0, 1].

    Attributes
    ----------
    outputs : list of str
        List containing 'hydro_properties'.

    Examples
    --------
    >>> preprocessor = WaterPreprocessor(
    ...     average_infiltration_rate=1e-8,
    ...     hydraulic_conductivity=1e-5,
    ...     porosity=0.4,
    ...     dispersivity=0.1,
    ...     van_genuchten_n=2.0,
    ...     van_genuchten_l=None,
    ...     residual_water_content=0.05
    ... )
    >>> result = preprocessor.compute()
    >>> 'hydro_properties' in result
    True
    """

    average_infiltration_rate: Annotated[float, Gt(0)]
    hydraulic_conductivity: Annotated[float, Gt(0)]
    porosity: Annotated[float, Interval(ge=0, le=1)]
    dispersivity: Annotated[float, Gt(0)]
    van_genuchten_n: Annotated[float, Gt(0)]
    van_genuchten_l: float = 0.5
    residual_water_content: Annotated[float, Interval(ge=0, le=1)]

    @field_validator("van_genuchten_l", mode="before")
    @classmethod
    def default_l_when_null(cls, v):
        """Treat None/'null' as 'not provided' and fall back to 0.5."""
        if v is None or (isinstance(v, str) and v.strip().lower() == "null"):
            return 0.5
        return v

    def compute(self):
        """
        Calculate hydraulic properties.

        Uses van Genuchten relative permeability function to solve for
        effective saturation, then computes water content, velocity, and
        dispersion coefficient.

        Returns
        -------
        dict
            Dictionary with key 'hydro_properties' containing a
            HydrologicalProperties instance with theta, v, and D values.
        """
        kr = self.average_infiltration_rate / self.hydraulic_conductivity
        m = 1 - 1 / self.van_genuchten_n

        def relperm(se):
            return se**self.van_genuchten_l * (1 - (1 - se**(1/m))**m)**2 - kr

        se = brentq(relperm, 1e-12, 1.0)
        theta = se * (self.porosity - self.residual_water_content) + self.residual_water_content
        v = self.average_infiltration_rate / theta
        d = v * self.dispersivity
        return {"hydro_properties": HydrologicalProperties(theta, v, d)}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["hydro_properties"]

class BoundaryPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    """Calculate boundary conditions for contaminant input.

    Converts solute concentrations and switching times into the C_list and
    T_list format required by the analytical solution.

    Parameters
    ----------
    C_list : list of float
        Inlet concentrations for each interval [M L⁻³]. ``C_list[j]`` is
        the concentration active from ``T_list[j]`` until ``T_list[j+1]``.
        The last interval extends to infinity. Must have the same length
        as ``T_list``. Use 0 for clean water with no PFAS input. Examples:
        - Continuous step:   ``C_list=[C0],          T_list=[0]``
        - Pulse from t=0:    ``C_list=[C0, 0],       T_list=[0, t1]``
        - Delayed pulse:     ``C_list=[0, C0, 0],    T_list=[0, t1, t2]``
        - Multiple pulses:   ``C_list=[f1, 0, f2],   T_list=[0, t1, t2]``
    T_list : list of float
        Switching times [T] at which the inlet concentration changes.
        Must have the same length as ``C_list``. ``T_list[0]`` must be 0.

    Attributes
    ----------
    outputs : list of str
        List containing 'boundary_conditions'.
    """

    C_list: list[Annotated[float, Ge(0), Field(description="[M L⁻³]")]]
    T_list: list[Annotated[float, Ge(0), Field(description="[T]")]]

    @model_validator(mode="after")
    def validate_c_and_t_list(self) -> "BoundaryPreprocessor":
        """Validate C_list and T_list are consistent."""
        if not self.T_list:
            raise ValueError("T_list must contain at least one entry.")
        if not self.C_list:
            raise ValueError("C_list must contain at least one entry.")
        if len(self.C_list) != len(self.T_list):
            raise ValueError(
                f"C_list (len={len(self.C_list)}) and T_list (len={len(self.T_list)}) "
                "must have the same length."
            )
        if self.T_list[0] != 0:
            raise ValueError("T_list[0] must be 0.")
        if any(self.T_list[i] >= self.T_list[i + 1] for i in range(len(self.T_list) - 1)):
            raise ValueError("T_list must be strictly increasing.")
        return self
    def compute(self) -> dict:
        """Pass through C_list and T_list as boundary conditions.

        Returns
        -------
        dict
            Dictionary with key 'boundary_conditions' containing a
            :class:`BoundaryConditions` instance.
        """
        bc = BoundaryConditions(
            C_list=self.C_list,
            T_list=self.T_list,
        )
        return {"boundary_conditions": bc}

    @property
    def outputs(self) -> list[str]:
        """List of output keys from compute() method."""
        return ["boundary_conditions"]


class GridGenerator(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Generate spatial and temporal discretization grids.

    Creates uniform grids for depth and time used in the analytical solution.

    Parameters
    ----------
    domain_length : float
        Total depth of domain (m). Must be positive.
    spatial_resolution : float
        Grid spacing in depth direction (m). Must be positive.
    time_resolution : float
        Time step size (s). Must be positive.
    time_total : float
        Total simulation time (s). Must be positive.

    Attributes
    ----------
    outputs : list of str
        List containing 'grid'.
    """

    domain_length: Annotated[float, Gt(0)]
    spatial_resolution: Annotated[float, Gt(0)]
    time_resolution: Annotated[float, Gt(0)]
    time_total: Annotated[float, Gt(0)]

    def compute(self):
        """
        Generate grid arrays.

        Returns
        -------
        dict
            Dictionary with key 'grid' containing a SimulationGrid instance
            with depth and time arrays.
        """
        grid_depth = np.linspace(
            self.spatial_resolution / 2.0,
            self.domain_length - self.spatial_resolution / 2.0,
            int(self.domain_length / self.spatial_resolution),
        )
        grid_time = np.linspace(
            0, self.time_total, int(self.time_total / self.time_resolution + 0.5)
        )
        return {"grid": SimulationGrid(grid_depth, grid_time)}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["grid"]
