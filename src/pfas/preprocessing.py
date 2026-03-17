"""
PFAS Preproccesing Module.

==========================================

This module contains preprocessor classes for preparing input parameters
for the PFAS analytical solver.

The preprocessors handle calculations for:
- Water flow properties
- Boundary conditions
- Spatial and temporal grids
- Adsorption parameters (solid-phase and air-water interface)
- Air-water interface adsorption
- Simulation execution

Classes
-------
WaterPreprocessor
    Computes hydraulic properties from infiltration and soil parameters.
BoundaryPreprocessor
    Calculates boundary conditions for contaminant input.
GridGenerator
    Generates spatial and temporal discretization grids.
SWCAdsorptionPreprocessor
    Calculates air-water interface area using thermodynamic relations.
AdsorptionCollector
    Consolidates solid-phase and air-water interface sorption parameters
    into a single Adsorption object.
SimulationRunner
    Executes the analytical solution with preprocessed parameters.

"""

from typing import Annotated, Optional

import numpy as np
from annotated_types import Ge, Gt, Interval
from pydantic import BaseModel, field_validator
from scipy.optimize import fsolve

from pfas.analytical_soln import (
    Adsorption,
    BoundaryConditions,
    HydrologicalProperties,
    SimulationGrid,
    analytical_soln,
)
from pfas.utils import aaw_func_thermo, aaw_func_tracer


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
    init_sat : float
        Initial saturation estimate (dimensionless). Range: [0, 1].
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
    ...     init_sat=0.5,
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
    init_sat: Annotated[float, Interval(ge=0, le=1)]
    residual_water_content: Annotated[float, Interval(ge=0, le=1)]

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
            return se**0.5 * (1 - (1 - se**(1/m))**m)**2 - kr

        se = fsolve(relperm, self.init_sat)
        theta = se[0] * (self.porosity - self.residual_water_content) + self.residual_water_content
        v = self.average_infiltration_rate / theta
        d = v * self.dispersivity
        return {"hydro_properties": HydrologicalProperties(theta, v, d)}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["hydro_properties"]


class BoundaryPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    """Calculate boundary conditions for contaminant input.

    Converts solute concentration and infiltration rate into a contaminant
    release rate suitable for the analytical solution. The total active pulse
    duration (sum of all interval lengths) is used to normalise the release
    rate.

    Parameters
    ----------
    average_infiltration_rate : float
        Average water infiltration rate (m/s). Must be positive.
    solute_concentration_influx : float
    Solute concentration in infiltrating water (mg/L). Must be non-negative.
    Use 0 for clean water infiltration with no PFAS input.
    pulse_intervals : list of (float, float)
        Inlet concentration on/off periods in physical time (s).
        Each tuple (t_start, t_end) defines one active pulse period.

    Examples
    --------
        - Continuous step:   ``[(0, np.inf)]``
        - Pulse from t=0:    ``[(0, 5000)]``
        - Delayed pulse:     ``[(2000, 5000)]``
        - Multiple pulses:   ``[(0, 1000), (3000, 5000)]``

    Attributes
    ----------
    outputs : list of str
        List containing 'boundary_conditions'.
    """

    average_infiltration_rate: Annotated[float, Gt(0)]
    solute_concentration_influx: Annotated[float, Ge(0)]
    pulse_intervals: list[tuple[float, float]]

    @field_validator("pulse_intervals")
    @classmethod
    def validate_pulse_intervals(
        cls, intervals: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Validate that all intervals are non-empty and non-overlapping."""
        if not intervals:
            raise ValueError("pulse_intervals must contain at least one interval.")
        for t_start, t_end in intervals:
            if t_start < 0:
                raise ValueError(
                    f"Pulse interval ({t_start}, {t_end}): t_start must be >= 0."
                )
            if t_start >= t_end:
                raise ValueError(
                    f"Pulse interval ({t_start}, {t_end}): "
                    "t_start must be strictly less than t_end."
                )
        return intervals

    def compute(self) -> dict:
        """Calculate boundary conditions.

        The contaminant release rate is normalised by the total active pulse
        duration (sum of all finite interval lengths). Infinite intervals
        (step inputs) are excluded from this sum since the rate is then
        defined per unit time directly.

        Returns
        -------
        dict
            Dictionary with key 'boundary_conditions' containing a
            :class:`BoundaryConditions` instance.
        """
        total_duration = sum(
            t_end - t_start
            for t_start, t_end in self.pulse_intervals
            if t_end != np.inf
        )

        # total_duration == 0 means no finite pulse intervals (e.g. only a
        # step input [(0, inf)], or an empty pulse list). In that case the
        # release rate is not normalised by duration but taken directly as
        # concentration * infiltration rate (continuous flux).
        if total_duration == 0:
            contaminant_release_rate = (
                self.solute_concentration_influx
                * self.average_infiltration_rate
            )
        else:
            contaminant_release_rate = (
                self.solute_concentration_influx
                * self.average_infiltration_rate
                / total_duration
            )

        bc = BoundaryConditions(
            pulse_intervals=self.pulse_intervals,
            contaminant_release_rate=contaminant_release_rate,
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


class SWCAdsorptionPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
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
        Calculate air-water interfacial area.

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

        if self.AWI["AWI_type"] == "SWC-based":
            aaw = aaw_func_thermo(
                self.sigma0, poro, alpha, n_vg, theta, thetar, thetas, self.scaling_factor_awi
            )
        elif self.AWI["AWI_type"] == "Guo":
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


class SpRetardationPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Compute solid-phase retardation factor.

    Calculates the retardation factor for sorption to solid phase using
    linear isotherm partitioning coefficient.

    Parameters
    ----------
    sorption_solid : dict
        Dictionary containing sorption parameters. Must include 'sorption_isotherm'
        and nested 'linear' dict with 'Kd_method' and 'Kd' values.
    bulk_density : float
        Soil bulk density (kg/m³). Must be positive.
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.

    Attributes
    ----------
    outputs : list of str
        List containing 'sp_retardation'.
    """

    sorption_solid: dict
    bulk_density: Annotated[float, Gt(0)]
    hydro_properties: HydrologicalProperties

    def compute(self):
        """
        Calculate solid-phase retardation.

        Returns
        -------
        dict
            Dictionary with keys 'sp_retardation' and 'Kd'.
        """
        linear = self.sorption_solid.get("linear")
        if not linear:
            raise ValueError("Missing 'linear' sorption parameters")

        if linear.get("Kd_method") != "direct_input":
            raise ValueError("Only 'direct_input' Kd_method is supported")

        if "Kd" not in linear:
            raise ValueError("Missing 'Kd' value for linear sorption")

        if self.sorption_solid["sorption_isotherm"] == "linear":
            linear = self.sorption_solid["linear"]
            if linear["Kd_method"] == "direct_input":
                Kd = linear["Kd"]  # noqa: N806
        sp_retardation = (self.bulk_density * Kd) / self.hydro_properties.water_content
        return {"sp_retardation": sp_retardation, "Kd": Kd}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["sp_retardation", "Kd"]


class SorptionKawiDirectInput(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Compute adsorption parameters for air-water interface.

    Calculates retardation factor for sorption at the air-water interface
    using directly specified partition coefficient.

    Parameters
    ----------
    kaw : float
        Air-water interface partition coefficient (dimensionless).
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    Kd : float
        Solid-phase partition coefficient (m³/kg).
    aaw : float
        Air-water interfacial area (m²/m³).

    Attributes
    ----------
    outputs : list of str
        List containing 'awi_retardation'.
    """

    kaw: float
    hydro_properties: HydrologicalProperties
    aaw: float

    def compute(self):
        """
        Calculate air-water interface retardation factor.

        Returns
        -------
        dict
            Dictionary with key 'awi_retardation'.
        """
        awi_retardation = (self.kaw * self.aaw) / self.hydro_properties.water_content
        return {"awi_retardation": awi_retardation}

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["awi_retardation"]


class AdsorptionCollector(BaseModel, validate_assignment=True, extra='forbid'):
    """
    Assemble the final Adsorption object from preprocessor outputs.

    Collects the results of SpRetardationPreprocessor and
    SorptionKawiDirectInput (and any future adsorption preprocessors)
    and assembles them into the Adsorption instance required by
    SimulationRunner.

    Parameters
    ----------
    Kd : float
        Solid-phase partition coefficient (m³/kg) from SpRetardationPreprocessor.
    sp_retardation : float
        Solid-phase retardation factor from SpRetardationPreprocessor.
    awi_retardation : float
        Air-water interface retardation factor from SorptionKawiDirectInput.
    sorption_solid : dict
        Dictionary containing solid-phase sorption parameters. Used to extract
        'rate_const' and 'fraction_instantaneous' for the Adsorption object.

    Attributes
    ----------
    outputs : list of str
        List containing 'adsorption'.

    Examples
    --------
    >>> collector = AdsorptionCollector(
    ...     Kd=0.001,
    ...     sp_retardation=3.75,
    ...     awi_retardation=0.5,
    ...     sorption_solid={"rate_const": 0.0, "fraction_instantaneous": 1.0},
    ... )
    >>> result = collector.compute()
    >>> 'adsorption' in result
    True
    """

    Kd: float  # noqa: N815
    sp_retardation: float
    awi_retardation: float
    sorption_solid: dict

    def compute(self):
        """
        Assemble the Adsorption object.

        Returns
        -------
        dict
            Dictionary with key 'adsorption' containing an Adsorption instance.
        """
        return {
            "adsorption": Adsorption(
                Kd=self.Kd,
                rate_const=self.sorption_solid.get("rate_const", 0.0),
                frac_int=self.sorption_solid.get("fraction_instantaneous", 1.0),
                sp_retardation=self.sp_retardation,
                awi_retardation=self.awi_retardation,
            )
        }

    @property
    def outputs(self):
        """List of output keys from compute() method."""
        return ["adsorption"]


class SimulationRunner(BaseModel, validate_assignment=True, extra='forbid',
                       arbitrary_types_allowed=True):
    """
    Execute the analytical solution with preprocessed parameters.

    This class runs the PFAS-LEACH-Screening analytical solver using
    all preprocessed input parameters to compute contaminant concentrations
    in water and solid phases over space and time.

    AdsorptionCollector is run internally -- the user only needs to supply
    the outputs of the individual adsorption preprocessors (sp_retardation,
    Kd, awi_retardation) and sorption_solid directly.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization from GridGenerator.
    bulk_density : float
        Soil bulk density (kg/m3). Must be positive.
    boundary_conditions : BoundaryConditions
        Boundary conditions from BoundaryPreprocessor.
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.
    sp_retardation : float
        Solid-phase retardation factor from SpRetardationPreprocessor.
    Kd : float
        Solid-phase partition coefficient (m3/kg) from SpRetardationPreprocessor.
    awi_retardation : float
        Air-water interface retardation factor from SorptionKawiDirectInput.
    sorption_solid : dict
        Solid-phase sorption parameters, used to extract 'rate_const' and
        'fraction_instantaneous'.
    kinetic_sorption : bool
        Whether to use kinetic (True) or equilibrium (False) sorption.
    volume_averaged : bool
        Whether to compute volume-averaged concentrations.
    """

    grid: SimulationGrid
    bulk_density: Annotated[float, Gt(0)]
    boundary_conditions: BoundaryConditions
    hydro_properties: HydrologicalProperties
    sorption_solid: dict
    awi_retardation: float
    kinetic_sorption: bool
    volume_averaged: bool
    initial_contaminant_concentration: Optional[np.ndarray] = None

    def _collect_adsorption(self) -> Adsorption:
        """Assemble the Adsorption object via SpRetardationPreprocessor and AdsorptionCollector."""
        sp_results = SpRetardationPreprocessor(
            sorption_solid=self.sorption_solid,
            bulk_density=self.bulk_density,
            hydro_properties=self.hydro_properties,
        ).compute()

        result = AdsorptionCollector(
            Kd=sp_results["Kd"],
            sp_retardation=sp_results["sp_retardation"],
            awi_retardation=self.awi_retardation,
            sorption_solid=self.sorption_solid,
        ).compute()
        return result["adsorption"]

    def compute(self):
        """
        Run the analytical solution.

        Assembles adsorption parameters internally before solving the
        advection-dispersion equation with retardation for PFAS transport
        through the vadose zone.

        Returns
        -------
        dict
            Dictionary with keys:

            - 'C1' : ndarray
                Aqueous phase concentration (mg/L)
            - 'C2' : ndarray
                Sorbed phase concentration (mg/kg)
            - 'C_tot' : ndarray
                Total concentration (mg/L bulk volume)
        """
        adsorption = self._collect_adsorption()
        initial_contaminant_concentration = (       
            self.initial_contaminant_concentration
            if self.initial_contaminant_concentration is not None
            else np.zeros(len(self.grid.depth))
        )
        C1, C2, C_tot = analytical_soln(  # noqa: N806
            grid=self.grid,
            bulk_density=self.bulk_density,
            boundary_conditions=self.boundary_conditions,
            initial_contaminant_concentration=initial_contaminant_concentration,
            hydro_properties=self.hydro_properties,
            adsorption=adsorption,
            kinetic=self.kinetic_sorption,
            volume_averaged=self.volume_averaged,
        )

        return {
            "C1": C1,
            "C2": C2,
            "C_tot": C_tot,
        }
