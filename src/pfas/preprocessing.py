# Copyright 2021-2022 Bo Guo (University of Arizona, Email: boguo@arizona.edu or guobo07@gmail.com).

# This file is part of the implementation for the PFAS-LEACH-Screening analytical 
# solver presented in the article Guo et al. (2022) AWR

# Guo, B., Zeng, J., Brusseau, M.L. and Zhang, Y., 2022. 
# A screening model for quantifying PFAS leaching in the vadose zone and 
# mass discharge to groundwater. Advances in Water Resources, 160, p.104102.

# Development of PFAS-LEACH-Screening is supported by the ESTCP Project ER21-5041. 
# Project Webpage: https://www.serdp-estcp.org/Program-Areas/Environmental-Restoration/ER21-5041

# The PFAS-LEACH-Screening analytical solver is distributed in the hope that 
# it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details, <http://www.gnu.org/licenses/>.

from typing import Annotated
from scipy.optimize import fsolve
from annotated_types import Gt, Interval
from pydantic import BaseModel, ConfigDict
from pfas.analytical_soln import SimulationGrid, BoundaryConditions, HydrologicalProperties, Adsorption
import numpy as np
from pfas.utils import Aaw_func_thermo 
from pfas.analytical_soln import analytical_soln


class WaterPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    average_infiltration_rate: Annotated[float, Gt(0)]
    hydraulic_conductivity: Annotated[float, Gt(0)]
    porosity: Annotated[float, Interval(ge=0, le=1)]
    dispersivity: Annotated[float, Gt(0)]
    van_genuchten_n: Annotated[float, Gt(0)]
    init_sat: Annotated[float, Interval(ge=0, le=1)]
    residual_water_content: Annotated[float, Interval(ge=0, le=1)]

    def compute(self):
        kr = self.average_infiltration_rate/self.hydraulic_conductivity
        m = 1 - 1/self.van_genuchten_n
        relperm = lambda se: se**0.5 * (1 - (1 - se**(1/m))**m)**2 - kr
        se = fsolve(relperm, self.init_sat)
        theta = se[0] * (self.porosity - self.residual_water_content) + self.residual_water_content
        # sw = theta/porosity
        v = self.average_infiltration_rate/theta
        D = v*self.dispersivity
        return {"hydro_properties": HydrologicalProperties(theta, v, D)}

    @property
    def outputs(self):
        return ["hydro_properties"]

class BoundaryPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    average_infiltration_rate: Annotated[float, Gt(0)]
    solute_concentration_influx: Annotated[float, Gt(0)]
    pulse_duration: Annotated[float, Gt(0)]

    def compute(self):
        #c10 is mg/s/m (what if this is 0?)
        # average_infiltration_rate = config["experimental_conditions"]["boundary"]["average_infiltration_rate"]
        # solute_concentration = config["experimental_conditions"]["boundary"]["solute_concentration_influx"]  #is there a better way of doing this? # units?
        # pulse_duration = config["experimental_conditions"]["boundary"]["pulse_duration"] 
        contaminant_release_rate_per_second = (
            self.solute_concentration_influx * self.average_infiltration_rate / self.pulse_duration)
        bc = BoundaryConditions(self.pulse_duration, contaminant_release_rate_per_second)
        return {"boundary_conditions": bc} #TODO do these need to match naming

    @property
    def outputs(self):
        return ["boundary_conditions"]

class GridGenerator(BaseModel, validate_assignment=True, extra='forbid'):
    domain_length: Annotated[float, Gt(0)]
    spatial_resolution: Annotated[float, Gt(0)]
    time_resolution: Annotated[float, Gt(0)]
    time_total: Annotated[float, Gt(0)]

    def compute(self):
        grid_depth = np.linspace(self.spatial_resolution/2.0,
                                 self.domain_length-self.spatial_resolution/2.0,
                                 int(self.domain_length/self.spatial_resolution))
        grid_time = np.linspace(0, self.time_total, int(self.time_total/self.time_resolution+0.5))
        return {"grid": SimulationGrid(grid_depth, grid_time)}

    @property
    def outputs(self):
        return ["grid"]

class SpRetardationPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    sorption_solid: dict
    bulk_density: Annotated[float, Gt(0)]
    hydro_properties: HydrologicalProperties
    
    def compute(self):
        if self.sorption_solid["sorption_isotherm"] == "linear":
            linear = self.sorption_solid["linear"]
            if linear["Kd_method"] == "direct_input":
                Kd = linear["Kd"]
        sp_retardation = (self.bulk_density * Kd) / self.hydro_properties.water_content
        return {"sp_retardation": sp_retardation, "Kd": Kd}

    @property
    def outputs(self):
        return ["sp_retardation"]

class SWCAdsorptionPreprocessor(BaseModel, validate_assignment=True, extra='forbid'):
    hydro_properties: HydrologicalProperties
    sigma0: Annotated[float, Gt(0)] = 0.072
    scaling_factor_awi: float
    soil: dict

    def compute(self):
        # sf = self.swc_based["scaling_factor_AWI"]
    # Get soil parameters
        poro = self.soil["porosity"]
        alpha = self.soil["van_genuchten_alpha"]
        n_vg = self.soil["van_genuchten_n"]
        theta = self.hydro_properties.water_content
        thetar = self.soil["residual_water_content"]
        thetas = poro
        
        # Call the thermodynamic function
        Aaw = Aaw_func_thermo(self.sigma0, poro, alpha, n_vg, theta, thetar, thetas,
                              self.scaling_factor_awi)
        return {"aaw": Aaw}

class SorptionKawiDirectInput(BaseModel, validate_assignment=True, extra='forbid'):
    kaw: float
    hydro_properties: HydrologicalProperties
    Kd: float
    aaw: float
    sorption_solid: dict
    sp_retardation: float
    # sorption_awi: ffloat

    def compute(self):
        # 
            # awi_sorption = config["sorption_AWI"]
        # if awi_sorption["Kawi_method"] == "direct_input":
            # Kaw = awi_sorption["Kaw"]
        awi_retardation = (self.kaw * self.aaw) / self.hydro_properties.water_content  # Raw
        return {"adsorption": Adsorption(
            Kd=self.Kd,
            rate_const=self.sorption_solid.get("rate_const", 0.0),
            frac_int=self.sorption_solid.get("fraction_instantaneous", 1.0),
            sp_retardation=self.sp_retardation,
            awi_retardation=awi_retardation,
        )}

    @property
    def outputs(self):
        return ["awi_retardation"]
        # else:
            # awi_retardation = 0.0
            # 

class SimulationRunner(BaseModel, validate_assignment=True, extra='forbid',
                       arbitrary_types_allowed=True):
    grid: SimulationGrid
    bulk_density: Annotated[float, Gt(0)]
    boundary_conditions: BoundaryConditions
    # contaminant_concentration: Annotated[float, Gt(0)]
    hydro_properties: HydrologicalProperties
    adsorption: Adsorption
    kinetic_sorption: bool
    volume_averaged: bool

    def compute(self):
        initial_contaminant_concentration = np.zeros(len(self.grid.depth))
        C1, C2, C_tot = analytical_soln(
            grid=self.grid,
            bulk_density=self.bulk_density,
            boundary_conditions=self.boundary_conditions,
            initial_contaminant_concentration=initial_contaminant_concentration,
            hydro_properties=self.hydro_properties,
            adsorption=self.adsorption,
            kinetic=self.kinetic_sorption,
            volume_averaged=self.volume_averaged
        )

        return {
            "C1": C1,
            "C2": C2,
            "C_tot": C_tot,
        }
