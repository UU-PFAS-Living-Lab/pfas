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

import numpy as np
from pfas.analytical_soln import analytical_soln
from pfas.utils import Aaw_func_thermo 
from pfas.utils import Aaw_func_tracer
from pfas.analytical_soln import SimulationGrid, BoundaryConditions, HydrologicalProperties, Adsorption
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def water_preprocessing(config, grid):
    infiltration_rate = config["experimental_conditions"]["boundary"]["average_infiltration_rate"]
    Ks = config["soil"]["hydraulic_conductivity"]
    kr = infiltration_rate/Ks
    poro = config["soil"]["porosity"]
    alphaL = config["soil"]["dispersivity"]  #100*0.83*(np.log10(grid.domain_length/100))**2.414  # dispersivity [cm] #TODO multiple options
    n_vg = config["soil"]["van_genuchten_n"]                                    # VG parameter
    m = 1 - 1/n_vg                              # VG parameter
    relperm = lambda se: se**0.5 * (1 - (1 - se**(1/m))**m)**2 - kr
    init_sat = config["experimental_conditions"]["initial"]["init_sat"] #good?
    se = fsolve(relperm, init_sat)
    res_water_content = config["soil"]["residual_water_content"]
    theta = se[0] * (poro - res_water_content) + res_water_content    # water content [cm^3/cm^3]
    sw = theta/poro     
    q = config["experimental_conditions"]["boundary"]["average_infiltration_rate"]                    # water saturation [cm^3/cm^3]
    v = q/theta                                 # pore velocity [cm/s]
    D = v*alphaL                                # dispersion coefficient [cm^2/s]
    return HydrologicalProperties(theta, v, D)

def boundary_preprocessing(config):
    #c10 is mg/s/m (what if this is 0?)
    average_infiltration_rate = config["experimental_conditions"]["boundary"]["average_infiltration_rate"]
    solute_concentration = config["experimental_conditions"]["boundary"]["solute_concentration_influx"]  #is there a better way of doing this? # units?
    pulse_duration = config["experimental_conditions"]["boundary"]["pulse_duration"] 
    contaminant_release_rate_per_second = solute_concentration* average_infiltration_rate/ pulse_duration
    return BoundaryConditions(pulse_duration, contaminant_release_rate_per_second) #TODO do these need to match naming

def adsorption_preprocessing(config, hydro_properties, bulk_density): 
    #TODO 
    sp_sorption = config["sorption_solid"]
    
    if sp_sorption["sorption_isotherm"] == "linear":
        linear = sp_sorption["linear"]
        if linear["Kd_method"] == "direct_input":
            Kd = linear["Kd"]
    
    sp_retardation = (bulk_density * Kd) / hydro_properties.water_content
    rate_const = sp_sorption.get("rate_constant", 0.0)  # alphas
    frac_int = sp_sorption.get("fraction_instantaneous", 1.0)  # Fs
    
    if "AWI" in config:
        awi = config["AWI"]
        
        if awi["AWI_type"] == "SWC-based":
            swc_based = awi["SWC-based"]
            sf = swc_based["scaling_factor_AWI"]
      # Get soil parameters
            soil = config["soil"]
            sigma0 = swc_based.get("sigma0", 0.072)  # surface tension
            poro = soil["porosity"]
            alpha = soil["van_genuchten_alpha"]
            n_vg = soil["van_genuchten_n"]
            theta = hydro_properties.water_content
            thetar = soil["residual_water_content"]
            thetas = poro  
            
            # Call the thermodynamic function
            Aaw = Aaw_func_thermo(sigma0, poro, alpha, n_vg, theta, thetar, thetas, sf)
    if "sorption_AWI" in config:
        awi_sorption = config["sorption_AWI"]
    if awi_sorption["Kawi_method"] == "direct_input":
        Kaw = awi_sorption["Kaw"]
        awi_retardation = (Kaw * Aaw) / hydro_properties.water_content  # Raw
    else:
        awi_retardation = 0.0
    
    return Adsorption(
        Kd=Kd,
        rate_const=rate_const,
        frac_int=frac_int,
        sp_retardation=sp_retardation,
        awi_retardation=awi_retardation
    )
    

def preprocess_configuration(config):
    """Complete preprocessing and return all parameters"""
    # Grid setup
    domain_length = config["experimental_conditions"]["domain_length"]
    dz = config["experimental_conditions"]["spatial_resolution"]
    grid_depth = np.linspace(dz/2.0, domain_length-dz/2.0, int(domain_length/dz))
    
    dt = config["experimental_conditions"]["time_resolution"]
    time_total = config["experimental_conditions"]["time_total"]
    grid_time = np.linspace(0, time_total, int(time_total/dt+0.5))
    
    grid = SimulationGrid(grid_depth, grid_time)
    
    # Bulk density
    bulk_density = config["soil"]["bulk_density"]
    
    # Preprocess components
    boundary_conditions = boundary_preprocessing(config)
    hydro_properties = water_preprocessing(config, grid)
    adsorption = adsorption_preprocessing(config, hydro_properties, bulk_density)
    
    # Initial contaminant concentration
    initial_contaminant_concentration = config["experimental_conditions"]["initial"].get(
        "contaminant_concentration", 
        np.zeros(len(grid_depth))
    )
    
    # Flags
    kinetic = config["sorption_solid"].get("kinetic_sorption", False)
    volume_averaged = config.get("volume_averaged", False)
    
    return {
        "grid": grid,
        "bulk_density": bulk_density,
        "boundary_conditions": boundary_conditions,
        "initial_contaminant_concentration": initial_contaminant_concentration,
        "hydro_properties": hydro_properties,
        "adsorption": adsorption,
        "kinetic": kinetic,
        "volume_averaged": volume_averaged
    }


# Main execution
def run_simulation(config):
    """Run the analytical solution with preprocessed parameters"""
    # Preprocess all parameters
    params = preprocess_configuration(config)
    
    # Call analytical solution
    C1, C2, C_tot = analytical_soln(
        grid=params["grid"],
        bulk_density=params["bulk_density"],
        boundary_conditions=params["boundary_conditions"],
        initial_contaminant_concentration=params["initial_contaminant_concentration"],
        hydro_properties=params["hydro_properties"],
        adsorption=params["adsorption"],
        kinetic=params["kinetic"],
        volume_averaged=params["volume_averaged"]
    )

    return C1, C2, C_tot, params["grid"]

