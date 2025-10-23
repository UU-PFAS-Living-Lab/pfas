import numpy as np
from scipy.special import iv
from scipy.special import erfc
from pfas_leach_screening import utils
from pfas_leach_screening.solvers import equilibrium_solver, kinetic_solver
from dataclasses import dataclass

@dataclass
class SimulationGrid():
    depth: np.array[float]
    time: np.array[float]

# @dataclass
# class SoilParameters():
    # bulk_density: float  # rhob

@dataclass
class BoundaryConditions():
    pulse_time: float
    # solute_concentration: float
    contaminant_release_rate: float # C10

# class InitialConditions():
    # contaminant_concentration: np.array[float] # Ci

@dataclass
class HydrologicalProperties():
    water_content: float # theta
    pore_velocity: float
    dispersion_coefficient: float  # D

@dataclass
class Adsorption():
    Kd: float
    rate_const: float  # alphas
    frac_int: float # Fs
    total_retardation: float # R
    sp_retardation: float # Rs
    awi_retardation: float # Raw


def analytical_soln(
    grid: SimulationGrid,
    bulk_density: float,
    boundary_conditions: BoundaryConditions,
    initial_contaminant_concentration: float,
    hydro_properties: HydrologicalProperties,
    adsorption: Adsorption,
    kinetic: bool = False,
    volume_averaged: bool = False):
    # spaflag,cflag):
    # Compute the aqueous concentration (C1), solid-phase adsorption in the kinetic 
    # sorption domain (C2), and the total concentration (C_tot)
    
    # Processes & conditions included: advection, dispersion, kinetic SPA,
    # equilibrium AWIA, and nonzero spatially variable initial condition
    
    # output parameters
    # C1                    aqueous concentration in space (micro mol/cm^3)
    #                       NB: C1 here is the C in Guo et al (2022) AWR.         
    # C2                    kinetic solid-phase adsorption (micro mol/g)
    #                       NB: C2 here is the Cs,2 in Guo et al (2022) AWR.  
    # C_tot                 total concentration per bulk volume in space (micro mol/cm^3) 
    
    # input parameters
    # t         time vector (second)
    #           NB: t can be a vector representing different points in time or
    #               a single value representing one point in time
    # z         length vector (cm)
    #           NB: z can be a vector representing different points in space or
    #               a single value representing one point in space (e.g.,outlet)
    # t0        duration of active contamination (second)
    #           NB: set t0 to zero if the simulation does not include an active
    #               contamination period
    # pfas_tot  total (single-component) PFAS mass that will be released (mg)
    #           NB: set pfas_tot to zero if the simulation does not include an
    #               active contamination period
    # Ci        initial aqueous concentration in space (micro mol/cm^3)
    # L         length of the domain (cm)
    # v         interstitial pore velocity (cm/s)
    #           NB: v = q/theta
    # theta     water content (-)
    # rhob      bulk density (g/cm^3)
    # D         dispersion coefficient (cm^2/s)
    # Kd        solid-phase adsorption coefficient (cm^3/g)
    # alphas    first-order rate constant for kinetic SPA (1/s)
    # Fs        fraction of sorbent for which sorption is instantaneous (-)
    # R         total retardation factor (-)
    # Rs        retardation factor associated with SPA
    # Raw       retardation factor associated with AWIA
    # spaflag   A flag to denote equilibrium or kinetic SPA
    #           NB: 0 - SPA is equilibrium, 1 - SPA is kinetic
    # cflag     A flag to denote volume-averaged or flux-averaged concentration
    #           NB: 0 - concentration is volume-averaged, 1 - concentration is flux-averaged
    #           

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
    
    # Compute releasing contaminant concentration (micro mol/cm^3)
    if t0 > 0:
        # if an active contamination period is considered
        C10 = pfas_tot / (t0*v*theta)
    else:
        # if the simulation does not include an active contamination period
        C10 = 0
    
    #Compute dimensionless variables
    Z = z/L            # dimensionless length
    T = t*(v/L)        # dimensionless time
    T0 = t0*(v/L)      # dimensionless duration of contamination period
    P = v*L/D          # Peclect number
    
    # Define dimensionless variables for kinetic SPA
    betas = (1+Fs*Rs)/(1+Rs)
    ws = alphas*(1-betas)*(1+Rs)*L/v   # Damköhler number
    beta = (betas*(1+Rs)+Raw) / (1+Rs+Raw)
    
    C2 = None #TODO
    # SPA is equilibrium
    if spaflag == 0:
        C1, C_tot = equilibrium_solver(R, Z, T, P, T0, C10, Ci, theta)
        
    # SPA is kinetic
    elif spaflag == 1:
        C1, C2, C_tot = kinetic_solver(R, Z, T, P, T0, C10, Ci, ws, betas, beta, cflag, Rs, Fs, Kd, theta, rhob)

    return C1, C2, C_tot
