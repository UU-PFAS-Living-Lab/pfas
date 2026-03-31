#%%% ----------------------------------------------------------------------------
# PFAS Python code check 
# March 2026, UU
# Alex Hockin
# ------------------------------------------------------------------------------

#%%%============================================================================
# General example
# AH tried to recreate same example as used in the RTM course 
#  
#===============================================================================

#loading relevant modules 
from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner
from pfas.configuration import read_toml
from pfas.model import Model
from matplotlib import pyplot as plt
import marimo as mo
#%%
# Shared parameters: 
bulk_dens = 1580 #kg/m3

#%%
# Step 1: Generate the grid
grid_gen = GridGenerator(
    domain_length=5, # 5 m deep in [m]
    spatial_resolution=0.025, # 200 grid cells, [m]
    time_resolution= 0.1, 
    time_total= 25, # 25 yrs 
)
grid_results = grid_gen.compute()

# Step 2: Compute water flow properties
water_prep = WaterPreprocessor(
    average_infiltration_rate = 0.869, #m/y, 
    hydraulic_conductivity = 1763, #m
    porosity=0.363, # [-]
    dispersivity=24.39/100, # [m]
    van_genuchten_n=2.72,
    init_sat=0.2, # [-]
    residual_water_content=0.054 # [-]
)
water_results = water_prep.compute()
print(water_results["hydro_properties"])

# Step 3: Setup boundary conditions
boundary_prep = BoundaryPreprocessor(
    C_list=[10.0e-6, 0],   # 10 ng/L in mg/L, then clean water
    T_list=[0, 25],        # pulse from t=0 to t=25 yrs
)
boundary_results = boundary_prep.compute()

print(boundary_results["boundary_conditions"])
# Step 4: Compute solid phase retardation
sorption_solid = {
    "kinetic_sorption": False,
    "sorption_isotherm": "linear",
    # "kinetic": {
    #     "frac_int": 0.3,
    #     "rate_const": 0.01
    # },
    "linear": {
        "Kd_method": "direct_input",
        "Kd": 5.3e-4 #m3/kg  
    },
}
sp_retard = SpRetardationPreprocessor(
    sorption_solid=sorption_solid,
    bulk_density=bulk_dens,
    hydro_properties=water_results["hydro_properties"]
)
sp_results = sp_retard.compute()

# Step 5: Compute AWI adsorption
swc_adsorp = SWCAdsorptionPreprocessor(
    hydro_properties=water_results["hydro_properties"],
    sigma0=71,
    scaling_factor_awi=1.0,
    AWI={
        "AWI_type": "SWC-based",
        "SWC-based": {
            "scaling_factor_awi": 1.0
        },
    },
    soil={
        "bulk_density": bulk_dens,
        "porosity": water_prep.porosity,
        "van_genuchten_alpha": 0.03133, #1/cm
        "van_genuchten_n": water_prep.van_genuchten_n,
        "saturated_water_content": 0.363, # [-]
        "residual_water_content": water_prep.residual_water_content,
        "hydraulic_conductivity": water_prep.hydraulic_conductivity,
        "dispersivity": water_prep.dispersivity
    }
)
awi_results = swc_adsorp.compute()

a = 1.499900e-01       # mol/m3 (PFOS)
MW = 414          # g/mol (PFOS)

C_rep_mgL = 10e-6 
C_rep = C_rep_mgL / MW  # mol/m3
kaw = 5.81e-06 
kaw_eff = kaw/(a+C_rep)
# Step 6: Compute Kawi sorption
kawi_sorp = SorptionKawiDirectInput(
    kaw = kaw_eff ,#0.5, [m3/m2]
    hydro_properties=water_results["hydro_properties"],
    aaw=  	34176.0569523106, # awi_results["aaw"], [m2/m3]
)
kawi_results = kawi_sorp.compute()
# Step 7: Run simulation
sim_runner = SimulationRunner(
    grid=grid_results["grid"],
    bulk_density=bulk_dens,
    boundary_conditions=boundary_results["boundary_conditions"],
    hydro_properties=water_results["hydro_properties"],
    awi_retardation=kawi_results["awi_retardation"],
    sorption_solid=sorption_solid,
    kinetic_sorption=False,
    volume_averaged=False
)
final_results = sim_runner.compute()
print("Simulation completed successfully!")
import numpy as np
#%%

simulation_grid = grid_results["grid"]
# Concentrations at bottom of profile over time
import pandas as pd
depth_idx = -1
t_idx_ss  = -1

C1_ss  = final_results['C1'][depth_idx, t_idx_ss]
C_tot_solver = final_results['C_tot'][depth_idx, t_idx_ss]

theta  = water_results["hydro_properties"].water_content
rho_b  = bulk_dens
Kd     = sorption_solid["linear"]["Kd"]
Kaw    = kawi_sorp.kaw
Aaw    = kawi_sorp.aaw

Cl     = C1_ss * theta
Cs     = Kd * rho_b * C1_ss
Cawi   = Kaw * Aaw * C1_ss
C_tot  = Cl + Cs + Cawi
Cl_m3w = C1_ss
Cs_kg  = Kd * C1_ss

df = pd.DataFrame({
    "Phase"  : ["C_tot (recomputed)", "C_tot (solver)", "Cl",    "Cs",    "Cawi",  "Cl",    "Cs"],
    "Units"  : ["mg/m3b",             "mg/m3b",         "mg/m3b","mg/m3b","mg/m3b","mg/m3w","mg/kg"],
    "Value"  : [C_tot,                C_tot_solver,      Cl,      Cs,      Cawi,    Cl_m3w,  Cs_kg],
})

pd.set_option("display.float_format", "{:.3e}".format)
print(f"Concentrations at depth = {simulation_grid.depth[depth_idx]:.1f} m, "
      f"t = {simulation_grid.time[t_idx_ss]:.1f} yr")
print(df.to_string(index=False))
print(f"\nMismatch (recomputed - solver): {C_tot - C_tot_solver:.3e} mg/m3b")
print(f"Relative mismatch:              {(C_tot - C_tot_solver) / C_tot_solver * 100:.2f}%")
# Select specific time indices to plot
t_len = final_results['C_tot'].shape[1]
# time_indices = [0, 4,  t_len//4, t_len//2, 3*t_len//4, -1]  # First, and some intermediate, and last time step
times_in_years = [0, 0.5, 1, 2, 3, 6, 11, 24.9]
import numpy as np
time_indices = [np.argmin(np.abs(simulation_grid.time - t)) for t in times_in_years] # 

plt.figure(figsize=(8, 6))

for t_idx in time_indices:
    plt.plot((final_results['C_tot'][:, t_idx] / 414.07), simulation_grid.depth, 
             label=f"t = {simulation_grid.time[t_idx] :.0f} yr")

# Should be ~1.0 if correct, tells you the exact scaling factor if not
#plt.xlabel("Total PFAS Concentration (mg/L)")
plt.xlabel("Total PFAS Concentration (mol/m3)")

plt.ylabel("Depth (cm)")
plt.title("PFAS Concentration Depth Profile at Different Times")
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Breakthrough curve: concentration at the bottom of the profile over time
time_years = simulation_grid.time  


depth_idx = -1  # last depth = deepest point

plt.figure(figsize=(8, 6))

plt.plot(time_years, final_results['C1'][depth_idx, :]*1e6)

plt.xlabel("Time (yr)")
plt.ylabel("PFAS Concentration (ng/L)")
plt.title(f"Breakthrough Curve at depth = {simulation_grid.depth[depth_idx]:.0f} m")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%


