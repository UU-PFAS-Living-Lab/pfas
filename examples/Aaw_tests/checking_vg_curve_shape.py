import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Utilizing data structure
    This tutorial demonstrates how to use the data structure provided by the pfas package, which includes experimental data from peer-reviewed studies and soil property information for various soil types.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner, SorptionKawCalculated
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import marimo as mo

    return SWCAdsorptionPreprocessor, mo, plt


@app.cell
def _():
    from pfas.data_loader import load_dataset

    PFASs = load_dataset("PFASs_neutral_forms")
    soils = load_dataset("soils")
    spa_matrix = load_dataset("spa_matrix")
    # See what's available
    print("Available PFAS compounds:")
    print(list(PFASs.keys()))

    print("\nAvailable soils:")
    print(list(soils.keys()))

    print("\nSoils with sorption parameter data (spa_matrix):")
    print(list(spa_matrix.keys()))
    return PFASs, soils, spa_matrix


@app.cell
def _(PFASs, soils, spa_matrix):
    # Pick a compound and soil for this run
    pfas_name = "PFOA"
    soil_name = "Staring-O05"

    pfas = PFASs[pfas_name]
    soil = soils[soil_name]

    # Inspect PFAS properties
    print(f"\nMolar mass     : {pfas['M']}")
    print(f"K_oc             : {pfas['K_oc']}")
    print(f"Diffusivity      : {pfas['diffusivity']}")

    # Inspect soil properties
    print(f"\nBulk density   : {soil['rho_b']}")
    print(f"K_sat            : {soil['K_sat']}")
    print(f"Porosity         : {soil['porosity']}")
    print(f"Guo_params       : {soil['tracer_fit']}")

    # Unpack van Genuchten parameters (stored as tuple of (field, value) pairs)
    vg_params = dict(soil["van_genuchten"])   # convert to dict for easy access
    print("\nVan Genuchten parameters:")
    print(vg_params)

    # Pull scalar soil values used in the simulation
    bulk_dens    = soil["rho_b"] ["value"]         # numeric value only (g/cm³)
    vg_n         = vg_params["n"]
    vg_l         = vg_params["l"]
    theta_r      = soil["theta_r"]
    vg_alpha     = vg_params["alpha"]["value"]    # numeric value (1/cm)
    dispersivity = 4.5                            # cm (taken from thesis Hugo vd Berg)
    porosity     = soil["porosity"]
    tracer_fit   = soil["tracer_fit"]
    C_rep = 1 #indication of nonlinearity for freundlich sorption, can be between 0 and 1
    # Check for solid phase adsorption parameters available:
    if soil_name in spa_matrix and pfas_name in spa_matrix[soil_name]:
        spa = dict(spa_matrix[soil_name][pfas_name])
        freundlich_k = spa["Freundlich_K"]["value"]   # numeric value
        freundlich_n = spa["Freundlich_N"]
        frac_int = spa["frac_instant_adsorption"]
        rate_const = spa["kinetic_adsorption_rate"]
        print(f"\nSorption parameters (spa_matrix) for {pfas_name} in {soil_name}:")
        print(f"  Freundlich K : {freundlich_k}")
        print(f"  Freundlich N : {freundlich_n}")
        print(f"  Frac instant : {frac_int}")
        print(f"  Kinetic rate : {rate_const} 1/h")
        use_spa = True
    else:
        print(f"\nNo spa_matrix entry for {pfas_name} in {soil_name}. Using fallback Kd.")
        use_spa = False

    frac_int = 1.0
    rate_const = 0.0

    #Fabregat-Palau prep
    f_silt_clay = (soil["f_clay"]["value"]/100) + (soil["f_silt"]["value"]/100)
    f_c  = soil["f_clay"]["value"]/100
    f_s  = soil["f_silt"]["value"]/100
    f_oc = soil["f_oc"]["value"]/100
    n_CFx = pfas["n_CFx"]
    print(f_c, f_s, f_oc, f_silt_clay, n_CFx)
    print('theta_r is', theta_r)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Running Simulation
    """)
    return


@app.cell
def _(SWCAdsorptionPreprocessor, plt, soils):
    import numpy as np

    def compute_aaw_for_soil(soil_name, Se_values):
        soil = soils[soil_name]
        vg_params = dict(soil["van_genuchten"])

        bulk_dens = soil["rho_b"]["value"]
        porosity = soil["porosity"]
        theta_r = soil["theta_r"]
        vg_alpha = vg_params["alpha"]["value"]
        vg_n = vg_params["n"]

        aaw_vals = []

        for Se in Se_values:
            theta = theta_r + Se * (porosity - theta_r)

            hydro = {
                "water_content": theta,
                "pore_velocity": 1e-5,
                "dispersion_coefficient": 1e-5,
            }

            swc_adsorp = SWCAdsorptionPreprocessor(
                hydro_properties=hydro,
                sigma0=71,
                scaling_factor_awi=4.15,
                AWI={
                    "AWI_type": "SWC-based",
                    "SWC-based": {
                        "scaling_factor_awi": 4.15,
                    },
                },
                soil={
                    "bulk_density": bulk_dens,
                    "porosity": porosity,
                    "van_genuchten_alpha": vg_alpha,
                    "van_genuchten_n": vg_n,
                    "saturated_water_content": porosity,
                    "residual_water_content": theta_r,
                    "hydraulic_conductivity": soil["K_sat"]["value"],
                    "dispersivity": 4.5,
                    "tracer_fit": soil["tracer_fit"],
                },
            )

            aaw = swc_adsorp.compute()["aaw"]
            aaw_vals.append(aaw)

        return np.array(aaw_vals)

    Se_values = np.linspace(0.05, 0.95, 50)

    aaw_o01 = compute_aaw_for_soil("Staring-O01", Se_values)
    aaw_o02 = compute_aaw_for_soil("Staring-O02", Se_values)
    aaw_o03 = compute_aaw_for_soil("Staring-O03", Se_values)
    aaw_o04 = compute_aaw_for_soil("Staring-O04", Se_values)
    aaw_o05 = compute_aaw_for_soil("Staring-O05", Se_values)
    aaw_o12 = compute_aaw_for_soil("Staring-O12", Se_values)
    aaw_o15 = compute_aaw_for_soil("Staring-O15", Se_values)
    aaw_o18 = compute_aaw_for_soil("Staring-O18", Se_values)

    plt.plot(Se_values, aaw_o01, label="Staring-O01")
    plt.plot(Se_values, aaw_o02, label="Staring-O02")
    plt.plot(Se_values, aaw_o03, label="Staring-O03")
    plt.plot(Se_values, aaw_o04, label="Staring-O04")
    plt.plot(Se_values, aaw_o05, label="Staring-O05")
    plt.plot(Se_values, aaw_o12, label="Staring-O12")
    plt.plot(Se_values, aaw_o15, label="Staring-O15")
    plt.plot(Se_values, aaw_o18, label="Staring-O18")
    plt.yscale("log")
    plt.xlabel("Effective saturation (-)")
    plt.ylabel("Aaw (cm²/cm³)")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting Results
    """)
    return


@app.cell
def _(final_results, grid_results, plt):
    simulation_grid = grid_results["grid"]
    # Select specific time indices to plot
    t_len = final_results['C_tot'].shape[1]
    time_indices = [0, t_len//4, t_len//2, 3*t_len//4, -1]  # First, and some intermediate, and last time step

    plt.figure(figsize=(8, 6))

    for t_idx in time_indices:
        plt.plot(final_results['C_tot'][:, t_idx], simulation_grid.depth, label=f"t = {simulation_grid.time[t_idx]:.0f} years")

    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (simulation_grid,)


@app.cell
def _():
    #HIER MOET NOG CODE OM DIT HIERONDER TE LATEN WERKEN
    return


@app.cell
def _(final_results, plt, simulation_grid):
    #Breakthrough plot of total concentration at bottom of grid over time
    C_tot = final_results['C_tot']

    seconds_per_year = 60*60*24*365
    bottom_idx = -1
    bottom_depth = simulation_grid.depth[bottom_idx]

    plt.figure(figsize=(8, 6))

    plt.plot(
        simulation_grid.time/seconds_per_year,
        C_tot[bottom_idx, :],
        label=f"Depth = {bottom_depth} cm",
        color="blue"
    )

    plt.xlabel("Time (years)") 
    plt.ylabel("Total PFAS Concentration (mg/L)")
    plt.title("PFAS Concentration Over Time")
    return C_tot, bottom_depth, bottom_idx, seconds_per_year


@app.cell
def _(
    C_tot,
    bottom_depth,
    bottom_idx,
    boundary_prep,
    plt,
    seconds_per_year,
    simulation_grid,
):
    #Breakthrough plot of relative concentration at bottom of grid over time
    # Relative concentration calculation!
    C_0 = boundary_prep.solute_concentration_influx
    C_rel = C_tot[bottom_idx, :]/C_0

    plt.figure(figsize=(8, 6))

    plt.plot(
        simulation_grid.time/seconds_per_year, 
        C_rel, 
        label=f"Depth = {bottom_depth} cm", 
        color="blue"
    )
    plt.xlabel("Time (years)") 
    plt.ylabel("Relative PFAS Concentration (-)")
    plt.title("PFAS Concentration Over Time")
    return


if __name__ == "__main__":
    app.run()
