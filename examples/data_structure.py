import marimo

__generated_with = "0.18.4"
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
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import marimo as mo
    return (
        BoundaryPreprocessor,
        GridGenerator,
        SWCAdsorptionPreprocessor,
        SimulationRunner,
        SorptionKawiDirectInput,
        SpRetardationPreprocessor,
        WaterPreprocessor,
        mo,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Checking available data

    This code block shows how to access the data that is available in the PFAS data structure.
    """)
    return


@app.cell
def _():
    from pfas.data_loader import load_dataset

    PFASs = load_dataset("PFASs")
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
    soil_name = "Accusand"

    pfas = PFASs[pfas_name]
    soil = soils[soil_name]

    # Inspect PFAS properties
    print(f"\nMolar mass       : {pfas['M']}")
    print(f"K_oc             : {pfas['K_oc']}")
    print(f"Diffusivity      : {pfas['diffusivity']}")

    # Inspect soil properties
    print(f"\nBulk density     : {soil['rho_b']}")
    print(f"Porosity         : {soil['porosity']}")
    print(f"K_sat            : {soil['K_sat']}")

    # Unpack van Genuchten parameters (stored as tuple of (field, value) pairs)
    vg_params = dict(soil["van_genuchten"])   # convert to dict for easy access
    print("\nVan Genuchten parameters:")
    print(vg_params)

    # Pull scalar soil values used in the simulation
    bulk_dens   = soil["rho_b"]  ["value"]         # numeric value only (g/cm³)
    porosity    = soil["porosity"]
    vg_n        = vg_params["n"]
    theta_r     = soil["theta_r"]
    vg_alpha    = vg_params["alpha"]["value"]    # numeric value (1/cm)
    dispersivity = 1.5                       # not present for Accusand, use default
    C_rep = 1 #indication of nonlinearity for freundlich sorption, can be between 0 and 1
    # Check for solid phase adsorption paraeters available:
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
    return (
        C_rep,
        bulk_dens,
        dispersivity,
        frac_int,
        freundlich_k,
        freundlich_n,
        porosity,
        rate_const,
        soil,
        theta_r,
        use_spa,
        vg_alpha,
        vg_n,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Running Simulation
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    C_rep,
    GridGenerator,
    SWCAdsorptionPreprocessor,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    frac_int,
    freundlich_k,
    freundlich_n,
    porosity,
    rate_const,
    soil,
    theta_r,
    use_spa,
    vg_alpha,
    vg_n,
):
    ## Running simulation 
    from pfas.utils import kd_freundlich
    # Step 1: Generate the grid
    grid_gen = GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=5000,
    )
    grid_results = grid_gen.compute()

    # Step 2: Compute water flow / hydraulic properties
    water_prep = WaterPreprocessor(
        average_infiltration_rate=1.5,
        hydraulic_conductivity=soil["K_sat"]["value"],   # pulled from soil data
        porosity=porosity,
        dispersivity=dispersivity,
        van_genuchten_n=vg_n,
        init_sat=0.2,
        residual_water_content=theta_r,
    )
    water_results = water_prep.compute()

    # Step 3: Setup boundary conditions
    boundary_prep = BoundaryPreprocessor(
        average_infiltration_rate=0.5,
        solute_concentration_influx=10.0,
        pulse_intervals=[(0, 2000)],        # pulse from t=0 to t=2000 s
    )
    boundary_results = boundary_prep.compute()
    sorption_solid = {
            "kinetic_sorption": use_spa,
            "sorption_isotherm": "linear",
            "kinetic": {
                "frac_int": frac_int,
                "rate_const": rate_const,
            },
            "linear": {
                "Kd_method": "direct_input",
                "Kd": kd_freundlich(C_rep, freundlich_k, freundlich_n),
            },
    }


    sp_retard = SpRetardationPreprocessor(
        sorption_solid= sorption_solid,
        bulk_density=bulk_dens,
        hydro_properties=water_results["hydro_properties"],
    )
    sp_results = sp_retard.compute()

    # Step 5: Air-water interface (AWI) adsorption
    swc_adsorp = SWCAdsorptionPreprocessor(
        hydro_properties=water_results["hydro_properties"],
        sigma0=71,
        scaling_factor_awi=1.0,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {
                "scaling_factor_awi": 1.0,
            },
        },
        soil={
            "bulk_density": bulk_dens,
            "porosity": water_prep.porosity,
            "van_genuchten_alpha": vg_alpha,
            "van_genuchten_n": water_prep.van_genuchten_n,
            "saturated_water_content": porosity,
            "residual_water_content": water_prep.residual_water_content,
            "hydraulic_conductivity": water_prep.hydraulic_conductivity,
            "dispersivity": water_prep.dispersivity,
        },
    )
    awi_results = swc_adsorp.compute()

    # Step 6: Kawi sorption
    # Step 6: Compute Kawi sorption
    kawi_sorp = SorptionKawiDirectInput(
        kaw=0.5,
        hydro_properties=water_results["hydro_properties"],
        aaw=awi_results["aaw"],
    )
    kawi_results = kawi_sorp.compute()

    # Step 7: Run the simulation
    sim_runner = SimulationRunner(
        grid=grid_results["grid"],
        bulk_density=bulk_dens,
        boundary_conditions=boundary_results["boundary_conditions"],
        hydro_properties=water_results["hydro_properties"],
        awi_retardation=kawi_results["awi_retardation"],
        sorption_solid=sorption_solid,
        kinetic_sorption=True,
        volume_averaged=True
    )
    final_results = sim_runner.compute()
    return final_results, grid_results


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
        plt.plot(final_results['C_tot'][:, t_idx], simulation_grid.depth, label=f"t = {simulation_grid.time[t_idx]:.0f} s")

    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
