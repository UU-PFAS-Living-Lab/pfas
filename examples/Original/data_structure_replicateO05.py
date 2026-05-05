import marimo

__generated_with = "0.23.1"
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
    soils = load_dataset("soils_Ksat")
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
    soil_name = "Staring-O15"

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
    return (
        bulk_dens,
        dispersivity,
        f_oc,
        f_silt_clay,
        frac_int,
        n_CFx,
        pfas,
        porosity,
        rate_const,
        soil,
        theta_r,
        tracer_fit,
        vg_alpha,
        vg_l,
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
    GridGenerator,
    SWCAdsorptionPreprocessor,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    f_oc,
    f_silt_clay,
    frac_int,
    n_CFx,
    pfas,
    porosity,
    rate_const,
    soil,
    theta_r,
    tracer_fit,
    vg_alpha,
    vg_l,
    vg_n,
):
    ## Running simulation 
    from pfas.utils import kd_fabregat_palau
    # Step 1: Generate the grid
    grid_gen = GridGenerator(
        domain_length=100,                              # cm
        spatial_resolution=1.0,
        time_resolution=0.5*(60*60*24*365),                 # 1 year (in seconds for calculations)
        time_total=250*(60*60*24*365),                  # 250 years (in seconds for calculations)
    )
    grid_results = grid_gen.compute()

    # Step 2: Compute water flow / hydraulic properties
    water_prep = WaterPreprocessor(
        average_infiltration_rate=9.51E-7,              # cm/s (is 300 mm/year)
        hydraulic_conductivity=soil["K_sat"]["value"],  # pulled from soil data (cm/s)
        porosity=porosity,
        dispersivity=dispersivity,
        van_genuchten_n=vg_n,
        van_genuchten_l=vg_l,
        init_sat=0.2,
        residual_water_content=theta_r,
    )
    water_results = water_prep.compute()


    pulse_duration = 25 * (60 * 60 * 24 * 365)
    # Step 3: Setup boundary conditions
    boundary_prep = BoundaryPreprocessor(
        C_list=[pfas['M']["value"] * 1e-9, 0.0], # mg/L for 1 pmol/L
        T_list=[0.0, pulse_duration]
    )
    boundary_results = boundary_prep.compute()

    # Step 4: Solid phase adsorption
    sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "kinetic": {
                "frac_int": frac_int,
                "rate_const": rate_const,
            },
            "linear": {
                "Kd_method": "direct_input",
                "Kd": kd_fabregat_palau(n_CFx, f_oc, f_silt_clay)
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
        scaling_factor_awi=4.15,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {
                "scaling_factor_awi": 4.15,
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
            "tracer_fit":tracer_fit
        },
    )
    awi_results = swc_adsorp.compute()

    # Step 6: Kawi sorption
    # Step 6: Compute Kawi sorption
    kawi_sorp = SorptionKawiDirectInput(
        kaw=3.89E-4,
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
        kinetic_sorption=False,
        volume_averaged=False
    )
    final_results = sim_runner.compute()
    return (
        awi_results,
        boundary_prep,
        final_results,
        grid_results,
        kawi_results,
        sp_results,
        water_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cheking results for saturation (S) and air-water interfacial area (aaw)
    """)
    return


@app.cell
def _(awi_results, kawi_results, porosity, sp_results, theta_r, water_results):
    print(awi_results)
    print(water_results)
    print(sp_results)
    print(kawi_results)

    print('pore velocity is',((water_results['hydro_properties'].pore_velocity))*(60*60*24*365*10),'(mm/year)')
    print('effective saturation is',(((water_results['hydro_properties'].water_content)-theta_r)/(porosity-theta_r)),'(-)')
    print('air-water interfacial area is',((awi_results)['aaw']),'(cm2/cm3)')
    return


@app.cell
def _(porosity, theta_r, tracer_fit, vg_alpha, vg_l, vg_n, water_results):
    print("theta_r:", theta_r)
    print("porosity:", porosity)
    print("vg_alpha:", vg_alpha)
    print("vg_n:", vg_n)
    print("vg_l:", vg_l)
    print("tracer_fit:", tracer_fit)
    print("water_content:", water_results["hydro_properties"].water_content)
    print("Se:", (water_results["hydro_properties"].water_content - theta_r) / (porosity - theta_r))
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
def _():
    #HIER MOET NOG CODE OM DIT HIERONDER TE LATEN WERKEN
    return


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
