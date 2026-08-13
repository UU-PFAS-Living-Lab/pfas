import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running simulations without TOML
    In this example, we will showcase how we can run a simulation without providing a TOML file.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.preprocessing import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        SpRetardationPreprocessor,
        SWCAdsorptionPreprocessor
    )
    from pfas.component import LinearSPsorption, Le2021_langmuir, Szyszkowski, SWCsorption, Retardation, EquilibriumSolver
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo
    import numpy as np

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        LinearSPsorption,
        Retardation,
        SWCsorption,
        Szyszkowski,
        WaterPreprocessor,
        load_dataset,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining shared parameters betweenthe model classes
    Some classes require the same parameters. When providing a TOML file, this is handled correctly, but when providing the parameters seperately to the different classes, we need to take care of this ourselves.

    In the next line of code, we will define bulk density. Furthermore, in our calling of the different classes, we will reuse some of the parameters that we have defined (`i.e. "residual_water_content": water_prep.residual_water_content` )
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling different classes and running the model

    Now, we will call all the classes and generate our model. If we are missing parameters, the code will tell us.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    EquilibriumSolver,
    LinearSPsorption,
    Retardation,
    SWCsorption,
    Szyszkowski,
    WaterPreprocessor,
    grid,
    grid_results,
    load_dataset,
    np,
):
    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")

    # ── PFAS: PFOA ────────────────────────────────────────────────────────────
    pfas_name = "PFOA"
    pfas      = pfas_db[pfas_name]
    n_CFx     = pfas["structural_properties"]["n_CFx"]
    K_oc      = pfas["K_oc"]["value"]   # L/kg
    K_sc      = pfas["K_sc"]["value"]   # L/kg

    soil_name = "Staring-O01"
    soil        = soil_db[soil_name]
    bulk_dens   = soil["rho_b"]["value"]                   # g/cm3
    porosity    = soil["porosity"]                         # -
    theta_r     = soil["theta_r"]                          # -
    theta_s     = soil["theta_s"]                          # -
    K_sat       = soil["K_sat"]["value"]                   # cm/s
    vg_alpha    = soil["van_genuchten"]["alpha"]["value"]  # 1/cm
    vg_n        = soil["van_genuchten"]["n"]               # -
    vg_l        = soil["van_genuchten"]["l"]               # -
    dispersivity = 4.5                                     # cm
    f_oc        = soil["f_oc"]["value"] / 100              # - (fraction)
    f_clay      = soil["f_clay"]["value"] / 100            # - (fraction)
    f_silt      = soil["f_silt"]["value"] / 100            # - (fraction)
    f_silt_clay = f_silt + f_clay                          # - (fraction)
    d50         = soil["d50"]["value"] / 10000             # µm → cm

    # Surface tension of water
    sigma0=72.8
    T = 293.15

    # Solid-phase adsorption: linear
    frac_int = 1.0
    rate_const = 0.0

    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc} L/kg  |  K_sc = {K_sc} L/kg")
    water_prep = WaterPreprocessor(
            average_infiltration_rate=9.51E-7,                 # cm/s
            hydraulic_conductivity=K_sat,
            porosity=porosity,
            dispersivity=dispersivity,
            van_genuchten_n=vg_n,
            van_genuchten_l=vg_l,
            residual_water_content=theta_r,
        )
    water_results = water_prep.compute()
    theta = water_results["hydro_properties"].water_content

    pulse_duration = 25 * (60 * 60 * 24 * 365)  # seconds
    boundary_prep = BoundaryPreprocessor(
        C_list=[pfas['M']["value"] * 1e-15, 0.0],           # g/cm3
        T_list=[0.0, pulse_duration]
    )
    boundary_results = boundary_prep.compute()

    # Step 4: Solid phase adsorption
    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 1.0, "rate_const": 0.0},
        "linear": {
            "Kd_method": "direct_input",
            # "Kd": kd_fabregat_palau(n_CFx, f_oc, f_silt_clay),
            "Kd": (K_oc*f_oc) + (K_sc*f_silt_clay),
        },
    }
    sorption = LinearSPsorption(sorption_solid=sorption_solid)

    spsorption_result = sorption.compute()



    # Aaw-preprocessor
    aaw_preprocessor = SWCsorption(
        hydro_properties=water_results["hydro_properties"],
        sigma0=sigma0,
        scaling_factor_awi=4.15,
        soil={
            "porosity": porosity,
            "van_genuchten_alpha": vg_alpha,
            "van_genuchten_n": vg_n,
            "residual_water_content": theta_r,
        },
    )
    aaw = aaw_preprocessor.compute()["aaw"]
    # Select Kaw estimation method; use Szyszkowski
    kawi_sorp = Szyszkowski(
                sigma0=sigma0,
                a = pfas["Szyszkowski_params"]["a"]["value"],
                b = pfas["Szyszkowski_params"]["b"]["value"],
                chi = 1,
                T = 293.15,
            )


    kawi_results = kawi_sorp.compute(Cw=1e-12)

    ret = Retardation(Kd = spsorption_result["Kd"], Kaw = kawi_results["Kaw"], aaw = aaw, kinetic ="False", bulk_density = bulk_dens, hydro_properties= water_results["hydro_properties"])
    ret_result = ret.compute()
        # Simulations
    sim_runner = EquilibriumSolver(
        grid=grid_results["grid"],
        bulk_density=bulk_dens,
        boundary_conditions=boundary_results["boundary_conditions"],
        hydro_properties=water_results["hydro_properties"],
        adsorption = ret_result["adsorption"],
        kinetic=False,
        volume_averaged=True,
        initial_contaminant_concentration= np.zeros(len(grid.depth))
    )
    final_results = sim_runner
    return (final_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results
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
