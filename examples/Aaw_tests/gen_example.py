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
    ## Defining shared parameters betweenthe model classes
    Some classes require the same parameters. When providing a TOML file, this is handled correctly, but when providing the parameters seperately to the different classes, we need to take care of this ourselves.

    In the next line of code, we will define bulk density. Furthermore, in our calling of the different classes, we will reuse some of the parameters that we have defined (`i.e. "residual_water_content": water_prep.residual_water_content` )
    """)
    return


@app.cell
def _():
    # Shared parameters: 

    bulk_dens = 1.6
    return (bulk_dens,)


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
    GridGenerator,
    SWCAdsorptionPreprocessor,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    bulk_dens,
):
    # Step 1: Generate the grid
    grid_gen = GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=10000
    )
    grid_results = grid_gen.compute()

    # Step 2: Compute water flow properties
    water_prep = WaterPreprocessor(
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        init_sat=0.2,
        residual_water_content=0.04
    )
    water_results = water_prep.compute()

    # Step 3: Setup boundary conditions
    boundary_prep = BoundaryPreprocessor(
        C_list=[10.0, 0],
        T_list=[0, 2000]
    )
    boundary_results = boundary_prep.compute()

    # Step 4: Compute solid phase retardation
    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {
            "frac_int": 0.3,
            "rate_const": 0.01
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5.0
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
            "van_genuchten_alpha": 0.019,
            "van_genuchten_n": water_prep.van_genuchten_n,
            "saturated_water_content": 0.34,
            "residual_water_content": water_prep.residual_water_content,
            "hydraulic_conductivity": water_prep.hydraulic_conductivity,
            "dispersivity": water_prep.dispersivity
        }
    )
    awi_results = swc_adsorp.compute()

    # Step 6: Compute Kawi sorption
    kawi_sorp = SorptionKawiDirectInput(
        kaw=0.5,
        hydro_properties=water_results["hydro_properties"],
        aaw=awi_results["aaw"],
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
        volume_averaged=True
    )
    final_results = sim_runner.compute()
    print("Simulation completed successfully!")
    return final_results, grid_results


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
