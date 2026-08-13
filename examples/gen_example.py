import marimo

__generated_with = "0.23.16"
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
        Model,
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


@app.cell
def _(WaterPreprocessor):
    from pydantic_core import PydanticUndefined

    for k, v in WaterPreprocessor.model_fields.items():
        print(k, v)
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
    GridGenerator,
    Model,
    SWCAdsorptionPreprocessor,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    boundary_results,
    bulk_dens,
    grid_results,
    kawi_results,
    water_results,
):
    # Step 1: Generate the grid
    model = Model()
    model.compute(
        GridGenerator,
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=10000
    )

    # Step 2: Compute water flow properties
    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04
    )


    # Step 3: Setup boundary conditions
    model.compute(BoundaryPreprocessor,
        C_list=[10.0, 0],
        T_list=[0, 2000]
    )

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
    model.compute(
        SpRetardationPreprocessor,
        sorption_solid=sorption_solid,
        bulk_density=bulk_dens,
    )

    # Step 5: Compute AWI adsorption
    model.compute(
        SWCAdsorptionPreprocessor,
        sigma0=71,
        scaling_factor_awi=1.0,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {
                "scaling_factor_awi": 1.0
            },
        },
        van_genuchten_alpha = 0.019,
    #    saturated_water_content = 0.34,
    )

    # Step 6: Compute Kawi sorption
    model.compute(
        SorptionKawiDirectInput,
        kaw=0.5,
    )

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
