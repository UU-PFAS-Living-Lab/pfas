import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic simulation
    In this example, we will showcase the basics of initializing our model instance.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.component import SWCsorption, LinearSPsorption, Retardation, WaterPreprocessor, BoundaryPreprocessor, GridGenerator
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import marimo as mo
    from pfas.component import EquilibriumSolver
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        LinearSPsorption,
        Model,
        Retardation,
        SWCsorption,
        WaterPreprocessor,
        mo,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining shared parameters between the model classes
    Some classes require the same parameters. When providing a TOML file, this is handled correctly, but when providing the parameters seperately to the different classes, we need to take care of this ourselves.

    In the next line of code, we will define bulk density. Furthermore, in our calling of the different classes, we will reuse some of the parameters that we have defined (`i.e. "residual_water_content": water_prep.residual_water_content` )
    """)
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
    GridGenerator,
    LinearSPsorption,
    Model,
    Retardation,
    SWCsorption,
    WaterPreprocessor,
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
        LinearSPsorption,
        sorption_solid=sorption_solid,
    )

    # Step 5: Compute AWI adsorption
    model.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha = 0.019,
    )

    # Step 6: Compute retardation
    model.compute(
        Retardation,
        Kaw=0.5,
        bulk_density=1.6,
    )

    # Step 7: Run simulation
    model.compute(
        EquilibriumSolver,
    )

    print("Simulation completed successfully!")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results
    """)
    return


@app.cell
def _(model, plt):
    simulation_grid = model.grid
    # Select specific time indices to plot
    t_len = model.C_tot.shape[1]
    time_indices = [0, t_len//4, t_len//2, 3*t_len//4, -1]  # First, and some intermediate, and last time step

    plt.figure(figsize=(8, 6))

    for t_idx in time_indices:
        plt.plot(model.C_tot[:, t_idx], simulation_grid.depth, label=f"t = {simulation_grid.time[t_idx]:.0f} s")

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
