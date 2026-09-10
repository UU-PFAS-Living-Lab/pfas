import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic simulation
    In this example, we will showcase the basics of initializing our model instance.

    We consider a 60cm long domain, in which we simulate a 10 mg/L pulse of a fictional PFAS for 2000s from the beginning of the considered model time. We run our model for a total of 10000s. There is no contamination present at the start.

    We keep everything else relatively simple, with direct input of sorption parameters $K_d$ and $K_aw$. We compute air-water interfacial area based on the soil-water characteristic.
    We consider equilibrium sorption as well.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.component import SWCsorption, LinearSPsorption, Retardation, WaterPreprocessor, BoundaryPreprocessor, GridGenerator
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
        average_infiltration_rate=1.5, #cm/s
        hydraulic_conductivity=6, #cm/s
        porosity=0.34,
        dispersivity=1.5, #cm
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
        "kinetic_sorption": True,
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5.0 #cm3/g
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
        bulk_density=1.6 #g/cm3,
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

    Here we plot C1 (the aqueous concentration) as a function of depth and of time.
    """)
    return


@app.cell
def _(model, plt):
    simulation_grid = model.grid
    # 1. Concentration depth profiles

    t_len = model.C1.shape[1]

    time_indices = [
        0,
        10,
        20,
        22,
        30,
    ]

    plt.figure(figsize=(8, 6))

    for t_idx in time_indices:
        plt.plot(
            model.C1[:, t_idx],
            simulation_grid.depth,
            label=f"t = {simulation_grid.time[t_idx]:.0f} s",
        )

    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Breakthrough curve at bottom of model
    bottom_concentration = model.C1[-1, :]

    plt.figure(figsize=(8, 5))

    plt.plot(
        simulation_grid.time,
        bottom_concentration,
        linewidth=2,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("PFAS Concentration at Bottom (mg/L)")
    plt.title("PFAS Breakthrough Curve at Bottom of Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
