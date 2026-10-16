import marimo

__generated_with = "0.23.16"
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
    from pfas.data_structure import SimulationGrid
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
        SimulationGrid,
        WaterPreprocessor,
        mo,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time concatenation of solving

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
    num_time_steps = 10
    model_series = []

    def time_iteration(previous_models):
        model = Model()

        # Step 1: Generate the grid
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
        if len(previous_models) == 0:
            model.compute(BoundaryPreprocessor,
                C_list=[10.0, 0],
                T_list=[0, 2000]
            )
            initial_concentration = None
        else:
            model.compute(BoundaryPreprocessor,
                C_list=[0],
                T_list=[0]
            )
            initial_concentration = model_series[-1].C_tot[:, -1]


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
            initial_contaminant_concentration=initial_concentration
        )
        return model

    for i_time_step in range(num_time_steps):
        model_series.append(time_iteration(model_series))

    print("Simulation completed successfully!")
    return (model_series,)


@app.cell
def _(SimulationGrid, model_series, np):
    C1_time = np.concat([m.C1 for m in model_series], axis=1)
    sim_grid_time = SimulationGrid(time=np.concat([model_series[-1].grid.time+i*10000 for i in range(len(model_series))]), depth=model_series[-1].grid.depth)
    return C1_time, sim_grid_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Space concatenation of solving
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
    SimulationGrid,
    WaterPreprocessor,
):
    import numpy as np
    def space_iteration(previous_models):
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
        if len(previous_models) == 0:
            model.compute(BoundaryPreprocessor,
                C_list=[10.0, 0],
                T_list=[0, 2000]
            )
        else:
            model.compute(BoundaryPreprocessor,
                C_list=np.absolute(previous_models[-1].C_tot[-1]),
                T_list=model.grid.time,
            )

        # Step 4: Compute solid phase retardation
        sorption_solid_space = {
            "kinetic_sorption": True,
            "sorption_isotherm": "linear",
            "linear": {
                "Kd_method": "direct_input",
                "Kd": 5.0 #cm3/g
            },
        }
        model.compute(
            LinearSPsorption,
            sorption_solid=sorption_solid_space,
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
        return model



    num_layers = 10
    model_series_space = []

    # layer_id should be in the order in which they should be solved.
    for layer_id in range(num_layers):
        space_model = space_iteration(previous_models=model_series_space)
        model_series_space.append(space_model)
        print("Iteration ", layer_id, "complete")
    print("Simulation completed successfully!")

    C1_space = np.concat([m.C1 for m in model_series_space], axis=0)
    sim_grid_space = SimulationGrid(depth=np.concat([model_series_space[-1].grid.depth+i*60 for i in range(len(model_series_space))]), time=model_series_space[-1].grid.time)

    return C1_space, np, sim_grid_space


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results

    Here we plot C1 (the aqueous concentration) as a function of depth and of time.
    """)
    return


@app.cell
def _(plt):
    def plot_results(simulation_grid, C1):
        # 1. Concentration depth profiles
    
        t_len = C1.shape[1]
    
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
                C1[:, t_idx],
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
        bottom_concentration = C1[-1, :]
    
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

    return (plot_results,)


@app.cell
def _(C1_space, plot_results, sim_grid_space):
    plot_results(sim_grid_space, C1_space)
    return


@app.cell
def _(C1_time, plot_results, sim_grid_time):
    plot_results(sim_grid_time, C1_time)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
