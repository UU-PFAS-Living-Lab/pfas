import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Example linear sorption
    In this example we will showcase how to use the pfas package to create a basic forward modelling exercise of PFAS leaching in the vadose zone.
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
    return mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading configuration file
    First we load our configuration file, in which we provide our model with most of the parameters needed to run our model. The input is later checked to see if it meets the requirements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Initializing Model and preprocessing data from TOML file

    We now create an instance of our model and do the preprocessing for different categories of our data. Simoultaneously, the input is checked for its validity. We run the analytical solution through the class SimulationRunner.

    In this step it is also possible to overwrite certain parameters from the TOML file, as examplified by the parameter porosity here.
    """)
    return


app._unparsable_cell(
    r"""
    model = Model()
    # Step 1: Generate the grid
    model.add(GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=10000
    ))

    model.add(WaterPreprocessor(
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        init_sat=0.2,
        residual_water_content=0.04
    ))
    # Step 3: Setup boundary conditions
    model.add(BoundaryPreprocessor(
        C_list=[10.0, 0],
        T_list=[0, 2000]
    ))

    model.add(KdDirectInput(
        Kd = 5.0
    ))

    model.add(FreundlichIsotherm(
        n_freund = 1.0
        C_rep = 1.0
    )) 

    model.add(SpRetardation(
    ))

 
    model.add(SwcAWI(
        scaling_factor_awi = 1.0,
        van_genuchten_alpha = 0.019,
        saturated_water_content = 0.34,
        sigma0 = 71
            
    ))

    model.add(Kaw_empirical(
        n_CF = 4, 
        functional_group = "CH3"
    ))          

    model.add(AWIRetardation(
    
    ))

    model.add(KineticSolver(
        frac_int = 0.03,
        rate_const = 0.01,
        initial_contaminant_concentration = 0
    ))

    #run it twice
    model.add(KineticSolver(
        frac_int = 0.03,
        rate_const = 0.01,
        initial_contaminant_concentration = model.C_tot
    ))

    model.outputs
    model.C1
    model.C2
    model.C_tot

    model.save_json("test.json") #time runs and layer runs
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Accessing generated data
    We access our data through model.generated_data, which prints all the output. Accessing it in this manneer also allows for easier plotting.
    """)
    return


@app.cell
def _(model):
    model.generated_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting of data

    The generated_data allows you to access both input and output of the model and thus also allows for relative simple plotting.
    """)
    return


@app.cell
def _(model):
    C_tot = model.generated_data["C_tot"]
    grid = model.generated_data["grid"]
    return C_tot, grid


@app.cell
def _(C_tot, grid, plt):
    #Breakthrough plot at bottom of grid over time
    plt.plot(grid.time, C_tot[0, :], label=f"Depth = {grid.depth} cm", color="blue")
    plt.xlabel("Time (s)") 
    plt.ylabel("Total PFAS Concentration (mg/L)")
    plt.title("PFAS Concentration Over Time")
    return


@app.cell
def _(C_tot, grid, plt):

    # Select specific time indices to plot
    t_len = C_tot.shape[1]
    time_indices = [0, t_len//4, t_len//2, 3*t_len//4, -1]  # First, and some intermediate, and last time step

    plt.figure(figsize=(8, 6))

    for t_idx in time_indices:
        plt.plot(C_tot[:, t_idx], grid.depth, label=f"t = {grid.time[t_idx]:.0f} s")

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
