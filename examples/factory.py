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
        read_toml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading configuration file
    First we load our configuration file, in which we provide our model with most of the parameters needed to run our model. The input is later checked to see if it meets the requirements.
    """)
    return


@app.cell
def _(read_toml):
    config = read_toml("examples/data/config.toml")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Initializing Model and preprocessing data from TOML file

    We now create an instance of our model and do the preprocessing for different categories of our data. Simoultaneously, the input is checked for its validity. We run the analytical solution through the class SimulationRunner.

    In this step it is also possible to overwrite certain parameters from the TOML file, as examplified by the parameter porosity here.
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
    config,
):
    model = Model(config)
    model.add(WaterPreprocessor, porosity=0.4)
    model.add(BoundaryPreprocessor)
    model.add(GridGenerator)
    model.add(SpRetardationPreprocessor)
    model.add(SWCAdsorptionPreprocessor)
    model.add(SorptionKawiDirectInput)
    model.add(SimulationRunner)
    return (model,)


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
