import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Example Linear Sorption""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loading in dependencies
    First we load in the necessary packages and functions
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pfas 
    import numpy as np
    from pfas.preprocessing import preprocess_configuration, run_simulation
    from pfas.configuration import read_toml, validate_config
    from pfas.analytical_soln import SimulationGrid, BoundaryConditions, HydrologicalProperties, Adsorption
    import matplotlib.pyplot as plt
    return Path, mo, pfas, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Importing and checking the toml file

    We load our default configuration file, including most of the parameters needed for our simulation. We use the validation function to ensure we have all needed parameters (for our basic scenario) and have no values outside of the expected ranges. If there are no problems, the function will return **True**.

    We will have the opportunity to override values from the toml file #TODO
    """
    )
    return


@app.cell
def _(Path, pfas):
    toml_path = Path("examples", "data", "config.toml")
    toml_file = pfas.configuration.read_toml(toml_path)
    pfas.configuration.validate_config(toml_file)
    return (toml_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preprocessing data
    We will preprocess our data to make it ready for our analysis. We have the opportunity to do this seperately for each category in the toml file (e.g. *water_preprocessing*) or do it at once through the function (e.g. *preprocess_configuration*) or directly run it through *run_simulation*
    """
    )
    return


@app.cell
def _(pfas, toml_file):
    params = pfas.preprocessing.preprocess_configuration(toml_file)
    return (params,)


@app.cell
def _(params, pfas):
    C1, C2, C_tot, grid = pfas.preprocessing.run_simulation(params)
    return C2, C_tot, grid


@app.cell
def _(C2):
    C2.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting
    Now we can plot our results, we will have better plotting functions later
    """
    )
    return


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
