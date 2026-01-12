import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pfas 
    from pprint import pprint
    import numpy as np
    from pfas.preprocessing import preprocess_configuration, run_simulation
    from pfas.toml_handler import read_toml, validate_config
    from pfas.analytical_soln import SimulationGrid, BoundaryConditions, HydrologicalProperties, Adsorption
    import matplotlib.pyplot as plt

    return mo, pfas, pprint


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Overriding example

    In this example we load in the basic config file with and process its default calculations, but adjust the values after.
    """
    )
    return


@app.cell
def _(pfas, pprint):
    toml_file = pfas.toml_handler.read_toml("examples\data\config.toml")
    ## Directly in the toml file 

    pfas.toml_handler.validate_config(toml_file)

    parameters= pfas.preprocessing.preprocess_configuration(toml_file)

    pprint(parameters, sort_dicts=False)

    parameters["bulk_density"]= 1.35

    #now changed to 1.35
    pprint(parameters, sort_dicts=False)

    return


if __name__ == "__main__":
    app.run()
