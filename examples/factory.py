import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    return (
        BoundaryPreprocessor,
        GridGenerator,
        Model,
        SWCAdsorptionPreprocessor,
        SimulationRunner,
        SorptionKawiDirectInput,
        SpRetardationPreprocessor,
        WaterPreprocessor,
        plt,
        read_toml,
    )


@app.cell
def _(read_toml):
    config = read_toml("examples/data/config.toml")
    return (config,)


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
    model.add(WaterPreprocessor, porosity=0.23)
    model.add(BoundaryPreprocessor)
    model.add(GridGenerator)
    model.add(SpRetardationPreprocessor)
    model.add(SWCAdsorptionPreprocessor)
    model.add(SorptionKawiDirectInput)
    model.add(SimulationRunner)
    return (model,)


@app.cell
def _(model):
    model.generated_data
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
