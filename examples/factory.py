import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner
    from pfas.configuration import read_toml
    from pfas.model import Model
    return (
        BoundaryPreprocessor,
        GridGenerator,
        Model,
        SWCAdsorptionPreprocessor,
        SimulationRunner,
        SorptionKawiDirectInput,
        SpRetardationPreprocessor,
        WaterPreprocessor,
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


if __name__ == "__main__":
    app.run()
