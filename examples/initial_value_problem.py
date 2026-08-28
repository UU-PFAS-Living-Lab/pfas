import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running simulations without TOML — with initial conditions
    In this example, we run a simulation without a TOML file, but with **initial conditions**: a PFAS concentration of 1 mg/L pre-loaded into the top 50% of the soil column.
    """)
    return


@app.cell
def _():
    # Loading relevant modules
    from pfas.model import Model
    from pfas.component import LinearSPsorption, SWCsorption, Retardation, KineticSolver, WaterPreprocessor, BoundaryPreprocessor, GridGenerator, FreundlichSPsorption, EquilibriumSolver

    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import numpy as np
    import marimo as mo
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
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining shared parameters between the model classes
    Some classes require the same parameters. When providing a TOML file, this is handled correctly, but when providing the parameters separately to the different classes, we need to take care of this ourselves.

    Here we define `bulk_dens`, which is reused across multiple preprocessors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling different classes and running the model

    We follow the same pipeline as before, but with one addition: after generating the grid, we construct an **initial condition array** where the top 50% of grid nodes are assigned a PFAS concentration of 1 mg/L and the bottom 50% start at 0.
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
    np,
):
    # ── Shared parameters ───────────────────────────────────────────────────────
    bulk_dens = 1.6

    model = Model()

    # ── Step 1: Generate the grid ───────────────────────────────────────────────
    model.compute(
        GridGenerator,
        domain_length=60,
        spatial_resolution=0.5,
        time_resolution=50,
        time_total=10000,
    )

    # ── Step 2: Build initial conditions — concentration of 1 in top 50% ──────
    n_nodes = len(model.grid.depth)
    cutoff = n_nodes // 2  # top 50% of depth nodes

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 1.0  # top half set to 1 mg/L

    print(f"Grid has {n_nodes} depth nodes.")
    print(f"Initial concentration of 1 mg/L applied to top {cutoff} nodes "
          f"(depth 0 – {model.grid.depth[cutoff - 1]:.1f} cm).")
    print(f"Bottom {n_nodes - cutoff} nodes initialised to 0.")

    # ── Step 3: Compute water flow properties ───────────────────────────────────
    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5e-2,
        hydraulic_conductivity=6e-1,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    # ── Step 4: Setup boundary conditions ───────────────────────────────────────
    model.compute(
        BoundaryPreprocessor,
        C_list=[0],
        T_list = [0])

    # ── Step 5: Compute solid-phase sorption (Kd) ───────────────────────────────
    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {
            "frac_int": 0.3,
            "rate_const": 0.1,
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5,
        },
    }
    model.compute(LinearSPsorption, sorption_solid=sorption_solid)

    # ── Step 6: Compute AWI adsorption ──────────────────────────────────────────
    model.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019,
    )

    # ── Step 7 + 8: Retardation (Kaw supplied directly) and solve ──────────────
    model.compute(Retardation, Kaw=0.5, bulk_density=bulk_dens)
    model.compute(
        EquilibriumSolver,
        initial_contaminant_concentration=initial_concentration,
    )

    print("Simulation completed successfully!")
    return cutoff, initial_concentration, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results

    The left panel shows the **initial concentration profile** (concentration of 1 in the top half, 0 below). The right panel shows how that profile evolves through time.
    """)
    return


@app.cell
def _(cutoff, initial_concentration, model, plt):
    t_len = model.C_tot.shape[1]
    time_indices = [0, t_len // 4, t_len // 2, 3 * t_len // 4, -1]

    fig_init_time, axes_init_time = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Left panel: initial condition ---
    ax_init = axes_init_time[0]
    ax_init.plot(initial_concentration, model.grid.depth, color="steelblue", linewidth=2)
    ax_init.axhline(
        y=model.grid.depth[cutoff - 1],
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"50 % depth boundary ({model.grid.depth[cutoff - 1]:.0f} cm)",
    )
    ax_init.set_xlabel("Initial PFAS Concentration (mg/L)")
    ax_init.set_ylabel("Depth (cm)")
    ax_init.set_title("Initial Condition")
    ax_init.invert_yaxis()
    ax_init.legend(fontsize=8)
    ax_init.grid(True, alpha=0.3)

    # --- Right panel: time evolution ---
    ax_time = axes_init_time[1]
    for t_idx in time_indices:
        ax_time.plot(
            model.C_tot[:, t_idx],
            model.grid.depth,
            label=f"t = {model.grid.time[t_idx]:.0f} s",
        )
    ax_time.set_xlabel("Aqueous PFAS concentration (mg/L)")
    ax_time.set_title("PFAS Concentration Depth Profile at Different Times")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    fig_init_time.suptitle("Initial Concentration in Top 50 % of Column", fontsize=13)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
