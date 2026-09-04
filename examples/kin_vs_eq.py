import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Equilibrium vs. kinetic sorption — a pulse input

    This notebook runs the **same** transport problem twice — once with
    `EquilibriumSolver` (instantaneous sorption) and once with
    `KineticSolver` (rate-limited sorption) — and compares them.

    - **A single pulse** at the inlet: concentration `C0` from `t = 0` to
      `t = T_pulse`, then back to `0`.

    The grid is intentionally coarser (few depth/time points) for `KineticSolver`
    as this is *much* more expensive than `EquilibriumSolver` — a coarse grid keeps this
    notebook fast while still showing the qualitative difference between
    the two solvers.
    """)
    return


@app.cell
def _():
    from pfas.model import Model
    from pfas.component import (
        LinearSPsorption,
        SWCsorption,
        Retardation,
        KineticSolver,
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        EquilibriumSolver,
    )

    from matplotlib import pyplot as plt
    import numpy as np
    import marimo as mo
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        KineticSolver,
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
    ## Shared parameters

    Grid, water flow, and sorption parameters are identical between the two
    runs — only the sorption *mode* (equilibrium vs. kinetic) and the solver
    class differ. The pulse is defined the same way for both: a
    concentration `C0` switched on at `T_list[0]` and back off at
    `T_list[1]`.
    """)
    return


@app.cell
def _():

    bulk_dens = 1.6 #g cm-3
    Kd = 8 # cm3 g-1
    Kaw = 10 #cm3 cm-2

    # ------------------------------------------------------------
    # Pulse boundary condition: C0 from t=0 to t=T_PULSE_END, then 0
    # ------------------------------------------------------------
    C0 = 1.0 # mg L-1
    T_PULSE_END = 100 #s
    return C0, Kaw, Kd, T_PULSE_END, bulk_dens


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equilibrium solver
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    C0,
    EquilibriumSolver,
    GridGenerator,
    Kaw,
    Kd,
    LinearSPsorption,
    Model,
    Retardation,
    SWCsorption,
    T_PULSE_END,
    WaterPreprocessor,
    bulk_dens,
    np,
):
    model_eq = Model()
    GRID_KWARGS_eq = dict(
        domain_length=10, #cm
        spatial_resolution=0.01,
        time_resolution=1,
        time_total=500, #s
    )
    model_eq.compute(GridGenerator, **GRID_KWARGS_eq)

    n_nodes_eq = len(model_eq.grid.depth)
    initial_concentration_eq = np.zeros(n_nodes_eq)  # no initial contamination

    model_eq.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5, #cm/s
        hydraulic_conductivity=6, #cm/s
        porosity=0.34,
        dispersivity=1.5, #cm
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    model_eq.compute(
        BoundaryPreprocessor,
        C_list=[C0, 0],
        T_list=[0, T_PULSE_END],
    )

    sorption_solid_eq = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "linear": {"Kd_method": "direct_input", "Kd": Kd},
    }

    model_eq.compute(LinearSPsorption, sorption_solid=sorption_solid_eq)

    model_eq.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019, #1/s
    )

    model_eq.compute(
        Retardation,
        Kaw=Kaw,
        bulk_density=bulk_dens,
    )

    model_eq.compute(
        EquilibriumSolver,
        initial_contaminant_concentration=initial_concentration_eq,
    )

    print("Equilibrium simulation completed.")
    return (model_eq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinetic solver

    Rate-limited sorption: `kinetic_sorption=True` in `sorption_solid`, and
    `Retardation` is called **explicitly** with `kinetic=True` and
    `kin_params`.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    C0,
    GridGenerator,
    Kaw,
    Kd,
    KineticSolver,
    LinearSPsorption,
    Model,
    Retardation,
    SWCsorption,
    T_PULSE_END,
    WaterPreprocessor,
    bulk_dens,
    np,
):
    FRAC_INT = 0.8
    RATE_CONST = 0.1

    GRID_KWARGS_kin  = dict(
        domain_length=10,
        spatial_resolution=0.5,
        time_resolution=10,
        time_total=500,
    )

    model_kin = Model()

    model_kin.compute(GridGenerator, **GRID_KWARGS_kin)

    n_nodes_kin = len(model_kin.grid.depth)
    initial_concentration_kin = np.zeros(n_nodes_kin)  # no initial contamination

    model_kin.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    model_kin.compute(
        BoundaryPreprocessor,
        C_list=[C0, 0],
        T_list=[0, T_PULSE_END],
    )

    sorption_solid_kin = {
        "kinetic_sorption": True,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": FRAC_INT, "rate_const": RATE_CONST},
        "linear": {"Kd_method": "direct_input", "Kd": Kd},
    }

    model_kin.compute(LinearSPsorption, sorption_solid=sorption_solid_kin)

    model_kin.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019,
    )

    model_kin.compute(
        Retardation,
        Kaw=Kaw,
        bulk_density=bulk_dens,
        kinetic=True,
        kin_params={"frac_int": FRAC_INT, "rate_const": RATE_CONST},
    )

    model_kin.compute(
        KineticSolver,
        initial_contaminant_concentration=initial_concentration_kin,

    )

    print("Kinetic simulation completed.")
    return (model_kin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing the two: breakthrough curve at one depth

    Since there's no initial contamination, the interesting comparison is
    the **breakthrough curve** — concentration over time at a fixed depth —
    which shows how each solver responds to the same inlet pulse.
    """)
    return


@app.cell
def _(model_eq, model_kin, plt):
    z_eq = model_eq.grid.depth
    t_eq = model_eq.grid.time
    C1_eq = model_eq.generated_data["C1"]

    z_kin = model_kin.grid.depth
    t_kin = model_kin.grid.time
    C1_kin = model_kin.generated_data["C1"]

    obs_depth = z_eq[-1]  # end of domain

    fig_compare, ax_compare = plt.subplots(figsize=(9, 5))

    ax_compare.plot(
        t_eq,
        C1_eq[-5, :],
        label="Equilibrium",
        color="steelblue",
        linewidth=2,
    )

    ax_compare.plot(
        t_kin,
        C1_kin[-1, :],
        label="Kinetic (frac_int=0.8, rate_const=0.1)",
        color="firebrick",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
    )

    ax_compare.set_xlabel("Time (s)")
    ax_compare.set_ylabel("Aqueous PFAS concentration (mg/L)")
    ax_compare.set_title(f"Breakthrough curve at depth z = {obs_depth:.1f} cm")
    ax_compare.legend()
    ax_compare.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
