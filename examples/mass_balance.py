import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mass balance

    In this example we will model a simple case and check our mass balance.
    """)
    return


@app.cell
def _():
    from pfas.component import LinearSPsorption, Le2021_langmuir, Szyszkowski, Retardation, EquilibriumSolver, WaterPreprocessor, BoundaryPreprocessor, GridGenerator
    from pfas.data_loader import load_dataset, available_datasets
    from pfas.component.awi import SWCsorption, GuoTracer, D50AWI, NonlinearD50AWI, GSSAAWI
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import marimo as mo
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        LinearSPsorption,
        Model,
        Retardation,
        WaterPreprocessor,
        mo,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared parameters
    """)
    return


@app.cell
def _():
    bulk_dens = 1580  # kg/m3
    return (bulk_dens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building and running the model

    Grid, water flow, boundary conditions, solid-phase sorption, AWI adsorption,
    and the final solve are all run in sequence on a single `Model` instance.
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
    WaterPreprocessor,
    bulk_dens,
):
    model = Model()

    # Step 1: Generate the grid
    model.compute(
        GridGenerator,
        domain_length=5,          # 5 m deep
        spatial_resolution=0.025,  # 200 grid cells
        time_resolution=0.1,
        time_total=25,             # 25 yrs
    )

    # Step 2: Compute water flow properties
    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=0.869,   # m/yr
        hydraulic_conductivity=1763,       # m
        porosity=0.363,
        dispersivity=24.39 / 100,          # m
        van_genuchten_n=2.72,
        residual_water_content=0.054,
    )
    print(model.hydro_properties)

    # Step 3: Setup boundary conditions
    model.compute(
        BoundaryPreprocessor,
        C_list=[10.0e-6, 0],  # 10 ng/L in mg/L, then clean water
        T_list=[0, 25],       # pulse from t=0 to t=25 yrs
    )

    # Step 4: Compute solid-phase sorption (Kd)
    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5.3e-4,  # m3/kg
        },
    }
    model.compute(LinearSPsorption, sorption_solid=sorption_solid)

    # Step 6: Compute effective Kaw and run Retardation + solve
    a = 1.499900e-01  # mol/m3 (PFOS)
    MW = 414          # g/mol (PFOS)

    C_rep_mgL = 10e-6
    C_rep = C_rep_mgL / MW  # mol/m3
    kaw = 5.81e-06
    kaw_eff = kaw / (a + C_rep)


    model.compute(
        Retardation,
        Kaw=kaw_eff,
        aaw=34176,  # m2/m3 
        bulk_density=bulk_dens,
    )
    model.compute(EquilibriumSolver)

    print("Simulation completed successfully!")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mass-balance check at the base of the profile, steady state

    Recomputes total concentration from its component phases (liquid, solid,
    air-water interface) and compares against the solver's own `C_tot` output.
    """)
    return


@app.cell
def _(bulk_dens, model, pd):
    depth_idx = -1
    t_idx_ss = -1

    C1_ss = model.C1[depth_idx, t_idx_ss]
    C_tot_solver = model.C_tot[depth_idx, t_idx_ss]

    theta = model.hydro_properties.water_content
    rho_b = bulk_dens
    Kd = model.Kd
    Kaw = model.Kaw
    Aaw = model.aaw

    Cl = C1_ss * theta
    Cs = Kd * rho_b * C1_ss
    Cawi = Kaw * Aaw * C1_ss
    C_tot = Cl + Cs + Cawi
    Cl_m3w = C1_ss
    Cs_kg = Kd * C1_ss

    df = pd.DataFrame({
        "Phase": ["C_tot (recomputed)", "C_tot (solver)", "Cl", "Cs", "Cawi", "Cl", "Cs"],
        "Units": ["mg/m3b", "mg/m3b", "mg/m3b", "mg/m3b", "mg/m3b", "mg/m3w", "mg/kg"],
        "Value": [C_tot, C_tot_solver, Cl, Cs, Cawi, Cl_m3w, Cs_kg],
    })

    pd.set_option("display.float_format", "{:.3e}".format)
    print(f"Concentrations at depth = {model.grid.depth[depth_idx]:.1f} m, "
          f"t = {model.grid.time[t_idx_ss]:.1f} yr")
    print(df.to_string(index=False))
    print(f"\nMismatch (recomputed - solver): {C_tot - C_tot_solver:.3e} mg/m3b")
    print(f"Relative mismatch:              {(C_tot - C_tot_solver) / C_tot_solver * 100:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Depth profiles at selected times
    """)
    return


@app.cell
def _(model, np, plt):
    times_in_years = [0, 0.5, 1, 2, 3, 6, 11, 24.9]
    time_indices = [np.argmin(np.abs(model.grid.time - t)) for t in times_in_years]

    fig_depth, ax_depth = plt.subplots(figsize=(8, 6))
    for t_idx in time_indices:
        ax_depth.plot(
            model.C1[:, t_idx] / 414.07,
            model.grid.depth,
            label=f"t = {model.grid.time[t_idx]:.0f} yr",
        )
    ax_depth.set_xlabel("Total PFAS Concentration (mol/m3)")
    ax_depth.set_ylabel("Depth (cm)")
    ax_depth.set_title("PFAS Concentration Depth Profile at Different Times")
    ax_depth.legend()
    ax_depth.invert_yaxis()
    ax_depth.grid(True, alpha=0.3)
    fig_depth.tight_layout()
    fig_depth
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Breakthrough curve at the bottom of the profile
    """)
    return


@app.cell
def _(model, plt):
    depth_idx_bt = -1

    fig_bt, ax_bt = plt.subplots(figsize=(8, 6))
    ax_bt.plot(model.grid.time, model.C1[depth_idx_bt, :] * 1e6)
    ax_bt.set_xlabel("Time (yr)")
    ax_bt.set_ylabel("PFAS Concentration (ng/L)")
    ax_bt.set_title(f"Breakthrough Curve at depth = {model.grid.depth[depth_idx_bt]:.0f} m")
    ax_bt.grid(True, alpha=0.3)
    fig_bt.tight_layout()
    fig_bt
    return


if __name__ == "__main__":
    app.run()
