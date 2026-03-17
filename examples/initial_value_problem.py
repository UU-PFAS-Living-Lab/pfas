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
    from pfas.preprocessing import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        SpRetardationPreprocessor,
        SWCAdsorptionPreprocessor,
        SorptionKawiDirectInput,
        SimulationRunner,
    )
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import numpy as np
    import marimo as mo
    return (
        BoundaryPreprocessor,
        GridGenerator,
        SWCAdsorptionPreprocessor,
        SimulationRunner,
        SorptionKawiDirectInput,
        SpRetardationPreprocessor,
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


@app.cell
def _():
    # Shared parameters
    bulk_dens = 1.6
    return (bulk_dens,)


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
    GridGenerator,
    SWCAdsorptionPreprocessor,
    SimulationRunner,
    SorptionKawiDirectInput,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    bulk_dens,
    np,
):
    # Step 1: Generate the grid
    grid_gen = GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=5000,
    )
    grid_results = grid_gen.compute()

    # Step 2: Build initial conditions — concentration of 1 in the top 50% of the grid
    simulation_grid = grid_results["grid"]
    n_nodes = len(simulation_grid.depth)
    cutoff = n_nodes // 2  # top 50% of depth nodes

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 1.0  # top half set to 1 mg/L

    print(f"Grid has {n_nodes} depth nodes.")
    print(f"Initial concentration of 1 mg/L applied to top {cutoff} nodes (depth 0 – {simulation_grid.depth[cutoff - 1]:.1f} cm).")
    print(f"Bottom {n_nodes - cutoff} nodes initialised to 0.")

    # Step 3: Compute water flow properties
    water_prep = WaterPreprocessor(
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        init_sat=0.34,
        residual_water_content=0.04,
    )
    water_results = water_prep.compute()

    # Step 4: Setup boundary conditions
    # The influx concentration is kept at 0 so that the simulation evolves
    # purely from the initial conditions rather than a continuous source.
    # Change solute_concentration_influx to a non-zero value if you also
    # want an ongoing surface flux.
    boundary_prep = BoundaryPreprocessor(
        average_infiltration_rate=10,
        solute_concentration_influx=5.0,
        pulse_intervals= [(0,2000)],
    )
    boundary_results = boundary_prep.compute()

    # Step 5: Compute solid-phase retardation
    sorption_solid = {
        "kinetic_sorption": True,
        "sorption_isotherm": "linear",
        "kinetic": {
            "frac_int": 0.3,
            "rate_const": 0.01,
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5.0,
        },
    }
    sp_retard = SpRetardationPreprocessor(
        sorption_solid=sorption_solid,
        bulk_density=bulk_dens,
        hydro_properties=water_results["hydro_properties"],
    )
    sp_results = sp_retard.compute()

    # Step 6: Compute AWI adsorption
    swc_adsorp = SWCAdsorptionPreprocessor(
        hydro_properties=water_results["hydro_properties"],
        sigma0=71,
        scaling_factor_awi=1.0,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {
                "scaling_factor_awi": 1.0,
            },
        },
        soil={
            "bulk_density": bulk_dens,
            "porosity": water_prep.porosity,
            "van_genuchten_alpha": 0.019,
            "van_genuchten_n": water_prep.van_genuchten_n,
            "saturated_water_content": 0.34,
            "residual_water_content": water_prep.residual_water_content,
            "hydraulic_conductivity": water_prep.hydraulic_conductivity,
            "dispersivity": water_prep.dispersivity,
        },
    )
    awi_results = swc_adsorp.compute()

    # Step 7: Compute Kawi sorption
    kawi_sorp = SorptionKawiDirectInput(
        kaw=0.5,
        hydro_properties=water_results["hydro_properties"],
        aaw=awi_results["aaw"],
    )
    kawi_results = kawi_sorp.compute()

    # Step 8: Run simulation — passing initial_concentration
    sim_runner = SimulationRunner(
        grid=grid_results["grid"],
        bulk_density=bulk_dens,
        boundary_conditions=boundary_results["boundary_conditions"],
        hydro_properties=water_results["hydro_properties"],
        awi_retardation=kawi_results["awi_retardation"],
        sorption_solid=sorption_solid,
        kinetic_sorption=True,
        volume_averaged=False,
        initial_contaminant_concentration=initial_concentration,  # <-- initial conditions
    )
    final_results = sim_runner.compute()
    print("Simulation completed successfully!")
    return cutoff, final_results, initial_concentration, simulation_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results

    The left panel shows the **initial concentration profile** (concentration of 1 in the top half, 0 below). The right panel shows how that profile evolves through time.
    """)
    return


@app.cell
def _(cutoff, final_results, initial_concentration, plt, simulation_grid):
    t_len = final_results["C_tot"].shape[1]
    time_indices = [0, t_len // 4, t_len // 2, 3 * t_len // 4, -1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Left panel: initial condition ---
    ax_init = axes[0]
    ax_init.plot(initial_concentration, simulation_grid.depth, color="steelblue", linewidth=2)
    ax_init.axhline(
        y=simulation_grid.depth[cutoff - 1],
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"50 % depth boundary ({simulation_grid.depth[cutoff - 1]:.0f} cm)",
    )
    ax_init.set_xlabel("Initial PFAS Concentration (mg/L)")
    ax_init.set_ylabel("Depth (cm)")
    ax_init.set_title("Initial Condition")
    ax_init.invert_yaxis()
    ax_init.legend(fontsize=8)
    ax_init.grid(True, alpha=0.3)

    # --- Right panel: time evolution ---
    ax_time = axes[1]
    for t_idx in time_indices:
        ax_time.plot(
            final_results["C_tot"][:, t_idx],
            simulation_grid.depth,
            label=f"t = {simulation_grid.time[t_idx]:.0f} s",
        )
    ax_time.set_xlabel("Total PFAS Concentration (mg/L)")
    ax_time.set_title("PFAS Concentration Depth Profile at Different Times")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    plt.suptitle("Initial Concentration in Top 50 % of Column", fontsize=13)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
