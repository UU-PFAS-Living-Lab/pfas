import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running simulations — with initial conditions
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
        time_resolution=20,
        time_total=10000,
    )

    # ── Step 2: Build initial conditions — concentration of 1 in top 50% ──────
    n_nodes = len(model.grid.depth)
    cutoff = n_nodes // 2  # top 50% of depth nodes

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 0.05  # top half set to 1 mg/L

    print(f"Grid has {n_nodes} depth nodes.")
    print(f"Initial concentration of 1 mg/L applied to top {cutoff} nodes "
          f"(depth 0 – {model.grid.depth[cutoff - 1]:.1f} cm).")
    print(f"Bottom {n_nodes - cutoff} nodes initialised to 0.")

    # ── Step 3: Compute water flow properties ───────────────────────────────────
    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
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
            "frac_int": 0.8,
            "rate_const": 0.1,
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 0,
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
    model.compute(Retardation, Kaw=0, bulk_density=bulk_dens)
    model.compute(
        EquilibriumSolver,
        initial_contaminant_concentration=initial_concentration,
        bc = "flux"
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


@app.cell
def _(np, plt):

    from scipy.special import erfc

    # ============================================================
    # Standalone IVP test
    # ============================================================

    # ------------------------------------------------------------
    # 1. Physical domain
    # ------------------------------------------------------------
    L = 60.0                  # cm
    dx = 0.5                  # cm

    x = np.arange(0.0, L + dx, dx)

    # Dimensionless spatial coordinate
    Z = x / L

    # Initial concentration: 5 mg/L in top 50% of domain
    Ci = np.zeros_like(x, dtype=float)
    Ci[x < L / 2] = 5.0

    # Dimensionless coordinate associated with Ci
    xi = x / L

    print("Number of spatial nodes:", len(x))
    print("Physical domain:", x[0], "to", x[-1], "cm")
    print("Dimensionless domain:", Z[0], "to", Z[-1])
    print("Initial concentration:", Ci.min(), "to", Ci.max(), "mg/L")
    print("Initial contaminated depth:", x[Ci > 0].min(), "to", x[Ci > 0].max(), "cm")


    # ------------------------------------------------------------
    # 2. Transport parameters
    # ------------------------------------------------------------
    R = 1.0       # Start with R=1 to isolate the IVP
    P = 40.0      # Peclet number

    # Dimensionless times
    T_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]


    # ------------------------------------------------------------
    # 3. Resident-boundary IVP Green's function
    # ------------------------------------------------------------
    def ivp_eq_resident(
        T,
        R,
        Z,
        P,
        xi,
    ):
        """Initial-value Green's function, first-type boundary."""

        if T <= 0.0:
            raise ValueError("Green's function is only evaluated for T > 0.")

        prefactor = np.sqrt(
            R * P / (4.0 * np.pi * T)
        )

        scale = 4.0 * R * T / P

        direct = np.exp(
            -(R * (xi - Z) - T) ** 2 / scale
        )

        image = np.exp(P * Z) * np.exp(
            -(R * (xi + Z) - T) ** 2 / scale
        )

        return prefactor * (direct - image)


    # ------------------------------------------------------------
    # 4. Calculate IVP solution
    # ------------------------------------------------------------
    C_ivp = np.zeros((len(Z), len(T_values)))

    for ti, T in enumerate(T_values):

        # The Green's function becomes a delta function at T=0.
        # Therefore impose the initial condition directly.
        if np.isclose(T, 0.0):
            C_ivp[:, ti] = Ci
            continue

        for zi, z in enumerate(Z):

            kernel = ivp_eq_resident(
                T=T,
                R=R,
                Z=z,
                P=P,
                xi=xi,
            )

            C_ivp[zi, ti] = np.trapezoid(
                kernel * Ci,
                xi,
            )


    # ------------------------------------------------------------
    # 5. Basic sanity checks
    # ------------------------------------------------------------
    print("\n===== SANITY CHECKS =====")

    # Initial condition must be recovered exactly
    initial_error = np.max(
        np.abs(C_ivp[:, 0] - Ci)
    )

    print(
        "Maximum error at T=0:",
        initial_error,
        "mg/L"
    )

    assert np.allclose(
        C_ivp[:, 0],
        Ci,
        atol=1e-12,
    ), "Initial condition is NOT reproduced at T=0!"


    # Check for negative concentrations
    print(
        "Minimum concentration:",
        C_ivp.min(),
        "mg/L"
    )

    if C_ivp.min() < -1e-10:
        print("WARNING: negative concentrations detected.")


    # Check maximum concentration
    print(
        "Maximum concentration:",
        C_ivp.max(),
        "mg/L"
    )


    # ------------------------------------------------------------
    # 6. Print concentration profiles
    # ------------------------------------------------------------
    print("\n===== PROFILE SUMMARY =====")

    for ti, T in enumerate(T_values):

        profile = C_ivp[:, ti]

        max_idx = np.argmax(profile)

        print(
            f"T={T:6.3f} | "
            f"max={profile[max_idx]:8.4f} mg/L | "
            f"location={x[max_idx]:7.2f} cm"
        )


    # ------------------------------------------------------------
    # 7. Plot the initial condition and IVP evolution
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    for ti, T in enumerate(T_values):

        ax.plot(
            x,
            C_ivp[:, ti],
            label=f"T = {T:g}",
        )

    ax.axvline(
        L / 2,
        linestyle="--",
        linewidth=1,
        label="Initial concentration boundary",
    )

    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Aqueous concentration (mg/L)")
    ax.set_title("Standalone IVP test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
