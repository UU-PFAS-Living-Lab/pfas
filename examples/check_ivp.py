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


    # ============================================================
    # RUN MODEL
    # ============================================================

    bulk_dens = 1.6

    model = Model()

    model.compute(
        GridGenerator,
        domain_length=60,
        spatial_resolution=0.5,
        time_resolution=20,
        time_total=10000,
    )

    n_nodes = len(model.grid.depth)
    cutoff = n_nodes // 2

    initial_concentration = np.zeros(n_nodes)
    initial_concentration[:cutoff] = 0.5

    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    model.compute(
        BoundaryPreprocessor,
        C_list=[0],
        T_list=[0],
    )

    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {
            "frac_int": 0.8,
            "rate_const": 0.1,
        },
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 8,
        },
    }

    model.compute(
        LinearSPsorption,
        sorption_solid=sorption_solid,
    )

    model.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019,
    )

    model.compute(
        Retardation,
        Kaw=10,
        bulk_density=bulk_dens,
    )

    model.compute(
        EquilibriumSolver,
        initial_contaminant_concentration=initial_concentration,
        bc="flux",
    )

    print("Simulation completed.")
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
    time_indices = [1, 4, 10, 50]

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
            model.C1[:, t_idx],
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
def _(initial_concentration, model, np):
    # ============================================================
    # INTERNAL MODEL TESTS
    # ============================================================
    # ============================================================
    # BLACK-BOX PHYSICAL / INTERNAL CONSISTENCY TESTS
    # ============================================================

    print("\n" + "=" * 70)
    print("BLACK-BOX TESTS — EQUILIBRIUM SOLVER")
    print("=" * 70)


    # ------------------------------------------------------------
    # 1. Inspect generated outputs
    # ------------------------------------------------------------

    print("\n[1] GENERATED DATA")

    for key, value in model.generated_data.items():
        print(f"\n{key}:")
        print(f"  type = {type(value)}")

        if isinstance(value, np.ndarray):
            print(f"  shape = {value.shape}")
            print(f"  min   = {np.nanmin(value)}")
            print(f"  max   = {np.nanmax(value)}")

        elif isinstance(value, dict):
            print(f"  keys = {list(value.keys())}")

        else:
            print(f"  value = {value}")


    # ------------------------------------------------------------
    # 2. Physical grid
    # ------------------------------------------------------------

    print("\n[2] PHYSICAL GRID")

    z = np.asarray(model.grid.depth)
    t = np.asarray(model.grid.time)

    L = z[-1]

    print("L =", L)
    print("number of spatial nodes =", len(z))
    print("z[0] =", z[0])
    print("z[-1] =", z[-1])

    print("number of time points =", len(t))
    print("t[0] =", t[0])
    print("t[-1] =", t[-1])

    assert np.all(np.diff(z) > 0)
    assert np.all(np.diff(t) >= 0)

    print("PASS")


    # ------------------------------------------------------------
    # 3. Water properties
    # ------------------------------------------------------------

    print("\n[3] WATER PROPERTIES")

    hydro = model.generated_data["hydro_properties"]

    theta = hydro.water_content
    v = hydro.pore_velocity
    D = hydro.dispersion_coefficient

    print("theta =", theta)
    print("v     =", v)
    print("D     =", D)

    assert 0 < theta <= 0.34
    assert v > 0
    assert D > 0

    # Independent calculation from your WaterPreprocessor
    expected_v = 1.5 / theta
    expected_D = expected_v * 1.5

    print("\nIndependent checks:")
    print("expected v =", expected_v)
    print("expected D =", expected_D)

    assert np.isclose(v, expected_v)
    assert np.isclose(D, expected_D)

    print("PASS")



    # ------------------------------------------------------------
    # 5. Retardation
    # ------------------------------------------------------------

    print("\n[5] RETARDATION")

    adsorption = model.generated_data["adsorption"]

    R = adsorption.total_retardation

    print("R =", R)

    # With Kd = 0 and Kaw = 0:
    # there should be no sorption retardation.
    #
    # Therefore:
    #
    #       R = 1

    assert 40 <= R <= 50, f"R={R} is outside the expected range [40, 50]"

    print("PASS")


    # ------------------------------------------------------------
    # 6. Initial condition
    # ------------------------------------------------------------

    print("\n[6] INITIAL CONDITION")

    Ci = np.asarray(initial_concentration)

    print("Ci shape =", Ci.shape)
    print("Ci min   =", Ci.min())
    print("Ci max   =", Ci.max())

    assert Ci.shape == z.shape
    assert np.isclose(Ci.min(), 0.0)
    assert np.isclose(Ci.max(), 0.5)

    # Check location of contamination
    contaminated = np.where(Ci > 0)[0]

    print(
        "contaminated depth =",
        z[contaminated[0]],
        "->",
        z[contaminated[-1]],
    )

    assert len(contaminated) > 0

    print("PASS")


    # ------------------------------------------------------------
    # 7. Solution
    # ------------------------------------------------------------

    print("\n[7] SOLUTION")

    C1 = model.generated_data["C1"]
    C_tot = model.generated_data["C_tot"]

    print("C1 shape    =", C1.shape)
    print("C_tot shape =", C_tot.shape)

    assert C1.shape == (len(z), len(t))
    assert C_tot.shape == C1.shape

    print("C1 range    =", np.nanmin(C1), "->", np.nanmax(C1))
    print("C_tot range =", np.nanmin(C_tot), "->", np.nanmax(C_tot))

    print("PASS")


    # ------------------------------------------------------------
    # 8. T = 0 test
    # ------------------------------------------------------------

    print("\n[8] INITIAL-TIME TEST")

    initial_error = np.max(
        np.abs(C1[:, 0] - Ci)
    )

    print("max |C(z,0) - Ci(z)| =", initial_error)

    # This is the physically expected condition:
    #
    #       C(z,0) = Ci(z)
    #
    # Do NOT enforce a tolerance yet.
    # We want to see what the existing implementation actually does.

    print("Expected physical error: approximately 0")
    print("Observed error:", initial_error)


    # ------------------------------------------------------------
    # 9. Negative concentration test
    # ------------------------------------------------------------

    print("\n[9] NEGATIVE CONCENTRATIONS")

    minimum = np.min(C1)

    print("minimum C =", minimum)

    negative_locations = np.where(C1 < -1e-12)

    print(
        "number of significantly negative values =",
        len(negative_locations[0]),
    )

    if len(negative_locations[0]) > 0:

        first = negative_locations[0][0]
        second = negative_locations[1][0]

        print(
            "first negative value:",
            C1[first, second],
        )

        print(
            "at depth =",
            z[first],
            "cm"
        )

        print(
            "at time =",
            t[second],
            "s"
        )


    # ------------------------------------------------------------
    # 10. Conservation / mass diagnostic
    # ------------------------------------------------------------

    print("\n[10] MASS DIAGNOSTIC")

    initial_mass = np.trapezoid(
        Ci,
        z,
    )

    print("Initial aqueous concentration integral:")
    print("  ∫Ci dz =", initial_mass)

    for j in [0, 1, 2, 5, 10, len(t) // 2, -1]:

        if j >= len(t):
            continue

        mass = np.trapezoid(
            C1[:, j],
            z,
        )

        print(
            f"t = {t[j]:8.1f} s | "
            f"∫C dz = {mass:.8g} | "
            f"ratio = {mass / initial_mass:.8g}"
        )


    print("\n" + "=" * 70)
    print("BLACK-BOX TESTS COMPLETE")
    print("=" * 70)
    return


@app.cell
def _():
    def run_standalone_ivp():
        import numpy as np
        from scipy.special import erfc

        # --------------------------------------------------------
        # Physical parameters
        # --------------------------------------------------------

        L = 60.0
        dz = 0.5
        dt = 20.0
        t_total = 10000.0

        theta = 0.34
        infiltration = 1.5
        dispersivity = 1.5

        R = 1.0

        v = infiltration / theta
        D = v * dispersivity
        P = v * L / D

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        z = np.arange(0.0, L + dz, dz)
        t = np.arange(0.0, t_total + dt, dt)

        Z = z / L
        T = t * v / L

        # --------------------------------------------------------
        # Initial condition
        # --------------------------------------------------------

        Ci = np.zeros_like(z)
        Ci[z < L / 2] = 0.05

        xi = z / L

        # --------------------------------------------------------
        # IVP Green's function
        # --------------------------------------------------------

        def ivp_resident(Ti, Ri, Zi, Pi, xii):

            if Ti == 0:
                return np.zeros_like(xii)

            scale = 4.0 * Ri * Ti / Pi

            prefactor = np.sqrt(
                Ri * Pi / (4.0 * np.pi * Ti)
            )

            direct = np.exp(
                -(Ri * (xii - Zi) - Ti) ** 2 / scale
            )

            image = np.exp(Pi * Zi) * np.exp(
                -(Ri * (xii + Zi) - Ti) ** 2 / scale
            )

            return prefactor * (direct - image)

        # --------------------------------------------------------
        # Solve
        # --------------------------------------------------------

        C = np.zeros((len(z), len(t)))

        for j, Tj in enumerate(T):

            if Tj == 0:
                C[:, j] = Ci
                continue

            for i, Zi in enumerate(Z):

                kernel = ivp_resident(
                    Tj,
                    R,
                    Zi,
                    P,
                    xi,
                )

                C[i, j] = np.trapezoid(
                    kernel * Ci,
                    xi,
                )

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        print("=" * 60)
        print("STANDALONE IVP")
        print("=" * 60)

        print(f"v = {v}")
        print(f"D = {D}")
        print(f"P = {P}")

        print("\nGrid:")
        print(f"  nodes = {len(z)}")
        print(f"  z = {z[0]} -> {z[-1]}")

        print("\nDimensionless:")
        print(f"  Z = {Z[0]} -> {Z[-1]}")
        print(f"  T = {T[0]} -> {T[-1]}")

        print("\nInitial condition:")
        print(f"  min = {Ci.min()}")
        print(f"  max = {Ci.max()}")

        print("\nSolution:")
        print(f"  min = {np.nanmin(C)}")
        print(f"  max = {np.nanmax(C)}")

        print("\nInitial-time error:")
        print(
            np.max(
                np.abs(C[:, 0] - Ci)
            )
        )

        print("\nNegative concentration:")
        print(f"  minimum = {C.min()}")

        return {
            "z": z,
            "t": t,
            "Z": Z,
            "T": T,
            "Ci": Ci,
            "C": C,
            "v": v,
            "D": D,
            "P": P,
            "R": R,
            "theta": theta,
        }


    standalone = run_standalone_ivp()

    def diagnose_ivp_kernel():

        import numpy as np

        L = 60.0
        theta = 0.34
        infiltration = 1.5
        dispersivity = 1.5
        R = 1.0

        v = infiltration / theta
        D = v * dispersivity
        P = v * L / D

        z = np.arange(0.0, L + 0.5, 0.5)
        xi = z / L

        # Examine several dimensionless times and depths
        test_T = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
        test_Z = [0.0, 0.25, 0.5, 0.75, 1.0]

        print("=" * 70)
        print("IVP KERNEL DIAGNOSTIC")
        print("=" * 70)

        print(f"P = {P}")
        print()

        for T in test_T:

            scale = 4 * R * T / P
            prefactor = np.sqrt(
                R * P / (4 * np.pi * T)
            )

            for Z in test_Z:

                direct = np.exp(
                    -(R * (xi - Z) - T) ** 2 / scale
                )

                image = np.exp(P * Z) * np.exp(
                    -(R * (xi + Z) - T) ** 2 / scale
                )

                kernel = prefactor * (direct - image)

                integral = np.trapezoid(kernel, xi)

                print(
                    f"T={T:7.3g}, "
                    f"Z={Z:4.2f} | "
                    f"direct={direct.max():.3e} | "
                    f"image={image.max():.3e} | "
                    f"kernel={kernel.min():.3e} .. {kernel.max():.3e} | "
                    f"integral={integral:.3e}"
                )


    diagnose_ivp_kernel()


    def test_direct_term_only():

        import numpy as np

        L = 60.0
        theta = 0.34
        infiltration = 1.5
        dispersivity = 1.5

        v = infiltration / theta
        D = v * dispersivity
        P = v * L / D

        R = 1.0

        z = np.arange(0.0, L + 0.5, 0.5)
        xi = z / L

        print("=" * 70)
        print("DIRECT TERM ONLY")
        print("=" * 70)

        for T in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:

            scale = 4 * R * T / P

            prefactor = np.sqrt(
                R * P / (4 * np.pi * T)
            )

            for Z in [0.0, 0.25, 0.5, 0.75, 1.0]:

                direct = np.exp(
                    -(R * (xi - Z) - T) ** 2 / scale
                )

                kernel = prefactor * direct

                integral = np.trapezoid(kernel, xi)

                print(
                    f"T={T:6.3f}, "
                    f"Z={Z:.2f} | "
                    f"kernel={kernel.min():.3e} .. "
                    f"{kernel.max():.3e} | "
                    f"integral={integral:.6e}"
                )


    test_direct_term_only()
    return


if __name__ == "__main__":
    app.run()
