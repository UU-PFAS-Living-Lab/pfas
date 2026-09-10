import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing Solid-Phase Sorption Models for PFOA in Accusand

    This example demonstrates how to run simulations using three different solid-phase
    sorption parameterisation approaches:

    1. **Linear Kd** — direct input of a distribution coefficient derived from K_oc and K_sc
    2. **Freundlich** — non-linear isotherm with capacity coefficient *K* and exponent *n*
    3. **Fabregat-Palau (2021)** — Kd estimated from molecular structure (number of CF₂
       groups) and soil composition (f_oc, f_silt_clay)

    All three runs use identical hydraulic, grid, and boundary settings so that
    differences in the output reflect only the sorption parameterisation.
    PFAS and soil properties are loaded from the bundled JSON datasets via
    `pfas.data_loader.load_dataset`.
    """)
    return


@app.cell
def _():
    from pfas.model import Model
    from pfas.component import LinearSPsorption, SWCsorption, Retardation, EquilibriumSolver, WaterPreprocessor, BoundaryPreprocessor, GridGenerator, FreundlichSPsorption

    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo


    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        FreundlichSPsorption,
        GridGenerator,
        LinearSPsorption,
        Model,
        Retardation,
        SWCsorption,
        WaterPreprocessor,
        load_dataset,
        mo,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — Load PFAS and soil data with `load_dataset`

    All parameters are read from the packaged JSON datasets bundled with the `pfas`
    library using `load_dataset("PFASs")` and `load_dataset("soils")`.
    No values are hardcoded — the cell below loads the full databases and extracts
    the entries for **PFOA** and **Accusand**.
    We manually increase the level of OM to 10% in order to see the effects on solid phase sorption.
    """)
    return


@app.cell
def _(load_dataset):
    # ── Load datasets from the pfas library ───────────────────────────────────
    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")

    # ── PFAS: PFOA ────────────────────────────────────────────────────────────
    pfas_name = "PFOA"
    pfas      = pfas_db[pfas_name]
    n_CFx     = pfas["structural_properties"]["n_CFx"]
    K_oc_pfoa = pfas["K_oc"]["value"]   # L/kg
    K_sc_pfoa = pfas["K_sc"]["value"]   # L/kg

    # ── Soil: Accusand ────────────────────────────────────────────────────────
    soil_name    = "Accusand"
    soil         = soil_db[soil_name]
    bulk_dens    = soil["rho_b"]["value"]
    porosity     = soil["porosity"]
    theta_r      = soil["theta_r"]
    theta_s      = soil["theta_s"]
    K_sat        = soil["K_sat"]["value"]
    vg_alpha     = soil["van_genuchten"]["alpha"]["value"]
    vg_n         = soil["van_genuchten"]["n"]
    vg_l         = soil["van_genuchten"]["l"]
    dispersivity = 3  # cm — not listed for Accusand; typical literature value

    # Soil composition: stored as percent in the database → convert to fractions
    f_oc        = soil["f_oc"]["value"] / 100
    f_clay      = soil["f_clay"]["value"] / 100
    f_silt      = soil["f_silt"]["value"] / 100
    f_silt_clay = f_silt + f_clay

    # We change organic content to see more difference
    f_oc = 10 / 100

    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc_pfoa} L/kg  |  K_sc = {K_sc_pfoa} L/kg")
    print(f"Soil : {soil_name}  |  ρ_b = {bulk_dens} g/cm³  |  porosity = {porosity}")
    print(f"       θ_r = {theta_r}  |  K_sat = {K_sat} cm/s  |  vg_n = {vg_n}")
    print(f"       f_oc = {f_oc:.5f}  |  f_silt_clay = {f_silt_clay:.4f}")
    return (
        K_oc_pfoa,
        K_sat,
        K_sc_pfoa,
        bulk_dens,
        dispersivity,
        f_oc,
        f_silt_clay,
        n_CFx,
        porosity,
        theta_r,
        vg_alpha,
        vg_l,
        vg_n,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Sorption parameters per method

    Each `sorption_solid` dict carries its method's *native* parameters —
    Kd itself is never computed here by hand. The one exception is the
    "Linear Kd" branch: its Kd comes from a tabulated K_oc/K_sc formula that
    isn't one of `LinearSPsorption`'s built-in `Kd_method` options
    (`direct_input`, `fabregat_palau`), so it's computed once and passed in
    as a `direct_input` value — everything else resolves inside `compute()`.

    | Model | Class | `Kd_method` / isotherm |
    |---|---|---|
    | **Linear Kd** | `LinearSPsorption` | `"linear"` / `Kd_method: "direct_input"` |
    | **Freundlich** | `FreundlichSPsorption` | `"freundlich"` |
    | **Fabregat-Palau** | `LinearSPsorption` | `"linear"` / `Kd_method: "fabregat_palau"` |

    In this code section we prefer the input for the model.
    """)
    return


@app.cell
def _(K_oc_pfoa, K_sc_pfoa, f_oc, f_silt_clay, n_CFx):
    Kd_linear_direct = K_oc_pfoa * f_oc + K_sc_pfoa * f_silt_clay

    K_freund = K_oc_pfoa * f_oc
    n_freund = 0.8
    C_rep    = 0.5

    sorption_linear = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "linear": {"Kd_method": "direct_input", "Kd": Kd_linear_direct},
    }

    sorption_freundlich = {
        "kinetic_sorption": False,
        "sorption_isotherm": "freundlich",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "freundlich": {
            "K_freund": K_freund,
            "n_freund": n_freund,
            "C_rep": C_rep,
        },
    }

    sorption_fabregat = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "linear": {
            "Kd_method": "fabregat_palau",
            "n_CFx": n_CFx,
            "f_oc": f_oc,
            "f_silt_clay": f_silt_clay,
        },
    }

    print(f"Linear Kd (tabulated, injected as direct_input) = {Kd_linear_direct:.4f} L/kg")
    return sorption_fabregat, sorption_freundlich, sorption_linear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Run one simulation per sorption model
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    EquilibriumSolver,
    FreundlichSPsorption,
    GridGenerator,
    K_sat,
    LinearSPsorption,
    Model,
    Retardation,
    SWCsorption,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    porosity,
    sorption_fabregat,
    sorption_freundlich,
    sorption_linear,
    theta_r,
    vg_alpha,
    vg_l,
    vg_n,
):

    def build_base_model():
        m = Model()

        m.compute(
            GridGenerator,
            domain_length=60,
            spatial_resolution=1.0,
            time_resolution=1000,
            time_total=40000,
        )

        m.compute(
            WaterPreprocessor,
            average_infiltration_rate=1.5e-2,
            hydraulic_conductivity=K_sat,
            porosity=porosity,
            dispersivity=dispersivity,
            van_genuchten_n=vg_n,
            van_genuchten_l=vg_l,
            residual_water_content=theta_r,
        )

        m.compute(
            BoundaryPreprocessor,
            C_list=[0.1, 0],
            T_list=[0, 2000],
        )

        m.compute(
            SWCsorption,
            sigma0=71,
            scaling_factor_awi=1.0,
            van_genuchten_alpha=vg_alpha,
        )

        return m

    # ── Run one simulation per sorption model ──────────────────────────────────
    sorption_methods = {
        "Linear Kd":      (LinearSPsorption, sorption_linear),
        "Freundlich":     (FreundlichSPsorption, sorption_freundlich),
        "Fabregat-Palau": (LinearSPsorption, sorption_fabregat),
    }

    branch_models = {}

    print("Running simulations ...")
    def run_branch(label, sorption_cls, sorption_solid):
        branch_model = build_base_model()
        branch_model.compute(sorption_cls, sorption_solid=sorption_solid)
        branch_model.compute(Retardation, Kaw=0.5, bulk_density=bulk_dens)
        branch_model.compute(EquilibriumSolver)
        print(f"  {label}: done  (Kd = {branch_model.Kd:.4f} L/kg)")
        return branch_model

    branch_models = {
        label: run_branch(label, sorption_cls, sorption_solid)
        for label, (sorption_cls, sorption_solid) in sorption_methods.items()
    }
    model_linear   = branch_models["Linear Kd"]
    model_freund   = branch_models["Freundlich"]
    model_fabregat = branch_models["Fabregat-Palau"]
    print("\nAll three simulations completed successfully!")
    return model_fabregat, model_freund, model_linear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Visualise and compare results

    The three panels below show:
    - **Left** — depth profiles of total PFAS concentration at t ≈ 2000 s
    - **Middle** — breakthrough curves at a fixed depth (30 cm)
    - **Right** — the Kd values each model actually resolved (read from
      `model.Kd`, not precomputed)
    """)
    return


@app.cell
def _(model_fabregat, model_freund, model_linear, plt):
    sim_grid = model_linear.grid

    t_idx_2000 = min(range(len(sim_grid.time)), key=lambda i: abs(sim_grid.time[i] - 10000))
    actual_time = sim_grid.time[t_idx_2000]

    branches = [
        ("Linear Kd",      model_linear,   "#1f77b4"),
        ("Freundlich",     model_freund,   "#ff7f0e"),
        ("Fabregat-Palau", model_fabregat, "#2ca02c"),
    ]

    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(16, 6))
    fig_cmp.suptitle(
        "Solid-phase sorption comparison — PFOA in Accusand + 10% OM\n"
        + " | ".join(f"{label} Kd = {m.Kd:.4f}" for label, m, _ in branches)
        + "  (all in L/kg)",
        fontsize=11,
    )

    ax1 = axes_cmp[0]
    for label, m, col in branches:
        ax1.plot(m.C1[:, t_idx_2000], sim_grid.depth, color=col, label=label, linewidth=2)
    ax1.set_xlabel("Total PFAS Concentration (mg/L)")
    ax1.set_ylabel("Depth (cm)")
    ax1.set_title(f"Depth profiles at t = {actual_time:.0f} s")
    ax1.invert_yaxis()
    ax1.legend(title="Sorption model")
    ax1.grid(True, alpha=0.3)

    ax2 = axes_cmp[1]
    depth_target_cm = 30
    depth_idx = min(range(len(sim_grid.depth)), key=lambda i: abs(sim_grid.depth[i] - depth_target_cm))
    actual_depth = sim_grid.depth[depth_idx]

    for label, m, col in branches:
        ax2.plot(sim_grid.time, m.C1[depth_idx, :], color=col, label=label, linewidth=2)

    ax2.axvline(x=actual_time, color="gray", linestyle="--", linewidth=1, label=f"t = {actual_time:.0f} s")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Total PFAS Concentration (mg/L)")
    ax2.set_title(f"Breakthrough curves\nat depth ≈ {actual_depth:.0f} cm")
    ax2.legend(title="Sorption model")
    ax2.grid(True, alpha=0.3)

    ax3 = axes_cmp[2]
    model_names = ["Linear Kd", "Freundlich\n(at 0.5 mg/L)", "Fabregat-\nPalau"]
    kd_values   = [m.Kd for _, m, _ in branches]
    bar_colours = [col for _, _, col in branches]

    bars = ax3.bar(model_names, kd_values, color=bar_colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, kd_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax3.set_ylabel("Kd (L/kg)")
    ax3.set_title("Distribution coefficients resolved per model")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0, max(kd_values) * 1.2)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    | Model | Kd derivation | Key assumption |
    |---|---|---|
    | **Linear Kd** | Direct input from tabulated K_oc and K_sc | Sorption is linear and proportional to soil organic carbon and mineral surface area |
    | **Freundlich** | Non-linear isotherm fitted to experimental data | Sorption efficiency decreases with increasing concentration (n < 1) |
    | **Fabregat-Palau** | Empirical regression on n_CFx, f_oc, f_silt_clay | Kd depends only on molecular chain length and soil composition |

    Differences in the depth profiles and breakthrough curves arise solely from the
    different Kd values. A higher Kd retards transport more strongly, shifting
    breakthrough to later times and keeping the plume closer to the surface.
    """)
    return


if __name__ == "__main__":
    app.run()
