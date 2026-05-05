import marimo

__generated_with = "0.23.1"
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
    from pfas.preprocessing import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        SWCAdsorptionPreprocessor,
        SorptionKawiDirectInput,
        SimulationRunner,
    )
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        GridGenerator,
        SWCAdsorptionPreprocessor,
        SimulationRunner,
        SorptionKawiDirectInput,
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

    | Property | Value | Dataset key |
    |---|---|---|
    | PFAS | PFOA | `PFASs["PFOA"]` |
    | n_CFx | 7 | `pfas["n_CFx"]` |
    | K_oc (PFOA) | 107 L/kg | `pfas["K_oc"]["value"]` |
    | K_sc (PFOA) | 3.3 L/kg | `pfas["K_sc"]["value"]` |
    | Soil | Accusand | `soils["Accusand"]` |
    | Bulk density | 1.65 g/cm³ | `soil["rho_b"]["value"]` |
    | Porosity | 0.294 | `soil["porosity"]` |
    | θ_r | 0.015 | `soil["theta_r"]` |
    | K_sat | 0.020964 cm/s | `soil["K_sat"]["value"]` |
    | van Genuchten α | 0.04479 cm⁻¹ | `soil["van_genuchten"]["alpha"]["value"]` |
    | van Genuchten n | 4.0 | `soil["van_genuchten"]["n"]` |
    | f_oc | 0.04 % | `soil["f_oc"]["value"]` |
    | f_clay | 0 % | `soil["f_clay"]["value"]` |
    | f_silt | 0 % | `soil["f_silt"]["value"]` |
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
    n_CFx     = pfas["n_CFx"]
    K_oc_pfoa = pfas["K_oc"]["value"]   # L/kg
    K_sc_pfoa = pfas["K_sc"]["value"]   # L/kg

    # ── Soil: Accusand ────────────────────────────────────────────────────────
    soil_name   = "Accusand"
    soil        = soil_db[soil_name]
    bulk_dens   = soil["rho_b"]["value"]
    porosity    = soil["porosity"]
    theta_r     = soil["theta_r"]
    theta_s     = soil["theta_s"]
    K_sat       = soil["K_sat"]["value"]
    vg_alpha    = soil["van_genuchten"]["alpha"]["value"]
    vg_n        = soil["van_genuchten"]["n"]
    dispersivity = 3 # cm — not listed for Accusand; typical literature value

    # Soil composition: stored as percent in the database → convert to fractions
    f_oc        = soil["f_oc"]["value"] / 100
    f_clay      = soil["f_clay"]["value"] / 100
    f_silt      = soil["f_silt"]["value"] / 100
    f_silt_clay = f_silt + f_clay

    #we change organic content to see more difference
    f_oc = 10/100

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
        theta_s,
        vg_alpha,
        vg_n,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Define sorption parameters for each model

    Instead of pre-computing Kd values manually, each method passes its own
    native parameters into `SpRetardationPreprocessor`, which resolves the
    effective Kd internally:

    | Model | `sorption_isotherm` | Kd resolution in `SpRetardationPreprocessor` |
    |---|---|---|
    | **Linear** | `"linear"` / `Kd_method: "direct_input"` | Uses `Kd` directly |
    | **Freundlich** | `"freundlich"` / `Kd_method: "freundlich"` | Calls `kd_freundlich(C_rep, K_freund, n_freund)` |
    | **Fabregat-Palau** | `"linear"` / `Kd_method: "fabregat_palau"` | Calls `kd_fabregat_palau(n_CFx, f_oc, f_silt_clay)` |

    The Freundlich effective Kd is evaluated at a representative concentration
    `C_rep = 1.0 mg/L` with exponent *n* = 0.8 (literature value for PFAS,
    n < 1 → favourable/concave isotherm).
    """)
    return


@app.cell
def _(K_oc_pfoa, K_sc_pfoa, f_oc, f_silt_clay, n_CFx):
    # ── 1. Linear Kd — direct input ───────────────────────────────────────────
    # Kd = K_oc * f_oc + K_sc * f_silt_clay  (uses tabulated PFOA values)
    from pfas.utils import kd_fabregat_palau, kd_freundlich

    # 1. Linear Kd: Kd = K_oc * f_oc + K_sc * f_silt_clay
    Kd_linear = K_oc_pfoa * f_oc + K_sc_pfoa * f_silt_clay

    # 2. Freundlich: effective Kd at a representative concentration
    #    n_freund < 1 -> favourable (concave) isotherm, typical for PFAS
    K_freund    = K_oc_pfoa * f_oc
    n_freund    = 0.8
    C_rep       = 0.5
    Kd_freund   = kd_freundlich(C_rep, K_freund, n_freund)

    # 3. Fabregat-Palau (2021): Kd from molecular structure + soil composition
    Kd_fabregat = kd_fabregat_palau(n_CFx, f_oc, f_silt_clay)

    print(f"Linear Kd      = {Kd_linear:.4f} L/kg")
    print(f"Freundlich Kd  = {Kd_freund:.4f} L/kg  (C_rep={C_rep} mg/L, n={n_freund})")
    print(f"Fabregat-Palau = {Kd_fabregat:.4f} L/kg")
    return C_rep, K_freund, Kd_fabregat, Kd_freund, Kd_linear, n_freund


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — Shared preprocessing (grid, water, boundary, AWI)

    These objects are computed once and reused in all three simulation runs.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    GridGenerator,
    K_sat,
    SWCAdsorptionPreprocessor,
    SorptionKawiDirectInput,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    porosity,
    theta_r,
    theta_s,
    vg_alpha,
    vg_n,
):
    # Grid
    grid_gen = GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=0.1,
        time_total=10000,
    )
    grid_results = grid_gen.compute()

    # Water flow
    water_prep = WaterPreprocessor(
        average_infiltration_rate=1.5/1000,
        hydraulic_conductivity=K_sat,
        porosity=porosity,
        dispersivity=dispersivity,
        van_genuchten_n=vg_n,
        van_genuchten_l=0.5,
        init_sat=0.2,
        residual_water_content=theta_r,
    )
    water_results = water_prep.compute()

    # Boundary conditions — 10 mg/L PFOA pulse for the first 2000 s
    boundary_prep = BoundaryPreprocessor(
        C_list=[0.1, 0],
        T_list=[0, 2000]
    )
    boundary_results = boundary_prep.compute()

    # AWI adsorption (SWC-based)
    swc_adsorp = SWCAdsorptionPreprocessor(
        hydro_properties=water_results["hydro_properties"],
        sigma0=71,
        scaling_factor_awi=1.0,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {"scaling_factor_awi": 1.0},
        },
        soil={
            "bulk_density": bulk_dens,
            "porosity": porosity,
            "van_genuchten_alpha": vg_alpha,
            "van_genuchten_n": vg_n,
            "saturated_water_content": theta_s,
            "residual_water_content": theta_r,
            "hydraulic_conductivity": K_sat,
            "dispersivity": dispersivity,
        },
    )
    awi_results = swc_adsorp.compute()

    # Kawi sorption (air–water interface)
    kawi_sorp = SorptionKawiDirectInput(
        kaw=0.5,
        hydro_properties=water_results["hydro_properties"],
        aaw=awi_results["aaw"],
    )
    kawi_results = kawi_sorp.compute()

    print("Shared preprocessing complete.")
    return boundary_results, grid_results, kawi_results, water_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Run one simulation per sorption model

    The `sorption_solid` dict is rebuilt for each model; everything else is identical.
    """)
    return


@app.cell
def _(
    C_rep,
    K_freund,
    Kd_fabregat,
    Kd_linear,
    SimulationRunner,
    boundary_results,
    bulk_dens,
    grid_results,
    kawi_results,
    n_freund,
    water_results,
):
    # All three sorption_solid dicts use "direct_input" with the pre-resolved Kd.
    # SpRetardationPreprocessor receives the correct Kd for each method directly,
    # so no back-dependency on the plotting cell is created.

    sorption_linear = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "linear": {"Kd_method": "direct_input", "Kd": Kd_linear},
    }

    sorption_freundlich = {
        "kinetic_sorption": False,
        "sorption_isotherm": "freundlich",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "freundlich": {
            "K_freund": K_freund,
            "n_freund": n_freund,
            "C_rep":    C_rep,
        },
    }

    sorption_fabregat = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "linear": {"Kd_method": "direct_input", "Kd": Kd_fabregat},
    }

    def run_simulation(sorption_solid, label):
        result = SimulationRunner(
            grid=grid_results["grid"],
            bulk_density=bulk_dens,
            boundary_conditions=boundary_results["boundary_conditions"],
            hydro_properties=water_results["hydro_properties"],
            awi_retardation=kawi_results["awi_retardation"],
            sorption_solid=sorption_solid,
            kinetic_sorption=True,
            volume_averaged=True,
        ).compute()
        print(f"  {label}: done")
        return result

    print("Running simulations ...")
    results_linear   = run_simulation(sorption_linear,    "Linear Kd     ")
    results_freund   = run_simulation(sorption_freundlich, "Freundlich    ")
    results_fabregat = run_simulation(sorption_fabregat,  "Fabregat-Palau")
    print("\nAll three simulations completed successfully!")
    return results_fabregat, results_freund, results_linear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Visualise and compare results

    The three panels below show:
    - **Left** — depth profiles of total PFAS concentration at five time snapshots
    - **Middle** — breakthrough curves at a fixed depth (30 cm)
    - **Right** — the Kd values used by each model (for reference)
    """)
    return


@app.cell
def _(
    Kd_fabregat,
    Kd_freund,
    Kd_linear,
    grid_results,
    plt,
    results_fabregat,
    results_freund,
    results_linear,
):
    sim_grid = grid_results["grid"]

    # Find the index closest to t = 2000 s
    t_idx_2000 = min(range(len(sim_grid.time)), key=lambda i: abs(sim_grid.time[i] - 2500))
    actual_time = sim_grid.time[t_idx_2000]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"Solid-phase sorption comparison — PFOA in Accusand + 10% OM\n"
        f"Linear Kd = {Kd_linear:.4f} | Freundlich Kd = {Kd_freund:.4f} | "
        f"Fabregat-Palau Kd = {Kd_fabregat:.4f}  (all in L/kg)",
        fontsize=11,
    )

    # Panel 1: depth profile at t = 2000 s
    ax1 = axes[0]
    for label, res, col in [
        ("Linear Kd",      results_linear,   "#1f77b4"),
        ("Freundlich",     results_freund,   "#ff7f0e"),
        ("Fabregat-Palau", results_fabregat, "#2ca02c"),
    ]:
        ax1.plot(
            res["C_tot"][:, t_idx_2000],
            sim_grid.depth,
            color=col,
            label=label,
            linewidth=2,
        )

    ax1.set_xlabel("Total PFAS Concentration (mg/L)")
    ax1.set_ylabel("Depth (cm)")
    ax1.set_title(f"Depth profiles at t = {actual_time:.0f} s")
    ax1.invert_yaxis()
    ax1.legend(title="Sorption model")
    ax1.grid(True, alpha=0.3)

    # Panel 2: breakthrough curves at 30 cm
    ax2 = axes[1]
    depth_target_cm = 30
    depth_idx = min(
        range(len(sim_grid.depth)),
        key=lambda i: abs(sim_grid.depth[i] - depth_target_cm),
    )
    actual_depth = sim_grid.depth[depth_idx]

    for label, res, col in [
        ("Linear Kd",      results_linear,   "#1f77b4"),
        ("Freundlich",     results_freund,   "#ff7f0e"),
        ("Fabregat-Palau", results_fabregat, "#2ca02c"),
    ]:
        ax2.plot(
            sim_grid.time,
            res["C_tot"][depth_idx, :],
            color=col,
            label=label,
            linewidth=2,
        )

    ax2.axvline(x=actual_time, color="gray", linestyle="--", linewidth=1, label=f"t = {actual_time:.0f} s")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Total PFAS Concentration (mg/L)")
    ax2.set_title(f"Breakthrough curves\nat depth ≈ {actual_depth:.0f} cm")
    ax2.legend(title="Sorption model")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Kd bar chart
    ax3 = axes[2]
    model_names = ["Linear Kd", "Freundlich\n(at 0.001 mg/L)", "Fabregat-\nPalau"]
    kd_values   = [Kd_linear, Kd_freund, Kd_fabregat]
    bar_colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

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
    ax3.set_title("Distribution coefficients\nresolved by SpRetardationPreprocessor")
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
    different Kd values.  A higher Kd retards transport more strongly, shifting
    breakthrough to later times and keeping the plume closer to the surface.

    > **Note:** Accusand has very low f_oc (0.04 %) and no silt/clay, so all three
    > methods yield small Kd values and relatively fast PFOA transport — consistent
    > with the sand being used in column experiments precisely because it has minimal
    > sorption capacity.
    """)
    return


if __name__ == "__main__":
    app.run()
