import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing Air-Water Interfacial Area Estimation Methods for PFOA in Staring-O05

    This example demonstrates how to run simulations using three different Aaw estimation approaches:

    1. **SWC-based** — Compute air-water interfacial area using thermodynamic approach
    2. **Func1** — Compute air-water interfacial area using the GSSA-based linear model.
    3. **Func2** — Compute air-water interfacial area using the d50 correlation

    NOG VERDER UPDATEN!!! _All three runs use identical hydraulic, grid, and boundary settings so that
    differences in the output reflect only the sorption parameterisation.
    PFAS and soil properties are loaded from the bundled JSON datasets via
    `pfas.data_loader.load_dataset`._
    """)
    return


@app.cell
def _():
    from pfas.preprocessing import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        SpRetardationPreprocessor,
        SWCAdsorptionPreprocessor,
        SorptionKawiDirectInput,
        SorptionKawCalculated,
        SimulationRunner,
    )
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        GridGenerator,
        SimulationRunner,
        SorptionKawCalculated,
        SpRetardationPreprocessor,
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
    pfas_db = load_dataset("PFASs_neutral_forms")
    soil_db = load_dataset("soils_Ksat_rho_b_d50")

    # ── PFAS: PFOA ────────────────────────────────────────────────────────────
    pfas_name = "PFOA"
    pfas      = pfas_db[pfas_name]
    n_CFx     = pfas["n_CFx"]
    K_oc      = pfas["K_oc"]["value"]   # L/kg
    K_sc      = pfas["K_sc"]["value"]   # L/kg

    # ── Soil: Staring-O## ────────────────────────────────────────────────────────
    soil_name   = "Staring-O18"
    soil        = soil_db[soil_name]
    bulk_dens   = soil["rho_b"]["value"]
    porosity    = soil["porosity"]
    theta_r     = soil["theta_r"]
    theta_s     = soil["theta_s"]
    K_sat       = soil["K_sat"]["value"]
    vg_alpha    = soil["van_genuchten"]["alpha"]["value"]
    vg_n        = soil["van_genuchten"]["n"]
    vg_l        = soil["van_genuchten"]["l"]
    dispersivity = 4.5 # cm — not listed for Staring-O05; typical literature value

    # Solid-phase adsorption: linear
    frac_int = 1.0
    rate_const = 0.0

    # Soil composition: stored as percent in the database → convert to fractions
    f_oc        = soil["f_oc"]["value"] / 100
    f_clay      = soil["f_clay"]["value"] / 100
    f_silt      = soil["f_silt"]["value"] / 100
    f_silt_clay = f_silt + f_clay


    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc} L/kg  |  K_sc = {K_sc} L/kg")
    print(f"Soil : {soil_name}  |  ρ_b = {bulk_dens} g/cm³  |  porosity = {porosity}")
    print(f"       θ_r = {theta_r}  |  K_sat = {K_sat} cm/s  |  vg_n = {vg_n}  |  vg_l = {vg_l}")
    print(f"       f_oc = {f_oc:.5f}  |  f_silt_clay = {f_silt_clay:.4f}")
    return (
        K_sat,
        bulk_dens,
        dispersivity,
        f_oc,
        f_silt_clay,
        frac_int,
        n_CFx,
        pfas,
        pfas_name,
        porosity,
        rate_const,
        soil,
        soil_name,
        theta_r,
        theta_s,
        vg_alpha,
        vg_l,
        vg_n,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Shared preprocessing (grid, water, boundary, Sp-sorption)

    These objects are computed once and reused in all three simulation runs.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    GridGenerator,
    K_sat,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    f_oc,
    f_silt_clay,
    frac_int,
    n_CFx,
    pfas,
    porosity,
    rate_const,
    theta_r,
    vg_l,
    vg_n,
):
    from pfas.utils import kd_fabregat_palau
    # Grid
    grid_gen = GridGenerator(
        domain_length=100,
        spatial_resolution=1.0,
        time_resolution=(60*60*24*365),
        time_total=250*(60*60*24*365),
    )
    grid_results = grid_gen.compute()

    # Water flow
    water_prep = WaterPreprocessor(
        average_infiltration_rate=9.51E-7,
        hydraulic_conductivity=K_sat,
        porosity=porosity,
        dispersivity=dispersivity,
        van_genuchten_n=vg_n,
        van_genuchten_l=vg_l,
        init_sat=0.2,
        residual_water_content=theta_r,
    )
    water_results = water_prep.compute()

    pulse_duration = 25 * (60 * 60 * 24 * 365)
    # Step 3: Setup boundary conditions
    boundary_prep = BoundaryPreprocessor(
        C_list=[pfas['M']["value"] * 1e-9, 0.0], # mg/L for 1 pmol/L
        T_list=[0.0, pulse_duration]
    )
    boundary_results = boundary_prep.compute()

    # Solid phase adsorption (linear)
    sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "kinetic": {
                "frac_int": frac_int,
                "rate_const": rate_const,
            },
            "linear": {
                "Kd_method": "direct_input",
                "Kd": kd_fabregat_palau(n_CFx, f_oc, f_silt_clay)
            },
    }

    sp_retard = SpRetardationPreprocessor(
        sorption_solid= sorption_solid,
        bulk_density=bulk_dens,
        hydro_properties=water_results["hydro_properties"],
    )
    sp_results = sp_retard.compute()

    print("Shared preprocessing complete.")
    print('effective saturation is',(((water_results['hydro_properties'].water_content)-theta_r)/(porosity-theta_r)),'(-)')
    return (
        boundary_results,
        grid_results,
        sorption_solid,
        sp_results,
        water_results,
    )


@app.cell
def _(sp_results):
    print(sp_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — Define sorption parameters for each model

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
def _(porosity, soil, theta_r, theta_s, vg_alpha, vg_n, water_results):
    from pfas.utils import aaw_func_thermo, aaw_func_1, aaw_func_2

    theta = water_results["hydro_properties"].water_content
    d50 = soil["d50"]["value"]/10000 #convert um to cm

    # 1. SWC-based
    func_thermo_Aaw = aaw_func_thermo(
        sigma0=71,
        poro=porosity,
        alpha=vg_alpha,
        n=vg_n,
        th=theta,
        thr=theta_r,
        ths=theta_s,
        sf=4.15
    )

    # 2. Func1
    func_1_Aaw = aaw_func_1(
        th=theta,
        ths=theta_s,
        poro=porosity,
        d50=d50
    )

    # 3. Func2
    func_2_Aaw = aaw_func_2(
        th=theta,
        ths=theta_s,
        d50=d50
    )

    print(f"Thermodynamic   = {func_thermo_Aaw:.4f} (cm²/cm³)")
    print(f"GSSA-based Aaw  = {func_1_Aaw:.4f} (cm²/cm³)")
    print(f"d50-based Aaw   = {func_2_Aaw:.4f} (cm²/cm³)")
    print(d50)
    return func_1_Aaw, func_2_Aaw, func_thermo_Aaw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Run one simulation per sorption model

    The `sorption_solid` dict is rebuilt for each model; everything else is identical.
    """)
    return


@app.cell
def _(
    SimulationRunner,
    SorptionKawCalculated,
    boundary_results,
    bulk_dens,
    func_1_Aaw,
    func_2_Aaw,
    func_thermo_Aaw,
    grid_results,
    pfas,
    sorption_solid,
    water_results,
):
    # All three sorption_solid dicts use "direct_input" with the pre-resolved Kd.
    # SpRetardationPreprocessor receives the correct Kd for each method directly,
    # so no back-dependency on the plotting cell is created.

    kaw_input = {
        "n_CFx"             : pfas["n_CFx"],
        "n_CHx"             : pfas["n_CHx"],
        "n_COO"             : pfas["n_COO"],
        "n_COOH"            : pfas["n_COOH"],
        "n_SO3"             : pfas["n_SO3"],
        "n_R4N"             : pfas["n_R4N"],
        "n_OH"              : pfas["n_OH"],
        "n_OSO3"            : pfas["n_OSO3"],
        "n__O_"             : pfas["n__O_"],
        "n__S_"             : pfas["n__S_"],
        "n_N_CH3_2_CH2_COO" : pfas["n_N_CH3_2_CH2_COO"],
    }

    # Kawi sorption (air–water interface)
    kawi_thermo = SorptionKawCalculated(
        **kaw_input,
        hydro_properties=water_results["hydro_properties"],
        aaw=func_thermo_Aaw,
    )
    kawi_results_thermo = kawi_thermo.compute()


    kawi_func1 = SorptionKawCalculated(
        **kaw_input,
        hydro_properties=water_results["hydro_properties"],
        aaw=func_1_Aaw,
    )
    kawi_results_func1 = kawi_func1.compute()


    kawi_func2 = SorptionKawCalculated(
        **kaw_input,
        hydro_properties=water_results["hydro_properties"],
        aaw=func_2_Aaw,
    )
    kawi_results_func2 = kawi_func2.compute()


    def run_simulation(kawi_results, label):
        result = SimulationRunner(
            grid=grid_results["grid"],
            bulk_density=bulk_dens,
            boundary_conditions=boundary_results["boundary_conditions"],
            hydro_properties=water_results["hydro_properties"],
            awi_retardation=kawi_results["awi_retardation"],
            sorption_solid=sorption_solid,
            kinetic_sorption=False,
            volume_averaged=True,
        ).compute()
        print(f"  {label}: done")
        return result

    print("Running simulations ...")
    results_thermo = run_simulation(kawi_results_thermo, "Thermodynamic")
    results_func_1 = run_simulation(kawi_results_func1,  "GSSA-based Aaw")
    results_func_2 = run_simulation(kawi_results_func2,  "d50-based Aaw")
    print("\nAll three simulations completed successfully!")
    return (
        kawi_results_func1,
        kawi_results_func2,
        kawi_results_thermo,
        results_func_1,
        results_func_2,
        results_thermo,
    )


@app.cell
def _(kawi_results_func1, kawi_results_func2, kawi_results_thermo):
    print(kawi_results_thermo, kawi_results_func1, kawi_results_func2)
    return


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
    func_1_Aaw,
    func_2_Aaw,
    func_thermo_Aaw,
    grid_results,
    kawi_results_func1,
    kawi_results_func2,
    kawi_results_thermo,
    pfas_name,
    plt,
    results_func_1,
    results_func_2,
    results_thermo,
    soil_name,
):
    sim_grid = grid_results["grid"]
    seconds_per_year = 60*60*24*365
    target_time_years = 50
    target_time = target_time_years * seconds_per_year

    # Find the index closest
    t_idx = min(range(len(sim_grid.time)), key=lambda i: abs(sim_grid.time[i] - target_time))
    actual_time = sim_grid.time[t_idx]
    actual_time_years = actual_time / seconds_per_year

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"Air-water interfacial sorption comparison - {pfas_name} in {soil_name}\n"
        f"Thermodynamic = {kawi_results_thermo['awi_retardation']:.4f} | "
        f"Function 1 = {kawi_results_func1['awi_retardation']:.4f} | "
        f"Function 2 = {kawi_results_func2['awi_retardation']:.4f}",
        fontsize=11,
    )

    # Panel 1: depth profile at selected time
    ax1 = axes[0]
    for label, res, col in [
        ("Thermodynamic", results_thermo, "#1f77b4"),
        ("Function 1",    results_func_1,  "#ff7f0e"),
        ("Function 2",    results_func_2,  "#2ca02c"),
    ]:
        ax1.plot(
            res["C_tot"][:, t_idx],
            sim_grid.depth,
            color=col,
            label=label,
            linewidth=2,
        )

    ax1.set_xlabel("Total PFAS Concentration (mg/L)")
    ax1.set_ylabel("Depth (cm)")
    ax1.set_title(f"Depth profiles at t = {actual_time_years:.0f} years")
    ax1.invert_yaxis()
    ax1.legend(title="Aaw estimation method")
    ax1.grid(True, alpha=0.3)

    # Panel 2: breakthrough curves at 50 cm
    ax2 = axes[1]
    depth_target_cm = 50
    depth_idx = min(
        range(len(sim_grid.depth)),
        key=lambda i: abs(sim_grid.depth[i] - depth_target_cm),
    )
    actual_depth = sim_grid.depth[depth_idx]

    for label, res, col in [
        ("Thermodynamic", results_thermo, "#1f77b4"),
        ("Function 1",    results_func_1,  "#ff7f0e"),
        ("Function 2",    results_func_2,  "#2ca02c"),
    ]:
        ax2.plot(
            sim_grid.time/seconds_per_year,
            res["C_tot"][depth_idx, :],
            color=col,
            label=label,
            linewidth=2,
        )

    ax2.axvline(x=actual_time_years, color="gray", linestyle="--", linewidth=1, label=f"t = {actual_time_years:.1f} yr")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Total PFAS Concentration (mg/L)")
    ax2.set_title(f"Breakthrough curves\nat depth ≈ {actual_depth:.0f} cm")
    ax2.legend(title="Aaw estimation method")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Kd bar chart
    ax3 = axes[2]
    model_names = ["Thermodynamic", "Function 1", "Function 2"]
    Aaw_values   = [func_thermo_Aaw, func_1_Aaw, func_2_Aaw]
    bar_colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax3.bar(model_names, Aaw_values, color=bar_colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, Aaw_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax3.set_ylabel("Aaw (cm²/cm³)")
    ax3.set_title("Distribution air-water interfacial area values\nresolved by SWCAdsorptionPreprocessor")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0, max(Aaw_values) * 1.2)

    plt.tight_layout()
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving images
    """)
    return


@app.cell
def _(fig, pfas_name, soil_name):
    import pathlib

    output_dir = pathlib.Path(
        r"C:\Users\youri\Documents\ESW\Thesis_v2\04_Processing\3_Aaw\output_marimo_Aaw_multiple_methods\neutral_acids"
    ) / pfas_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / f"{pfas_name}_{soil_name}_Aaw_comparison_neutral_acids.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved to: {output_dir}")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
