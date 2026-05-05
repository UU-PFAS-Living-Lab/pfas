import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Aaw loop for all Staring soils
    <!-- In this example we will showcase how to use the pfas package to create a basic forward modelling exercise of PFAS leaching in the vadose zone. -->
    """)
    return


@app.cell
def _():
    #loading relevant modules 
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
        WaterPreprocessor,
        available_datasets,
        load_dataset,
        mo,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading PFAS data
    <!-- First we load our configuration file, in which we provide our model with most of the parameters needed to run our model. The input is later checked to see if it meets the requirements. -->
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

    # Solid-phase adsorption: linear
    frac_int = 1.0
    rate_const = 0.0

    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc} L/kg  |  K_sc = {K_sc} L/kg")
    return n_CFx, pfas, pfas_name, soil_db


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Utilities from the original code (Aaw_mulitple_methods.py)
    """)
    return


@app.cell
def _(GridGenerator, pfas):
    from pfas.utils import kd_fabregat_palau
    # Grid
    grid_gen = GridGenerator(
        domain_length=100,
        spatial_resolution=1.0,
        time_resolution=(60*60*24*365),
        time_total=250*(60*60*24*365),
    )
    grid_results = grid_gen.compute()

    pulse_duration = 25 * (60 * 60 * 24 * 365)

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
    return grid_results, kaw_input, kd_fabregat_palau, pulse_duration


@app.cell
def _(available_datasets):
    print(available_datasets())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Defining loop for all Staring soils

    <!-- We now create an instance of our model and do the preprocessing for different categories of our data. Simoultaneously, the input is checked for its validity. We run the analytical solution through the class SimulationRunner.

    In this step it is also possible to overwrite certain parameters from the TOML file, as examplified by the parameter porosity here. -->
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    SimulationRunner,
    SorptionKawCalculated,
    WaterPreprocessor,
    grid_results,
    kaw_input,
    kd_fabregat_palau,
    n_CFx,
    pfas,
    pulse_duration,
    soil_db,
):
    from pfas.utils import aaw_func_thermo, aaw_func_1, aaw_func_2
    # ── Loop over all 18 Staring soil types ──────────────────────────────────
    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")] 
    # Or define explicitly if order matters:
    # staring_soils = [f"Staring-O{i:02d}" for i in range(1, 19)]

    all_soil_results = {}  # keyed by soil_name

    for soil_name in staring_soils:
        soil        = soil_db[soil_name]
        bulk_dens   = soil["rho_b"]["value"]
        porosity    = soil["porosity"]
        theta_r     = soil["theta_r"]
        theta_s     = soil["theta_s"]
        K_sat       = soil["K_sat"]["value"]
        vg_alpha    = soil["van_genuchten"]["alpha"]["value"]
        vg_n        = soil["van_genuchten"]["n"]
        vg_l        = soil["van_genuchten"]["l"]
        dispersivity = 4.5
        f_oc        = soil["f_oc"]["value"] / 100
        f_clay      = soil["f_clay"]["value"] / 100
        f_silt      = soil["f_silt"]["value"] / 100
        f_silt_clay = f_silt + f_clay
        d50         = soil["d50"]["value"] / 10000  # µm → cm

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
        theta = water_results["hydro_properties"].water_content

        # Aaw — all three methods
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
        func_1_Aaw = aaw_func_1(
            th=theta, 
            ths=theta_s, 
            poro=porosity, 
            d50=d50
        )
        func_2_Aaw = aaw_func_2(
            th=theta, 
            ths=theta_s, 
            d50=d50
        )

        # Boundary conditions (same pulse for all soils)
        boundary_results = BoundaryPreprocessor(
            C_list=[pfas['M']["value"] * 1e-9, 0.0],
            T_list=[0.0, pulse_duration]
        ).compute()

        # Solid-phase sorption
        sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "kinetic": {"frac_int": 1.0, "rate_const": 0.0},
            "linear": {
                "Kd_method": "direct_input",
                "Kd": kd_fabregat_palau(n_CFx, f_oc, f_silt_clay)
            },
        }

        # Kawi for each Aaw method
        kawi_results = {}
        for aaw_label, aaw_val in [
            ("thermo", func_thermo_Aaw),
            ("func1",  func_1_Aaw),
            ("func2",  func_2_Aaw),
        ]:
            kawi = SorptionKawCalculated(
                **kaw_input,
                hydro_properties=water_results["hydro_properties"],
                aaw=aaw_val,
            )
            kawi_results[aaw_label] = kawi.compute()

        # Simulations
        sim_results = {}
        for aaw_label, kawi_res in kawi_results.items():
            sim_results[aaw_label] = SimulationRunner(
                grid=grid_results["grid"],
                bulk_density=bulk_dens,
                boundary_conditions=boundary_results["boundary_conditions"],
                hydro_properties=water_results["hydro_properties"],
                awi_retardation=kawi_res["awi_retardation"],
                sorption_solid=sorption_solid,
                kinetic_sorption=False,
                volume_averaged=True,
            ).compute()

        all_soil_results[soil_name] = {
            "Aaw": {"thermo": func_thermo_Aaw, "func1": func_1_Aaw, "func2": func_2_Aaw},
            "kawi": kawi_results,
            "sim": sim_results,
            "water": water_results,
            "soil": soil,
        }
        print(f"  {soil_name}: Aaw = {func_thermo_Aaw:.4f} | {func_1_Aaw:.4f} | {func_2_Aaw:.4f}")

    print(f"\nDone — {len(all_soil_results)} soils processed.")
    print(theta)
    return (all_soil_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Calculate difference between methods with respect to the Thermodynamic method
    """)
    return


@app.cell
def _(all_soil_results):
    def add_aaw_differences(all_soil_results):
        for soil_name, results in all_soil_results.items():
            aaw = results["Aaw"]

            results["Aaw_diff"] = {
                "func1_vs_thermo": (aaw["func1"] - aaw["thermo"]) / aaw["thermo"],
                "func2_vs_thermo": (aaw["func2"] - aaw["thermo"]) / aaw["thermo"],
            }
        return all_soil_results


    all_soil_results_with_diffs = add_aaw_differences(all_soil_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving variables to separate dataframe
    """)
    return


@app.cell
def _(all_soil_results, grid_results, kd_fabregat_palau, n_CFx, pfas_name):
    def save_results():
        # ── 1. Scalar summary table (one row per soil) ──────────────────────────────
        import pandas as pd
        import numpy as np
        import os

        out_dir=r"C:\Users\youri\Documents\ESW\Thesis_v2\04_Processing\4_dataframe_with_key_variables/results_v1"
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for soil_name, res in all_soil_results.items():
            soil    = res["soil"]
            water   = res["water"]
            theta   = water["hydro_properties"].water_content

            rows.append({
                "soil":             soil_name,
                "theta":            float(theta),
                "f_oc":             soil["f_oc"]["value"] / 100,
                "f_clay":           soil["f_clay"]["value"] / 100,
                "f_silt":           soil["f_silt"]["value"] / 100,
                "porosity":         soil["porosity"],
                "rho_b":            soil["rho_b"]["value"],
                "Kd":               kd_fabregat_palau(n_CFx,
                                        soil["f_oc"]["value"] / 100,
                                        (soil["f_silt"]["value"] + soil["f_clay"]["value"]) / 100),
                "Aaw_thermo":       res["Aaw"]["thermo"],
                "Aaw_func1":        res["Aaw"]["func1"],
                "Aaw_func2":        res["Aaw"]["func2"],
                "Aaw_diff_func1":   res["Aaw_diff"]["func1_vs_thermo"],
                "Aaw_diff_func2":   res["Aaw_diff"]["func2_vs_thermo"],
                "pfas_name":        pfas_name,
                "Effective_sat":    ((float(theta)-soil["theta_r"])/(soil["theta_s"]-soil["theta_r"])),
                "Kaw_thermo":       res["kawi"]["thermo"]["Kaw"],
                "Kaw_func1":        res["kawi"]["func1"]["Kaw"],
                "Kaw_func2":        res["kawi"]["func2"]["Kaw"],
            })

        df_scalars = pd.DataFrame(rows)
        df_scalars.to_parquet(os.path.join(out_dir, "scalars.parquet"), index=False)

        # ── 2. Breakthrough curves (depth × time, per soil × method) ────────────────
        time_arr  = grid_results["grid"].time
        depth_arr = grid_results["grid"].depth

        # Save arrays separately as numpy (cleaner than object columns)
        np.save(os.path.join(out_dir, "time.npy"),  time_arr)
        np.save(os.path.join(out_dir, "depth.npy"), depth_arr)

        btc_index = []
        for soil_name, res in all_soil_results.items():
            for method in ("thermo", "func1", "func2"):
                C_tot = res["sim"][method]["C_tot"]
                fname = os.path.join(out_dir, f"btc_{soil_name}_{method}.npy")
                np.save(fname, C_tot)
                btc_index.append({"soil": soil_name, "method": method})

        pd.DataFrame(btc_index).to_parquet(os.path.join(out_dir, "btc_index.parquet"), index=False)

        print(f"Saved all results to:\n {out_dir}")

    save_results()
    return


@app.cell
def _(df):
    print(df[["soil", "Kaw_thermo", "Aaw_thermo"]])
    print(df["Kaw_thermo"].describe())
    print(df["Kaw_thermo"].nunique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting Aaw for all soils
    <!-- We access our data through model.generated_data, which prints all the output. Accessing it in this manneer also allows for easier plotting. -->
    """)
    return


@app.cell
def _(all_soil_results, plt):
    # ── Plot: Aaw comparison across all 18 Staring soils ─────────────────────
    exclude = "Staring-017"
    soil_names_plot = [s for s in all_soil_results.keys() if s not in exclude]
    aaw_thermo  = [all_soil_results[s]["Aaw"]["thermo"] for s in soil_names_plot]
    aaw_func1   = [all_soil_results[s]["Aaw"]["func1"]  for s in soil_names_plot]
    aaw_func2   = [all_soil_results[s]["Aaw"]["func2"]  for s in soil_names_plot]

    x     = range(len(soil_names_plot))
    width = 0.28

    fig_aaw, ax_aaw = plt.subplots(figsize=(16, 5))
    ax_aaw.bar([i - width for i in x], aaw_thermo, width, label="Thermodynamic", color="#1f77b4")
    ax_aaw.bar([i         for i in x], aaw_func1,  width, label="Function 1",    color="#ff7f0e")
    ax_aaw.bar([i + width for i in x], aaw_func2,  width, label="Function 2",    color="#2ca02c")
    ax_aaw.set_xticks(list(x))
    ax_aaw.set_xticklabels(soil_names_plot, rotation=45, ha="right", fontsize=8)
    ax_aaw.set_ylabel("Aaw (cm²/cm³)")
    ax_aaw.set_yscale("log")
    ax_aaw.set_title(f"Air-water interfacial area — all Staring soils")
    ax_aaw.legend()
    ax_aaw.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_aaw
    return


@app.cell
def _(all_soil_results, pfas_name, plt):
    # ── Plot: Aaw difference vs thermodynamic baseline ───────────────────────
    exclude_soils = ["Staring-O17", "Staring-O11", "Staring-O12", "Staring-O13"] # list, not string
    soil_names_plot_diff = [s for s in all_soil_results.keys() if s not in exclude_soils]

    func1_vs_thermo = [all_soil_results[s]["Aaw_diff"]["func1_vs_thermo"] for s in soil_names_plot_diff]
    func2_vs_thermo = [all_soil_results[s]["Aaw_diff"]["func2_vs_thermo"] for s in soil_names_plot_diff]

    x_diff     = range(len(soil_names_plot_diff))
    width_diff = 0.35

    fig_aaw_diff, ax_aaw_diff = plt.subplots(figsize=(16, 5))
    ax_aaw_diff.bar([i - width_diff / 2 for i in x_diff], func1_vs_thermo, width_diff, label="Function 1", color="#ff7f0e")
    ax_aaw_diff.bar([i + width_diff / 2 for i in x_diff], func2_vs_thermo, width_diff, label="Function 2", color="#2ca02c")
    ax_aaw_diff.axhline(0, color="black", linewidth=0.8)  # zero reference line
    ax_aaw_diff.set_xticks(list(x_diff))                  # ← was list(x), wrong range
    ax_aaw_diff.set_xticklabels(soil_names_plot_diff, rotation=45, ha="right", fontsize=8)
    ax_aaw_diff.set_ylabel("ΔAaw vs Thermodynamic (fraction)")
    ax_aaw_diff.set_title(f"Air-water interfacial area difference — all Staring soils | PFAS: {pfas_name}")
    ax_aaw_diff.legend()
    ax_aaw_diff.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_aaw_diff
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting BTC for all soils

    <!-- The generated_data allows you to access both input and output of the model and thus also allows for relative simple plotting. -->
    """)
    return


@app.cell
def _(all_soil_results, grid_results, pfas_name, plt):
    # ── Plot: breakthrough curves at depth ≈ 50 cm, all soils ────────────────
    seconds_per_year_bt = 60 * 60 * 24 * 365
    depth_target_cm = 50
    depth_idx_bt = min(
        range(len(grid_results["grid"].depth)),
        key=lambda i: abs(grid_results["grid"].depth[i] - depth_target_cm)
    )
    actual_depth_bt = grid_results["grid"].depth[depth_idx_bt]

    fig_bt, axes_bt = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    method_labels_bt = {"thermo": "Thermodynamic", "func1": "Function 1", "func2": "Function 2"}
    cmap_bt = plt.get_cmap("tab20", len(all_soil_results))

    for ax_idx, (aaw_key, aaw_label_bt) in enumerate(method_labels_bt.items()):
        ax_bt = axes_bt[ax_idx]
        for s_idx, sname in enumerate(all_soil_results.keys()):
            C = all_soil_results[sname]["sim"][aaw_key]["C_tot"][depth_idx_bt, :]
            ax_bt.plot(
                grid_results["grid"].time / seconds_per_year_bt, C,
                color=cmap_bt(s_idx), label=sname, linewidth=1.2
            )
        ax_bt.set_title(f"Aaw: {aaw_label_bt}")
        ax_bt.set_xlabel("Time (years)")
        if ax_idx == 0:
            ax_bt.set_ylabel("Total PFAS Concentration (mg/L)")
        ax_bt.grid(True, alpha=0.3)

    axes_bt[-1].legend(title="Soil", bbox_to_anchor=(1.02, 1),
                       loc="upper left", fontsize=7, ncol=1)
    fig_bt.suptitle(
        f"Breakthrough at ≈{actual_depth_bt:.0f} cm — {pfas_name}, all Staring soils",
        fontsize=12
    )
    plt.tight_layout()
    fig_bt
    return


if __name__ == "__main__":
    app.run()
