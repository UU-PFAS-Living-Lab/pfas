import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loop for all Staring soils
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
        #SorptionKawiDirectInput,
        #SorptionKawCalculated,
        #SorptionKawLangmuir,
        #SorptionKawSzyszkowski,
        SimulationRunner,
    )
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    from pathlib import Path
    import marimo as mo
    import matplotlib.ticker as ticker
    import pandas as pd
    import numpy as np
    import os

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        GridGenerator,
        Path,
        SimulationRunner,
        WaterPreprocessor,
        available_datasets,
        load_dataset,
        mo,
        np,
        os,
        pd,
        plt,
        ticker,
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
    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")

    pfas_names = ['TFA', 'PFBA', 'HFPO-DA', 'PFPeA', 'PFBS', 'PFHxA', 'PFHpA', 'PFHxS', 'PFOA', 'PFNA', 'PFOS', 'PFDA']  # add/remove as needed

    print("Available PFAS:", list(pfas_db.keys()))  # to see what's available

    sigma0=72.8

    # Solid-phase adsorption: linear
    frac_int = 1.0
    rate_const = 0.0
    return pfas_db, pfas_names, sigma0, soil_db


@app.cell
def _(available_datasets):
    print(available_datasets())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Utilities from the original code (Aaw_mulitple_methods.py)
    """)
    return


@app.cell
def _(GridGenerator):
    from pfas.utils import kd_fabregat_palau
    # Grid
    grid_gen = GridGenerator(
        domain_length=100,
        spatial_resolution=0.5,
        time_resolution=(1/12)*(60*60*24*365),
        time_total=250*(60*60*24*365),
    )
    grid_results = grid_gen.compute()

    pulse_duration = 25 * (60 * 60 * 24 * 365)
    return grid_results, kd_fabregat_palau, pulse_duration


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Defining loop for all Staring soils & PFAS-types

    <!-- We now create an instance of our model and do the preprocessing for different categories of our data. Simoultaneously, the input is checked for its validity. We run the analytical solution through the class SimulationRunner.

    In this step it is also possible to overwrite certain parameters from the TOML file, as examplified by the parameter porosity here. -->
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    SimulationRunner,
    SorptionKawLangmuir,
    SorptionKawSzyszkowski,
    WaterPreprocessor,
    grid_results,
    kd_fabregat_palau,
    pfas_db,
    pfas_names,
    pulse_duration,
    sigma0,
    soil_db,
):
    from pfas.utils import aaw_func_thermo, aaw_func_GSSA, aaw_func_d50, aaw_func_nonlinear_d50
    # Loop over all 18 Staring soil types
    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")] 
    all_pfas_results = {}  # keyed by pfas_name → soil_name

    for pfas_name in pfas_names:
        pfas  = pfas_db[pfas_name]
        n_CFx = pfas["n_CFx"]

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

        all_soil_results = {}

        for soil_name in staring_soils:
            soil         = soil_db[soil_name]
            bulk_dens    = soil["rho_b"]["value"]
            porosity     = soil["porosity"]
            theta_r      = soil["theta_r"]
            theta_s      = soil["theta_s"]
            K_sat        = soil["K_sat"]["value"]
            vg_alpha     = soil["van_genuchten"]["alpha"]["value"]
            vg_n         = soil["van_genuchten"]["n"]
            vg_l         = soil["van_genuchten"]["l"]
            dispersivity = 4.5
            f_oc         = soil["f_oc"]["value"] / 100
            f_clay       = soil["f_clay"]["value"] / 100
            f_silt       = soil["f_silt"]["value"] / 100
            f_silt_clay  = f_silt + f_clay
            d50          = soil["d50"]["value"] / 10000  # µm → cm

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

            # Aaw — all four methods
            # 1. SWC-based
            func_thermo_aaw = aaw_func_thermo(
                sigma0=71,
                poro=porosity,
                alpha=vg_alpha,
                n=vg_n,
                th=theta,
                thr=theta_r,
                ths=theta_s,
                sf=4.15
            )
            # 2. GSSA-based, linear saturation model
            func_GSSA_aaw = aaw_func_GSSA(
                th=theta,
                ths=theta_s,
                poro=porosity,
                d50=d50
            )
            # 3. d50 correlation, linear saturation model
            func_d50_aaw = aaw_func_d50(
                th=theta,
                ths=theta_s,
                d50=d50
            )
            # 4. d50 correlation, nonlinear saturation model
            func_nonlinear_d50_aaw = aaw_func_nonlinear_d50(
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

            def select_Kaw_method(pfas: dict, hydro_properties, aaw: float):
                """
                Uses Szyszkowski method when Szyszkowski_params are available from PFASs.
                Falls back to Langmuir when Szyszkowski_params are missing.
                """
                a = pfas["Szyszkowski_params"]["a"]["value"]
                b = pfas["Szyszkowski_params"]["b"]["value"]

                if a is not None and b is not None:
                    return SorptionKawSzyszkowski(
                        sigma0=sigma0,
                        a=a,
                        b=b,
                        hydro_properties=hydro_properties,
                        aaw=aaw,
                        chi=1,
                        T=293.15,
                    )
                else:
                    return SorptionKawLangmuir(
                        **kaw_input,
                        hydro_properties=hydro_properties,
                        aaw=aaw,
                    )

            for aaw_label, aaw_val in [
                ("thermo", func_thermo_aaw),
                ("func_GSSA", func_GSSA_aaw),
                ("func_d50", func_d50_aaw),
                ("func_nonlin_d50", func_nonlinear_d50_aaw),
            ]:
                kawi_sorp = select_Kaw_method(
                    pfas=pfas,
                    hydro_properties=water_results["hydro_properties"],
                    aaw=aaw_val,
                )

                kawi_results[aaw_label] = kawi_sorp.compute(Cw=1e-12)

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
                "Aaw": {
                    "thermo": func_thermo_aaw,
                    "func_GSSA": func_GSSA_aaw,
                    "func_d50": func_d50_aaw, 
                    "func_nonlin_d50": func_nonlinear_d50_aaw,
                       },
                "kawi": kawi_results,
                "sim": sim_results,
                "water": water_results,
                "soil": soil,
            }
            print(f"  {soil_name}: Aaw = {func_thermo_aaw:.4f} | {func_GSSA_aaw:.4f} | {func_d50_aaw:.4f} | {func_nonlinear_d50_aaw:.4f}")

        all_pfas_results[pfas_name] = all_soil_results  # ← the missing line
        print(f"Done — {pfas_name}: {len(all_soil_results)} soils processed.")
    return all_pfas_results, all_soil_results, pfas_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving variables to separate dataframe
    """)
    return


@app.cell(disabled=True)
def _(all_pfas_results, grid_results, kd_fabregat_palau, np, os, pd, pfas_db):
    def save_results():

        # ── Add Aaw differences ──────────────────────────────────────────────────
        for pfas_name, all_soil_results in all_pfas_results.items():
            for soil_name, results in all_soil_results.items():
                aaw = results["Aaw"]
                results["Aaw_diff"] = {
                    "func1_vs_thermo": (aaw["func1"] - aaw["thermo"]) / aaw["thermo"],
                    "func2_vs_thermo": (aaw["func2"] - aaw["thermo"]) / aaw["thermo"],
                }

        # ── Save results ─────────────────────────────────────────────────────────
        out_dir = r"C:\Users\youri\Documents\ESW\Thesis_v2\04_Processing\4_dataframe_with_key_variables\results_v3_neutralPFASs"
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        btc_index = []

        for pfas_name, all_soil_results in all_pfas_results.items():
            for soil_name, res in all_soil_results.items():
                soil  = res["soil"]
                water = res["water"]
                theta = water["hydro_properties"].water_content
                n_CFx = pfas_db[pfas_name]["n_CFx"]

                rows.append({
                    "pfas_name":      pfas_name,
                    "soil":           soil_name,
                    "theta":          float(theta),
                    "f_oc":           soil["f_oc"]["value"] / 100,
                    "f_clay":         soil["f_clay"]["value"] / 100,
                    "f_silt":         soil["f_silt"]["value"] / 100,
                    "porosity":       soil["porosity"],
                    "rho_b":          soil["rho_b"]["value"],
                    "Kd":             kd_fabregat_palau(n_CFx,
                                          soil["f_oc"]["value"] / 100,
                                          (soil["f_silt"]["value"] + soil["f_clay"]["value"]) / 100),
                    "Aaw_thermo":     res["Aaw"]["thermo"],
                    "Aaw_func1":      res["Aaw"]["func1"],
                    "Aaw_func2":      res["Aaw"]["func2"],
                    "Aaw_diff_func1": res["Aaw_diff"]["func1_vs_thermo"],
                    "Aaw_diff_func2": res["Aaw_diff"]["func2_vs_thermo"],
                    "Effective_sat":  ((float(theta) - soil["theta_r"]) / (soil["theta_s"] - soil["theta_r"])),
                    "Kaw":            res["kawi"]["thermo"]["Kaw"],
                })

                for method in ("thermo", "func1", "func2"):
                    C_tot = res["sim"][method]["C_tot"]
                    fname = os.path.join(out_dir, f"btc_{pfas_name}_{soil_name}_{method}.npy")
                    np.save(fname, C_tot)
                    btc_index.append({"pfas_name": pfas_name, "soil": soil_name, "method": method})

        pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "scalars.parquet"), index=False)
        pd.DataFrame(btc_index).to_parquet(os.path.join(out_dir, "btc_index.parquet"), index=False)
        np.save(os.path.join(out_dir, "time.npy"),  grid_results["grid"].time)
        np.save(os.path.join(out_dir, "depth.npy"), grid_results["grid"].depth)

        print(f"Saved {len(rows)} rows to {out_dir}")

    save_results()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting Aaw for all soils
    <!-- We access our data through model.generated_data, which prints all the output. Accessing it in this manneer also allows for easier plotting. -->
    """)
    return


@app.cell
def _(all_soil_results, np, plt):
    # ── Plot: Aaw comparison across all 18 Staring soils ─────────────────────
    exclude = ""
    soil_names_plot = [s for s in all_soil_results.keys() if s not in exclude]
    aaw_thermo     = [all_soil_results[s]["Aaw"]["thermo"]          for s in soil_names_plot]
    aaw_GSSA       = [all_soil_results[s]["Aaw"]["func_GSSA"]       for s in soil_names_plot]
    aaw_d50        = [all_soil_results[s]["Aaw"]["func_d50"]        for s in soil_names_plot]
    aaw_nonlin_d50 = [all_soil_results[s]["Aaw"]["func_nonlin_d50"] for s in soil_names_plot]

    x     =np.arange(len(soil_names_plot))
    width = 0.20

    fig_aaw, ax_aaw = plt.subplots(figsize=(10, 5))
    ax_aaw.bar(x - 1.5 * width, aaw_thermo,     width, label="Thermodynamic",        color="#1f77b4")
    ax_aaw.bar(x - 0.5 * width, aaw_GSSA,       width, label="GSSA-based",           color="#ff7f0e")
    ax_aaw.bar(x + 0.5 * width, aaw_d50,        width, label="d50-based",            color="#2ca02c")
    ax_aaw.bar(x + 1.5 * width, aaw_nonlin_d50, width, label="d50, nonlinear saturation", color="#d62728")

    ax_aaw.set_xticks(list(x))
    ax_aaw.set_xticklabels(soil_names_plot, rotation=45, ha="right", fontsize=8)
    ax_aaw.set_ylabel("Aaw (cm²/cm³)")
    # ax_aaw.set_yscale("log")
    ax_aaw.set_title(f"Air-water interfacial area — all Staring soils")
    ax_aaw.legend()
    ax_aaw.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_aaw
    return


@app.cell
def _(all_soil_results, np, plt):
    # Calculate: Aaw difference vs thermodynamic baseline
    def add_aaw_differences(all_soil_results):
        for soil_name, results in all_soil_results.items():
            aaw = results["Aaw"]

            results["Aaw_diff"] = {
                "GSSA_vs_thermo"     : ((aaw["func_GSSA"] - aaw["thermo"]) / aaw["thermo"]),
                "d50_vs_thermo"      : ((aaw["func_d50"] - aaw["thermo"]) / aaw["thermo"]),
                "nonlinear_vs_thermo": ((aaw["func_nonlin_d50"] - aaw["thermo"]) / aaw["thermo"])
            }
        return all_soil_results
    all_soil_results_with_diffs = add_aaw_differences(all_soil_results)

    # Plot: Aaw difference vs thermodynamic baseline
    exclude_soils = [""] # list, not string
    soil_names_plot_diff = [s for s in all_soil_results.keys() if s not in exclude_soils]

    GSSA_vs_thermo      = [all_soil_results[s]["Aaw_diff"]["GSSA_vs_thermo"] for s in soil_names_plot_diff]
    d50_vs_thermo       = [all_soil_results[s]["Aaw_diff"]["d50_vs_thermo"] for s in soil_names_plot_diff]
    nonlinear_vs_thermo = [all_soil_results[s]["Aaw_diff"]["nonlinear_vs_thermo"] for s in soil_names_plot_diff]

    x_diff = np.arange(len(soil_names_plot_diff))
    width_diff = 0.25

    fig_aaw_diff, ax_aaw_diff = plt.subplots(figsize=(10, 5))
    ax_aaw_diff.bar(x_diff-width_diff, GSSA_vs_thermo,      width_diff, label="GSSA-based", color="#ff7f0e")
    ax_aaw_diff.bar(x_diff,            d50_vs_thermo,       width_diff, label="d50-based",  color="#2ca02c")
    ax_aaw_diff.bar(x_diff+width_diff, nonlinear_vs_thermo, width_diff, label="d50, nonlinear saturation",  color="#d62728")
    ax_aaw_diff.axhline(0, color="black", linewidth=0.8)
    ax_aaw_diff.set_xticks(list(x_diff))    
    ax_aaw_diff.set_xticklabels(soil_names_plot_diff, rotation=45, ha="right", fontsize=8)
    ax_aaw_diff.set_ylabel("ΔAaw vs thermodynamic method (-)")
    ax_aaw_diff.set_title(f"Air-water interfacial area difference — all Staring soils | theta = theta")
    ax_aaw_diff.set_yscale("symlog") #, linthresh=10)
    ax_aaw_diff.legend()
    ax_aaw_diff.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_aaw_diff
    return


@app.cell
def _(all_soil_results, np, plt):
    # Calculate: Aaw Log10-ratio vs thermodynamic baseline
    def add_aaw_log_ratio(all_soil_results):
        for soil_name, results in all_soil_results.items():
            aaw = results["Aaw"]

            results["Aaw_log_ratio"] = {
                "GSSA_vs_thermo"         : np.log10(aaw["func_GSSA"]/aaw["thermo"]),
                "d50_linear_vs_thermo"   : np.log10(aaw["func_d50"]/aaw["thermo"]),
                "d50_nonlinear_vs_thermo": np.log10(aaw["func_nonlin_d50"]/aaw["thermo"])
            }
        return all_soil_results
    all_soil_results_log_ratio = add_aaw_log_ratio(all_soil_results)

    # Plot: Aaw Log10-ratio vs thermodynamic baseline
    def plot_log_ratio_soils(all_soil_results):
        soil_names_plot=list(all_soil_results.keys())
        fig, ax = plt.subplots(figsize=(10,5))

        GSSA_vs_thermo = [all_soil_results[soil_name]["Aaw_log_ratio"]["GSSA_vs_thermo"] for soil_name in soil_names_plot]
        d50_linear_vs_thermo = [all_soil_results[soil_name]["Aaw_log_ratio"]["d50_linear_vs_thermo"] for soil_name in soil_names_plot]
        d50_nonlinear_vs_thermo = [all_soil_results[soil_name]["Aaw_log_ratio"]["d50_nonlinear_vs_thermo"] for soil_name in soil_names_plot]

        x_diff = np.arange(len(soil_names_plot))
        width_diff = 0.25

        ax.bar(x_diff-width_diff, GSSA_vs_thermo, width_diff, label="GSSA vs thermodynamic", color="#ff7f0e")
        ax.bar(x_diff, d50_linear_vs_thermo, width_diff, label="d50_linear vs thermodynamic", color="#2ca02c")
        ax.bar(x_diff+width_diff, d50_nonlinear_vs_thermo, width_diff, label="d50_nonlinear vs thermodynamic",  color="#d62728")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x_diff))    
        ax.set_xticklabels(soil_names_plot, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("log10(Aaw method/Aaw thermodynamic)")
        ax.set_title(f"Air-water interfacial area log ratio — all Staring soils | theta = theta")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        plt.show()
        return fig

    fig_log_ratios = plot_log_ratio_soils(
        all_soil_results=all_soil_results,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting BTC for all soils

    <!-- The generated_data allows you to access both input and output of the model and thus also allows for relative simple plotting. -->
    """)
    return


@app.cell
def _(aaw_key, all_soil_results, sname):
    print("sname:", sname)
    print("aaw_key:", aaw_key)
    print("available sim keys:", all_soil_results[sname]["sim"].keys())
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

    fig_bt, axes_bt = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    method_labels_bt = {"thermo": "Thermodynamic", "func_GSSA": "GSSA-based", "func_d50": "d50-based", "func_nonlin_d50": "d50, nonlinear saturation"}
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
    return aaw_key, sname


@app.cell
def _(all_pfas_results, pd, pfas_names, plt, ticker):
    # Uses local Marimo variables already present in the notebook:
    # all_pfas_results, pfas_names

    kaw_local_rows = []

    for kaw_local_pfas_name in pfas_names:
        kaw_local_soil_results = all_pfas_results[kaw_local_pfas_name]

        # Kaw is independent of soil here, so take the first available soil entry.
        kaw_local_first_soil_name = next(iter(kaw_local_soil_results))
        kaw_local_kaw = kaw_local_soil_results[kaw_local_first_soil_name]["kawi"]["thermo"]["Kaw"]

        kaw_local_rows.append({
            "pfas_name": kaw_local_pfas_name,
            "Kaw": kaw_local_kaw,
        })

    kaw_local_df = pd.DataFrame(kaw_local_rows)

    kaw_local_pfas_order = [
        "TFA",
        "PFBA",
        "HFPO-DA",
        "PFPeA",
        "PFBS",
        "PFHxA",
        "PFHpA",
        "PFHxS",
        "PFOA",
        "PFNA",
        "PFOS",
        "PFDA",
    ]

    kaw_local_df_ordered = (
        kaw_local_df
        .set_index("pfas_name")
        .reindex(kaw_local_pfas_order)
        .reset_index()
    )

    kaw_local_missing = kaw_local_df_ordered[kaw_local_df_ordered["Kaw"].isna()]["pfas_name"].tolist()
    if kaw_local_missing:
        raise ValueError(f"Missing PFAS in all_pfas_results: {kaw_local_missing}")

    kaw_local_fig, kaw_local_ax = plt.subplots(figsize=(6.5, 4.2))

    kaw_local_ax.scatter(
        kaw_local_df_ordered["pfas_name"],
        kaw_local_df_ordered["Kaw"],
        s=55,
        color="red",
    )

    kaw_local_ax.set_yscale("log")

    # Set only the major ticks you want to show
    kaw_local_ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    kaw_local_ax.set_title(
        r"Partitioning coefficient air-water interface "
        r"$(C = 1.0\ \frac{\mathrm{pmol}}{\mathrm{l}})$",
        fontsize=13,
    )

    kaw_local_ax.set_xlabel("PFAS", fontsize=12)
    kaw_local_ax.set_ylabel(r"$K_{\mathrm{aw}}\ [\mathrm{cm}^3/\mathrm{cm}^2]$", fontsize=12)

    kaw_local_ax.grid(True, which="major", axis="both", alpha=0.55)

    kaw_local_ax.tick_params(axis="x", rotation=35, labelsize=10)
    kaw_local_ax.tick_params(axis="y", labelsize=10)

    kaw_local_fig.tight_layout()
    kaw_local_fig
    return kaw_local_df_ordered, kaw_local_pfas_order


@app.cell
def _(Path, np, pd, plt):
    def plot_kaw_comparison():
        csv_path = Path(__file__).parent / "Kaw_v3.csv"

        kaw_df = pd.read_csv(
            csv_path,
            na_values=["n.a.", "n.a", "NA", "N/A", ""]
        )

        pfas_col = "PFAS"
        method_cols = [col for col in kaw_df.columns if col != pfas_col]

        for col in method_cols:
            kaw_df[col] = pd.to_numeric(kaw_df[col], errors="coerce")

        kaw_x = np.arange(len(kaw_df))

        # markers = {
        #     "Brusseau & Van Glubt (2019) QSPR-model": "D",
        #     "Le et al. (2021) Anion acid form": "o",
        #     "Le et al. (2021) Neutral acid form": "s",
        #     "Guo et al. (2022) Experimental data": "X",
        #     "Brusseau & Guo (2022) Experimental data": "^",
        # }

        # colours = {
        #     "Brusseau & Van Glubt (2019) QSPR-model": "tab:red",
        #     "Le et al. (2021) Anion acid form": "tab:purple",
        #     "Le et al. (2021) Neutral acid form": "tab:orange",
        #     "Guo et al. (2022) Experimental data": "tab:blue",
        #     "Brusseau & Guo (2022) Experimental data": "tab:green",
        # }


        markers = {
            "QSPR-model": "D",
            "GC-model, anion acid": "o",
            "GC-model, neutral acid": "s",
            "Experimental data, Guo": "X",
            "Experimental data, B&G": "^",
        }

        colours = {
            "QSPR-model": "tab:red",
            "GC-model, anion acid": "tab:purple",
            "GC-model, neutral acid": "tab:orange",
            "Experimental data, Guo": "tab:blue",
            "Experimental data, B&G": "tab:green",
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        for col in method_cols:
            ax.scatter(
                kaw_x,
                kaw_df[col],
                label=col,
                s=110,
                marker=markers.get(col, "o"),
                facecolors="none",
                edgecolors=colours.get(col, "black"),
                linewidths=1.8
            )

        ax.set_yscale("log")
        ax.set_ylim(5e-8, 3)
        ax.set_xticks(kaw_x)
        ax.set_xticklabels(kaw_df[pfas_col], rotation=45, ha="right")

        ax.set_xlabel("PFAS")
        ax.set_ylabel(r"$K_{aw}$ [cm$^3$/cm$^2$]")
        # ax.set_title(
        #     r"Air-water interfacial adsorption coefficients for selected PFAS"
        #     # r"($C = 1.0\ \frac{\mathrm{pmol}}{\mathrm{l}}$)"
        # )

        ax.grid(True, which="major", alpha=0.5)
        ax.legend()

        print(method_cols)

        plt.tight_layout()
        return fig

    plot_kaw_comparison()
    return


@app.cell(disabled=True, hide_code=True)
def _(kaw_local_df_ordered, kaw_local_pfas_order, pfas_db):
    from pfas.utils import Kaw_0_Le2021, dG0_Le2021, Kaw_langmuir_Le2021
    import numpy as debug_np
    import pandas as debug_pd

    for _, debug_existing_row in kaw_local_df_ordered.iterrows():
        print(
            f"{debug_existing_row['pfas_name']:8s}  "
            f"Kaw = {debug_existing_row['Kaw']:.3e}  "
            f"log10 = {debug_np.log10(debug_existing_row['Kaw']):.3f}"
        )


    debug_kaw_rows = []

    for debug_pfas_name in kaw_local_pfas_order:
        debug_pfas_i = pfas_db[debug_pfas_name]

        debug_kaw0 = Kaw_0_Le2021(
            debug_pfas_i["n_CFx"],
            debug_pfas_i["n_CHx"],
            debug_pfas_i["n_COO"],
            debug_pfas_i["n_COOH"],
            debug_pfas_i["n_SO3"],
            debug_pfas_i["n_R4N"],
            debug_pfas_i["n_OH"],
            debug_pfas_i["n_OSO3"],
            debug_pfas_i["n__O_"],
            debug_pfas_i["n__S_"],
            debug_pfas_i["n_N_CH3_2_CH2_COO"],
        )

        debug_dg0 = dG0_Le2021(
            debug_pfas_i["n_CFx"],
            debug_pfas_i["n_CHx"],
            debug_pfas_i["n_COO"],
            debug_pfas_i["n_COOH"],
            debug_pfas_i["n_SO3"],
            debug_pfas_i["n_R4N"],
            debug_pfas_i["n_OH"],
            debug_pfas_i["n_OSO3"],
            debug_pfas_i["n__O_"],
            debug_pfas_i["n__S_"],
            debug_pfas_i["n_N_CH3_2_CH2_COO"],
        )

        debug_kaw = Kaw_langmuir_Le2021(
            debug_kaw0,
            debug_dg0,
            Cw=1e-12,
        )

        debug_kaw_rows.append({
            "pfas_name": debug_pfas_name,
            "n_CFx": debug_pfas_i["n_CFx"],
            "n_COO": debug_pfas_i["n_COO"],
            "n_SO3": debug_pfas_i["n_SO3"],
            "n__O_": debug_pfas_i["n__O_"],
            "Kaw0": debug_kaw0,
            "dG0": debug_dg0,
            "Kaw_at_1_pmol_L": debug_kaw,
            "log10_Kaw": debug_np.log10(debug_kaw),
        })

    debug_kaw_df = debug_pd.DataFrame(debug_kaw_rows)
    debug_kaw_df
    return


@app.cell
def _(
    all_pfas_results,
    kd_fabregat_palau,
    pd,
    pfas_db,
    pfas_names,
    plt,
    soil_db,
    ticker,
):
    # Uses local Marimo variables already present in the notebook:
    # all_pfas_results, pfas_names, pfas_db, soil_db, kd_fabregat_palau

    kd_box_rows = []

    for kd_box_pfas_name in pfas_names:
        kd_box_pfas = pfas_db[kd_box_pfas_name]
        kd_box_n_CFx = kd_box_pfas["n_CFx"]

        kd_box_soil_results = all_pfas_results[kd_box_pfas_name]

        for kd_box_soil_name, kd_box_soil_entry in kd_box_soil_results.items():
            kd_box_soil = kd_box_soil_entry["soil"]

            kd_box_f_oc = kd_box_soil["f_oc"]["value"] / 100
            kd_box_f_clay = kd_box_soil["f_clay"]["value"] / 100
            kd_box_f_silt = kd_box_soil["f_silt"]["value"] / 100
            kd_box_f_silt_clay = kd_box_f_silt + kd_box_f_clay

            kd_box_Kd = kd_fabregat_palau(
                kd_box_n_CFx,
                kd_box_f_oc,
                kd_box_f_silt_clay
            )

            kd_box_rows.append({
                "pfas_name": kd_box_pfas_name,
                "soil_name": kd_box_soil_name,
                "Kd": kd_box_Kd,
            })

    kd_box_df = pd.DataFrame(kd_box_rows)

    kd_box_soil_order = [s for s in soil_db.keys() if s.startswith("Staring-O")]

    kd_box_data = [
        kd_box_df.loc[kd_box_df["soil_name"] == kd_box_soil_name, "Kd"].dropna().values
        for kd_box_soil_name in kd_box_soil_order
    ]

    kd_box_labels = kd_box_soil_order

    kd_box_fig, kd_box_ax = plt.subplots(figsize=(10, 5))

    kd_box_ax.boxplot(
        kd_box_data,
        tick_labels=kd_box_labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
            "alpha": 1.0,
        },
    )

    kd_box_ax.set_yscale("log")
    kd_box_ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # kd_box_ax.set_title(
    #     r"Distribution of solid-water partitioning coefficients across PFAS",
    #     fontsize=13,
    # )

    kd_box_ax.set_xlabel("Soil", fontsize=12)
    kd_box_ax.set_ylabel(r"$K_{\mathrm{d}}\ [\mathrm{cm}^3/\mathrm{g}]$", fontsize=12)

    kd_box_ax.grid(True, which="major", axis="both", alpha=0.45)

    kd_box_ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    kd_box_ax.tick_params(axis="y", labelsize=10)

    for kd_box_label in kd_box_ax.get_xticklabels():
        kd_box_label.set_ha("right")

    kd_box_fig.tight_layout()
    kd_box_fig
    return


@app.cell
def _(all_pfas_results, pd, pfas_names, plt, soil_db):
    # Uses local Marimo variables already present in the notebook:
    # all_pfas_results, pfas_names, soil_db

    se_rows = []

    se_first_pfas_name = pfas_names[0]
    se_soil_results = all_pfas_results[se_first_pfas_name]

    for se_soil_name, se_soil_entry in se_soil_results.items():
        se_soil = se_soil_entry["soil"]

        se_theta = se_soil_entry["water"]["hydro_properties"].water_content
        se_theta_r = se_soil["theta_r"]
        se_theta_s = se_soil["theta_s"]

        se_effective_saturation = (se_theta - se_theta_r) / (se_theta_s - se_theta_r)

        se_rows.append({
            "soil_name": se_soil_name,
            "effective_saturation": se_effective_saturation,
        })

    se_df = pd.DataFrame(se_rows)

    se_soil_order = [s for s in soil_db.keys() if s.startswith("Staring-O")]

    se_df_ordered = (
        se_df
        .set_index("soil_name")
        .reindex(se_soil_order)
        .reset_index()
    )

    se_missing = se_df_ordered[se_df_ordered["effective_saturation"].isna()]["soil_name"].tolist()
    if se_missing:
        raise ValueError(f"Missing soils in all_pfas_results: {se_missing}")

    se_fig, se_ax = plt.subplots(figsize=(10, 5))

    se_x = range(len(se_df_ordered))

    se_ax.scatter(
        se_x,
        se_df_ordered["effective_saturation"],
        s=55,
    )

    se_ax.set_xticks(se_x)
    se_ax.set_xticklabels(
        se_df_ordered["soil_name"],
        rotation=45,
        ha="right"
    )

    # se_ax.set_title(
    #     r"Effective saturation of the Staring soils",
    #     fontsize=13,
    # )

    se_ax.set_xlabel("Soil", fontsize=12)
    se_ax.set_ylabel(r"Effective saturation $S_{\mathrm{e}}\ (-)$", fontsize=12)

    se_ax.set_ylim(0, 1.05)

    se_ax.grid(True, which="major", axis="both", alpha=0.45)

    se_ax.tick_params(axis="y", labelsize=10)

    se_fig.tight_layout()
    se_fig
    return


if __name__ == "__main__":
    app.run()
