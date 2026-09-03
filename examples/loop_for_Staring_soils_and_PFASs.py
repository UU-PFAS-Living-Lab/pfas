import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Methods for computing Aaw for different PFAS and different soils
    In this notebook we run our model, utilizing the different available Aaw approaches, for different PFAS and different soil types.
    We keep one method for calculating the Kd and one method for calculating Kaw.
    For completion, these differences are also shown at the bottom of this script.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.component import LinearSPsorption, Le2021_langmuir, Szyszkowski, Retardation, EquilibriumSolver, WaterPreprocessor, BoundaryPreprocessor, GridGenerator
    from pfas.data_loader import load_dataset, available_datasets
    from pfas.component.awi import SWCsorption, GuoTracer, D50AWI, NonlinearD50AWI, GSSAAWI
    from matplotlib import pyplot as plt
    from pathlib import Path
    import marimo as mo
    import matplotlib.ticker as ticker
    import pandas as pd
    import numpy as np
    import os
    from pfas.model import Model

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        D50AWI,
        EquilibriumSolver,
        GSSAAWI,
        GridGenerator,
        Le2021_langmuir,
        LinearSPsorption,
        Model,
        NonlinearD50AWI,
        Retardation,
        SWCsorption,
        Szyszkowski,
        WaterPreprocessor,
        load_dataset,
        mo,
        np,
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
    return pfas_db, pfas_names, soil_db


@app.cell
def _():

    # Grid
    #model = Model()
    # model.compute(GridGenerator,
    #     domain_length=100,                      # cm
    #     spatial_resolution=0.5,                 # cm
    #     time_resolution=(1/12) * (60*60*24*365),  # seconds
    #     time_total=250*(60*60*24*365),          # seconds   
    #              )

    pulse_duration = 25 * (60 * 60 * 24 * 365)
    return (pulse_duration,)


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
    D50AWI,
    EquilibriumSolver,
    GSSAAWI,
    GridGenerator,
    Le2021_langmuir,
    LinearSPsorption,
    Model,
    NonlinearD50AWI,
    Retardation,
    SWCsorption,
    Szyszkowski,
    WaterPreprocessor,
    pfas_db,
    pfas_names,
    pulse_duration,
    soil_db,
):

    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")]
    all_pfas_results = {}

    for pfas_name in pfas_names:
        pfas = pfas_db[pfas_name]
        all_soil_results = {}

        for soil_name in staring_soils:
            model = Model()  
            model.compute(GridGenerator,
                domain_length=100,                        # cm
                spatial_resolution=0.5,                   # cm
                time_resolution=(1/12) * (60*60*24*365),  # seconds
                time_total=250*(60*60*24*365),            # seconds
            )

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
            d50          = soil["d50"]["value"] / 10000

            # 1. Water flow
            model.compute(WaterPreprocessor,
                average_infiltration_rate=9.51e-7,
                hydraulic_conductivity=K_sat,
                porosity=porosity,
                dispersivity=dispersivity,
                van_genuchten_n=vg_n,
                van_genuchten_l=vg_l,
                residual_water_content=theta_r,
            )

            # 2. Boundary conditions
            model.compute(BoundaryPreprocessor,
                C_list=[pfas["M"]["value"] * 1e-9, 0.0],
                T_list=[0.0, pulse_duration],
            )

            # 3. Solid-phase sorption
            sorption_solid = {
                "kinetic_sorption": False,
                "sorption_isotherm": "linear",
                "kinetic": {"frac_int": 1.0, "rate_const": 0.0},
                "linear": {
                    "Kd_method": "fabregat_palau",
                    "n_CFx": pfas["structural_properties"]["n_CFx"],
                    "f_oc": f_oc,
                    "f_silt_clay": f_silt_clay,
                },
            }
            model.compute(LinearSPsorption, sorption_solid=sorption_solid)

            # 4. Kaw method
            a = pfas["Szyszkowski_params"]["a"]["value"]
            b = pfas["Szyszkowski_params"]["b"]["value"]
        
            if a is not None and b is not None:
                model.compute(Szyszkowski,
                    sigma0=71, a=a, b=b, chi=1, T=293.15, Cw=1e-12,
                )
            else:
                model.compute(Le2021_langmuir,
                    structural_properties=pfas["structural_properties"],
                    Cw=1e-12,
                )
                model.input_data["sigma0"] = 71 

            # 5. Aaw + Retardation + simulation
            aaw_methods = {
                "thermo": (SWCsorption, dict(scaling_factor_awi=4.15, van_genuchten_alpha=vg_alpha)),
                "func_GSSA": (GSSAAWI, dict(soil={"porosity": porosity, "d50": d50})),
                "func_d50": (D50AWI, dict(soil={"porosity": porosity, "d50": d50})),
                "func_nonlin_d50": (NonlinearD50AWI, dict(soil={"porosity": porosity, "d50": d50})),
            }

            aaw_results = {}
            sim_results = {}
            retardation_results = {}

            for aaw_label, (aaw_model_cls, aaw_kwargs) in aaw_methods.items():
                aaw_model = Model()
                aaw_model.input_data = dict(model.input_data)
                aaw_model.generated_data = dict(model.generated_data)
                aaw_model.default_values = dict(model.default_values)

                aaw_model.compute(aaw_model_cls, **aaw_kwargs)
                aaw_results[aaw_label] = aaw_model.aaw

                aaw_model.compute(Retardation, bulk_density=bulk_dens)
                retardation_results[aaw_label] = aaw_model.adsorption

                aaw_model.compute(EquilibriumSolver)
                sim_results[aaw_label] = aaw_model.C1

            all_soil_results[soil_name] = {
                "Aaw": aaw_results,
                "kawi": model.Kaw,
                "sim": sim_results,
                "water": model.hydro_properties,
                "soil": soil,
                "retardation": retardation_results,
                "Kd": model.Kd,
                "grid": model.grid,
            }

            print(
                f"  {soil_name}: "
                f"Aaw thermo = {aaw_results['thermo']:.4f}, "
                f"GSSA = {aaw_results['func_GSSA']:.4f}, "
                f"d50 = {aaw_results['func_d50']:.4f}, "
                f"nonlinear d50 = {aaw_results['func_nonlin_d50']:.4f}"
            )

        all_pfas_results[pfas_name] = all_soil_results
        print(f"Done — {pfas_name}: {len(all_soil_results)} soils processed.")
    return all_pfas_results, all_soil_results, model, pfas_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving variables to separate dataframe
    """)
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
    ax_aaw_diff.set_title(f"Air-water interfacial area difference — all Staring soils")
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
        ax.set_title(f"Air-water interfacial area log ratio — all Staring soils")
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
def _(all_soil_results, model, pfas_name, plt):
    # ── Plot: breakthrough curves at depth ≈ 50 cm, all soils ────────────────
    seconds_per_year_bt = 60 * 60 * 24 * 365
    depth_target_cm = 50
    depth_idx_bt = min(range(len(model.grid.depth)),
        key=lambda i: abs(model.grid.depth[i] - depth_target_cm)
    )
    actual_depth_bt = model.grid.depth[depth_idx_bt]

    fig_bt, axes_bt = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    method_labels_bt = {"thermo": "Thermodynamic", "func_GSSA": "GSSA-based", "func_d50": "d50-based", "func_nonlin_d50": "d50, nonlinear saturation"}
    cmap_bt = plt.get_cmap("tab20", len(all_soil_results))

    for ax_idx, (aaw_key, aaw_label_bt) in enumerate(method_labels_bt.items()):
        ax_bt = axes_bt[ax_idx]
        for s_idx, sname in enumerate(all_soil_results.keys()):
            C = all_soil_results[sname]["sim"][aaw_key][depth_idx_bt, :]
            ax_bt.plot(
                model.grid.time / seconds_per_year_bt, C,
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
def _(mo):
    mo.md(r"""
    #Other parameters
    In this code block we look at the differences in computed Kaw values, Kd values and effective saturation with the chosen methods.
    """)
    return


@app.cell
def _(all_pfas_results, pd, pfas_names, plt, ticker):
    PFAS_ORDER = [
        "TFA", "PFBA", "HFPO-DA", "PFPeA", "PFBS", "PFHxA",
        "PFHpA", "PFHxS", "PFOA", "PFNA", "PFOS", "PFDA",
    ]

    def get_kaw(soil_results: dict) -> float:
        """Extract Kaw for a PFAS, verifying it is identical across all soils."""
        kaw_values = {sname: res["kawi"] for sname, res in soil_results.items()}
        unique_vals = set(kaw_values.values())
        if len(unique_vals) > 1:
            raise ValueError(f"Kaw is not soil-independent: {kaw_values}")
        return next(iter(unique_vals))

    kaw_df = pd.DataFrame(
        {"pfas_name": name, "Kaw": get_kaw(all_pfas_results[name])}
        for name in pfas_names
    )

    kaw_df = kaw_df.set_index("pfas_name").reindex(PFAS_ORDER).reset_index()

    missing = kaw_df.loc[kaw_df["Kaw"].isna(), "pfas_name"].tolist()
    if missing:
        raise ValueError(f"Missing PFAS in all_pfas_results: {missing}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(kaw_df["pfas_name"], kaw_df["Kaw"], s=55, color="red")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.set_title(
        r"Partitioning coefficient air-water interface "
        r"$(C = 1.0\ \frac{\mathrm{pmol}}{\mathrm{l}})$",
        fontsize=13,
    )
    ax.set_xlabel("PFAS", fontsize=12)
    ax.set_ylabel(r"$K_{\mathrm{aw}}\ [\mathrm{cm}^3/\mathrm{cm}^2]$", fontsize=12)
    ax.grid(True, which="major", axis="both", alpha=0.55)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig
    return


@app.cell
def _(all_pfas_results, plt, soil_db, ticker):
    #Kd
    soil_order = [s for s in soil_db.keys() if s.startswith("Staring-O")]

    kd_by_soil = {
        soil_name: [
            soil_results[soil_name]["Kd"]
            for soil_results in all_pfas_results.values()
            if soil_name in soil_results
        ]
        for soil_name in soil_order
    }

    fig_kd, ax_kd = plt.subplots(figsize=(10, 5))
    ax_kd.boxplot(
        [kd_by_soil[s] for s in soil_order],
        tick_labels=soil_order,
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

    ax_kd.set_yscale("log")
    ax_kd.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax_kd.set_xlabel("Soil", fontsize=12)
    ax_kd.set_ylabel(r"$K_{\mathrm{d}}\ [\mathrm{cm}^3/\mathrm{g}]$", fontsize=12)
    ax_kd.grid(True, which="major", axis="both", alpha=0.45)
    ax_kd.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax_kd.tick_params(axis="y", labelsize=10)
    for label in ax_kd.get_xticklabels():
        label.set_ha("right")

    fig_kd.tight_layout()
    fig_kd
    return


@app.cell
def _(all_pfas_results, pfas_names, plt, soil_db):
    soil_order_se = [s for s in soil_db.keys() if s.startswith("Staring-O")]

    def get_effective_saturation(soil_entry: dict) -> float:
        """Compute Se = (theta - theta_r) / (theta_s - theta_r) for one soil."""
        soil = soil_entry["soil"]
        theta = soil_entry["water"].water_content
        theta_r = soil["theta_r"]
        theta_s = soil["theta_s"]
        return (theta - theta_r) / (theta_s - theta_r)

    # Effective saturation doesn't depend on PFAS, so any single PFAS's
    # results give the full set of soils.
    soil_results_se = all_pfas_results[pfas_names[0]]

    se_by_soil = {
        soil_name: get_effective_saturation(soil_results_se[soil_name])
        for soil_name in soil_order_se
        if soil_name in soil_results_se
    }

    missing_se = [s for s in soil_order_se if s not in se_by_soil]
    if missing_se:
        raise ValueError(f"Missing soils in all_pfas_results: {missing_se}")

    fig_se, ax_se = plt.subplots(figsize=(10, 5))
    x_se = range(len(soil_order_se))
    ax_se.scatter(x_se, [se_by_soil[s] for s in soil_order_se], s=55)

    ax_se.set_xticks(x_se)
    ax_se.set_xticklabels(soil_order_se, rotation=45, ha="right")
    ax_se.set_xlabel("Soil", fontsize=12)
    ax_se.set_ylabel(r"Effective saturation $S_{\mathrm{e}}\ (-)$", fontsize=12)
    ax_se.set_ylim(0, 1.05)
    ax_se.grid(True, which="major", axis="both", alpha=0.45)
    ax_se.tick_params(axis="y", labelsize=10)

    fig_se.tight_layout()
    fig_se
    return


if __name__ == "__main__":
    app.run()
