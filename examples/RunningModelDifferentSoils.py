import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running simulations for different soils
    In this example, we will showcase how we can run a simulation over 18 Dutch soil types for one PFAS and one set of hydrological parameter initialization.
    This notebook illustrates the differences in sorption between different soils.
    """)
    return


@app.cell
def _():
    #loading relevant modules 
    from pfas.component import LinearSPsorption, Le2021_langmuir, Szyszkowski, SWCsorption, Retardation, EquilibriumSolver, WaterPreprocessor, BoundaryPreprocessor, GridGenerator
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo
    import numpy as np
    from pfas.model import Model
    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        Le2021_langmuir,
        LinearSPsorption,
        Model,
        Retardation,
        SWCsorption,
        Szyszkowski,
        WaterPreprocessor,
        load_dataset,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining shared parameters betweenthe model classes

    re of this ourselves.
    In this example we will use PFOA as an example. We print some sorption information that is available for this compound ($K_{oc}$ and $K_{sc}$). This represents its sorption affinity to organic matter and silt and clay, respectively.
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
    K_oc      = pfas["K_oc"]["value"]   # L/kg
    K_sc      = pfas["K_sc"]["value"]   # L/kg

    # Surface tension of water
    sigma0=72.8
    T = 293.15

    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc} L/kg  |  K_sc = {K_sc} L/kg")


    return K_oc, K_sc, T, pfas, pfas_name, soil_db


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling different classes and running the model

    Now, we will call all the classes and generate our model. If we are missing parameters, the code will tell us.

    In this example we will mimick a situation in which PFAS is released for 25 years at a relatively low concentration. We run our model for 250 years in total. Sorption is assumed to be instantaneous and the solid-phase sorption coefficient ($K_d$) is calculated based on the following relationship:

    \[ K_d = K_{oc} \cdot f_{oc} + K_{sc} \cdot f_{silt\_clay} \]

    Because we only run the model for one type of PFAS  $K_{aw}$ is the same for all runs.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    EquilibriumSolver,
    GridGenerator,
    K_oc,
    K_sc,
    Le2021_langmuir,
    LinearSPsorption,
    Model,
    Retardation,
    SWCsorption,
    Szyszkowski,
    T,
    WaterPreprocessor,
    pfas,
    soil_db,
):
    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")]

    # If the order needs to be explicitly O01 → O18, use:
    # staring_soils = [f"Staring-O{i:02d}" for i in range(1, 19)]

    print(f"Running {len(staring_soils)} Staring soils:")
    print(staring_soils)

    # Store results for every soil
    all_soil_results = {}


    pulse_duration = 25 * (60 * 60 * 24 * 365)  # seconds
    for soil_name in staring_soils:
        model = Model()  

        soil        = soil_db[soil_name]
        bulk_dens   = soil["rho_b"]["value"]                   # g/cm3
        porosity    = soil["porosity"]                         # -
        theta_r     = soil["theta_r"]                          # -
        theta_s     = soil["theta_s"]                          # -
        K_sat       = soil["K_sat"]["value"]                   # cm/s
        vg_alpha    = soil["van_genuchten"]["alpha"]["value"]  # 1/cm
        vg_n        = soil["van_genuchten"]["n"]               # -
        vg_l        = soil["van_genuchten"]["l"]               # -
        dispersivity = 4.5                                     # cm
        f_oc        = soil["f_oc"]["value"] / 100              # - (fraction)
        f_clay      = soil["f_clay"]["value"] / 100            # - (fraction)
        f_silt      = soil["f_silt"]["value"] / 100            # - (fraction)
        f_silt_clay = f_silt + f_clay                          # - (fraction)
        d50         = soil["d50"]["value"] / 10000             # converted to cm

        # ---------------------------------------------------------------------
        # 1. Water / hydraulic properties
        # ---------------------------------------------------------------------
        model.compute(
            WaterPreprocessor,
            average_infiltration_rate=9.51e-7,
            hydraulic_conductivity=K_sat,
            porosity=porosity,
            dispersivity=dispersivity,
            van_genuchten_n=vg_n,
            van_genuchten_l=vg_l,
            residual_water_content=theta_r,
        )
        model.compute(GridGenerator,
        domain_length=100,                      # cm
        spatial_resolution=0.5,                 # cm
        time_resolution=(1/12) * (60*60*24*365),  # seconds
        time_total=250*(60*60*24*365),          # seconds
            )

        model.compute(BoundaryPreprocessor,
            C_list=[
                pfas["M"]["value"] * 1e-15,
                0.0,
            ],
            T_list=[
                0.0,
                pulse_duration,
            ],
        )

        # ---------------------------------------------------------------------
        # 3. Solid-phase adsorption
        # ---------------------------------------------------------------------
        sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "kinetic": {
                "frac_int": 1.0,
                "rate_const": 0.0,
            },
            "linear": {
                "Kd_method": "direct_input",
                "Kd": (
                    K_oc * f_oc
                    + K_sc * f_silt_clay
                ),
            },
        }

        model.compute(LinearSPsorption,
            sorption_solid=sorption_solid,
        )

        model.compute(
            SWCsorption,
            sigma0=71,
            scaling_factor_awi=4.15,
            van_genuchten_alpha=vg_alpha,
        )

        # ---------------------------------------------------------------------
        # 5. Select Kaw method
        #
        # Use Szyszkowski when the PFAS has the required parameters.
        # Otherwise use Le2021_langmuir.
        # ---------------------------------------------------------------------
        a = pfas["Szyszkowski_params"]["a"]["value"]
        b = pfas["Szyszkowski_params"]["b"]["value"]

        if a is not None and b is not None:
            model.compute(
                Szyszkowski,
                a=a,
                b=b,
                chi=1,
                T=T,
                Cw=1e-12,
            )
        else:
            model.compute(
                Le2021_langmuir,
                structural_properties=pfas["structural_properties"],  
                Cw=1e-12,
            )

        model.compute(
            Retardation,
            bulk_density=bulk_dens,
        )

        model.compute(
            EquilibriumSolver,
        )

        # ---------------------------------------------------------------------
        # Store all results for this soil
        # ---------------------------------------------------------------------
        print(type(model))

        all_soil_results[soil_name] = {
            "Aaw": model.aaw,
            "kawi": model.Kaw,
            "sim": model.C1,
            "water": model.hydro_properties,
            "soil": soil,
            "retardation": model.adsorption,
            "Kd": model.Kd,
            "grid": model.grid,
        }

        print(
            f"{soil_name} completed | "
            f"Kd = {model.Kd:.4g} | "
            f"Kaw = {model.Kaw:.4g}"
        )
    return all_soil_results, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results

    We print here our results for all soils. As we can observe, effective pore velocity and saturation levels differ over the soils, even though we used the same hydrological data. This is logical due to the varying soil properties. Because of this, the total air-water interfacial area ($A_{aw}$) is also different across soils.

    When we plot our breakthrough curves, we observe significant differences between the selected soils.
    """)
    return


@app.cell
def _(all_soil_results):
    def print_retardation_values(all_soil_results):
        for soil_name, results in all_soil_results.items():
            soil = results["soil"]
            hydro = results["water"]
            theta = hydro.water_content
            theta_s = soil["theta_s"]
            theta_r = soil["theta_r"]

            # Retardation object
            retardation = results["retardation"]

            # Derived quantities
            Se = (theta - theta_r) / (theta_s - theta_r)
            v = hydro.pore_velocity
            v_mm_yr = v * (60 * 60 * 24 * 365 * 10)

            print(f"\n{soil_name}")
            print(f"  v:   {v_mm_yr:.2f} mm/yr")
            print(f"  Se:  {Se:.2f} (-)")
            print(f"  Aaw: {results['Aaw']:.2f} cm2/cm3")
            print(f"  Kd:  {results['Kd']:.4e} cm3/g")
            print(f"  Solid-phase retardation: {retardation.sp_retardation:.4f}")
            print(f"  AWI retardation:         {retardation.awi_retardation:.4f}")

    print_retardation_values(all_soil_results)
    return


@app.cell
def _(all_soil_results, model, pfas, pfas_name):
    def plot_breakthrough(all_soil_results, pfas):
        import matplotlib.pyplot as plt
        import numpy as np

        selected_soils = ["Staring-O05", "Staring-O12", "Staring-O15", "Staring-O18"]
        soil_colours = {
            "Staring-O05": "red",
            "Staring-O12": "green",
            "Staring-O15": "blue",
            "Staring-O18": "brown",
        }

        t          = model.grid.time / (60 * 60 * 24 * 365)
        C_recharge = pfas['M']["value"] * 1e-15  # g/cm3, equivalent to 1 pmol/L

        fig, ax = plt.subplots(figsize=(5, 4))

        for soil in selected_soils:
            res = all_soil_results[soil]
            print(res)
            C1  =  res["sim"]
            ax.plot(t, C1[-1, :] / C_recharge, color=soil_colours[soil], label=soil)

        ax.set_ybound(0, 1)
        ax.set_ylabel("Caq/Crecharge (-)")
        ax.set_xlabel("Time (yrs)")
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{pfas_name}")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    plot_breakthrough(all_soil_results, pfas) 
    return


@app.cell
def _(all_soil_results, model, np, pfas, pfas_name, plt):
    def plot_concentration_profile(all_soil_results,pfas):

        selected_soils = [
            "Staring-O05", 
            "Staring-O12", 
            "Staring-O15", 
            "Staring-O18", 
        ]

        soil_colours = {
            "Staring-O05": "red",
            "Staring-O12": "green",
            "Staring-O15": "blue",
            "Staring-O18": "brown",
        }

        time       = model.grid.time / (60*60*24*365)
        depth      = model.grid.depth
        C_recharge = pfas['M']["value"] * 1e-15  # g/cm3, equivalent to 1 pmol/L
        average_infiltration_rate  =9.51E-7

        C_profile_time = 35 # years
        idx_35     = np.argmin(np.abs(time - C_profile_time))
        time_35    = time[idx_35]

        BTC_depth  = 100 # cm
        idx_100    = np.argmin(np.abs(depth - BTC_depth))
        depth_100  = depth[idx_100]

        fig, axes  = plt.subplots(1, 2, figsize=(8, 4))
        for soil in selected_soils:
            res    = all_soil_results[soil]
            C1     = res["sim"]           # mg/cm

            C_profile_35yr = C1[:, idx_35] / C_recharge

            axes[0].plot(C_profile_35yr, depth, color=soil_colours[soil], label=soil)
            axes[1].plot(time, C1[-1, :] / C_recharge, color=soil_colours[soil])

        axes[0].set_ylabel("Depth (cm)")
        axes[0].set_xlabel("Caq / Crecharge (-)")
        axes[0].invert_yaxis()
        axes[1].set_ylabel("Caq/Crecharge (-)")
        axes[1].set_xlabel("Time (yrs)")
        axes[0].set_title(f"Concentration profile for {pfas_name}, t = {C_profile_time:.0f} years")
        axes[1].set_title(f"Breakthrough curve for {pfas_name}, d = {depth_100:.0f} centimeters")
        for ax in axes:
            ax.grid(True)
            ax.title.set_size(10)
        fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()        

    plot_concentration_profile(all_soil_results, pfas)
    return


if __name__ == "__main__":
    app.run()
