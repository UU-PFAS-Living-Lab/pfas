import marimo

__generated_with = "0.18.4"
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
        #SorptionKawiDirectInput,
        #SorptionKawCalculated,
        #SorptionKawLangmuir,
        #SorptionKawSzyszkowski,
        SimulationRunner,
    )
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo
    import numpy as np

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        GridGenerator,
        SimulationRunner,
        SpRetardationPreprocessor,
        WaterPreprocessor,
        available_datasets,
        load_dataset,
        mo,
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

    # Solid-phase adsorption: linear
    frac_int = 1.0
    rate_const = 0.0

    print(f"PFAS : {pfas_name}  |  n_CFx = {n_CFx}")
    print(f"       K_oc = {K_oc} L/kg  |  K_sc = {K_sc} L/kg")
    return K_oc, K_sc, pfas, pfas_name, sigma0, soil_db


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Utilities from the original code (Aaw_mulitple_methods.py)
    """)
    return


@app.cell
def _(GridGenerator):
    from pfas.utils import kd_fabregat_palau
    # Step 1: generate grid
    grid_gen = GridGenerator(
        domain_length=100,                      # cm
        spatial_resolution=0.5,                 # cm
        time_resolution=(1/12) * (60*60*24*365),  # seconds
        time_total=250*(60*60*24*365),          # seconds
    )
    grid_results = grid_gen.compute()

    pulse_duration = 25 * (60 * 60 * 24 * 365)  # seconds


    return grid_results, pulse_duration


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
    K_oc,
    K_sc,
    SimulationRunner,
    SorptionKawLangmuir,
    SorptionKawSzyszkowski,
    SpRetardationPreprocessor,
    WaterPreprocessor,
    grid_results,
    kaw_input,
    pfas,
    pulse_duration,
    sigma0,
    soil_db,
):
    from pfas.utils import aaw_func_thermo
    # ── Loop over all 18 Staring soil types ──────────────────────────────────
    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")] 
    # Or define explicitly if order matters:
    # staring_soils = [f"Staring-O{i:02d}" for i in range(1, 19)]

    all_soil_results = {}  # keyed by soil_name

    for soil_name in staring_soils:
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
        d50         = soil["d50"]["value"] / 10000             # µm → cm

        # Step 2: Compute water flow / hydraulic properties
        water_prep = WaterPreprocessor(
            average_infiltration_rate=9.51E-7,                 # cm/s
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

        # Step 3: Boundary conditions (same pulse for all soils)
        boundary_prep = BoundaryPreprocessor(
            C_list=[pfas['M']["value"] * 1e-15, 0.0],           # g/cm3
            T_list=[0.0, pulse_duration]
        )
        boundary_results = boundary_prep.compute()

        # Step 4: Solid phase adsorption
        sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "kinetic": {"frac_int": 1.0, "rate_const": 0.0},
            "linear": {
                "Kd_method": "direct_input",
                # "Kd": kd_fabregat_palau(n_CFx, f_oc, f_silt_clay),
                "Kd": (K_oc*f_oc) + (K_sc*f_silt_clay),
            },
        }

        sp_prep = SpRetardationPreprocessor(
            sorption_solid=sorption_solid,
            bulk_density=bulk_dens,
            hydro_properties=water_results["hydro_properties"],
        )
        sp_results = sp_prep.compute()

        # Aaw-preprocessor
        func_thermo_Aaw = aaw_func_thermo(
            sigma0=sigma0, 
            poro=porosity, 
            alpha=vg_alpha, 
            n=vg_n,
            th=theta, 
            thr=theta_r, 
            ths=theta_s, 
            sf=4.15
        )

        # Select Kaw estimation method; use Szyszkowski parameters when available, otherwise use Langmuir
        def select_Kaw_method(pfas: dict, hydro_properties, aaw: float):
            a = pfas["Szyszkowski_params"]["a"]["value"]
            b = pfas["Szyszkowski_params"]["b"]["value"]

            if a is not None and b is not None:
                return SorptionKawSzyszkowski(
                    sigma0=sigma0,
                    a=a,
                    b=b,
                    hydro_properties=hydro_properties,
                    aaw=func_thermo_Aaw,
                    chi = 1,
                    T = 293.15,
                )
            else:
                return SorptionKawLangmuir(
                    **kaw_input,
                    hydro_properties=water_results["hydro_properties"],
                    aaw=func_thermo_Aaw,
                )

        # Kawi for each Aaw method
        kawi_sorp = select_Kaw_method(pfas, water_results["hydro_properties"], func_thermo_Aaw)
        kawi_results = kawi_sorp.compute(Cw=1e-12)

        # Simulations
        sim_runner = SimulationRunner(
            grid=grid_results["grid"],
            bulk_density=bulk_dens,
            boundary_conditions=boundary_results["boundary_conditions"],
            hydro_properties=water_results["hydro_properties"],
            awi_retardation=kawi_results["awi_retardation"],
            sorption_solid=sorption_solid,
            kinetic_sorption=False,
            volume_averaged=True,
        )
        final_results = sim_runner.compute()

        all_soil_results[soil_name] = {
            "Aaw": {"thermo": func_thermo_Aaw},
            "kawi": kawi_results,
            "sim": final_results,
            "water": water_results,
            "soil": soil,
            "sp_retardation": sp_results["sp_retardation"],
            "Kd": sp_results["Kd"],
        }

    print(f"\nDone — {len(all_soil_results)} soils processed.")
    return (all_soil_results,)


@app.cell
def _(all_soil_results):
    def print_retardation_values(all_soil_results):
        for soil_name, results in all_soil_results.items():
            soil      = results["soil"]
            hydro     = results["water"]["hydro_properties"]
            theta     = hydro.water_content
            theta_s   = soil["theta_s"]
            theta_r   = soil["theta_r"]

            # Derived quantities
            Se        = (theta - theta_r) / (theta_s - theta_r)  # effective saturation (-)
            v         = hydro.pore_velocity                      # cm/s
            Aaw       = results["Aaw"]["thermo"]                 # cm2/cm3
            v_mm_yr   = v * (60 * 60 * 24 * 365 * 10)            # mm/yr

            print(f"\n{soil_name}")
            print(f"  v:   {v_mm_yr:.2f} mm/yr")
            print(f"  Se:  {Se:.2f} (-)")
            print(f"  Aaw: {Aaw:.2f} cm2/cm3")
            print(f"  Kd:  {results['Kd']:.4e} cm3/g")
            print(f"  Solid-phase retardation: {results['sp_retardation']}")
            print(f"  AWI retardation:         {results['kawi']['awi_retardation']}")

    print_retardation_values(all_soil_results)
    return


@app.cell
def _(all_soil_results):
    print(dir(all_soil_results["Staring-O05"]["water"]["hydro_properties"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting BTC for all soils

    <!-- The generated_data allows you to access both input and output of the model and thus also allows for relative simple plotting. -->
    """)
    return


@app.cell
def _(all_soil_results, grid_results, pfas, pfas_name):
    # def plot_breakthrough(all_soil_results, grid_results, pfas):
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     selected_soils = ["Staring-O05", "Staring-O12", "Staring-O15", "Staring-O18"]

    #     soil_colours = {
    #         "Staring-O05": "red",
    #         "Staring-O12": "green",
    #         "Staring-O15": "blue",
    #         "Staring-O18": "brown",
    #     }

    #     t          = grid_results["grid"].time / (60 * 60 * 24 * 365)
    #     depth      = grid_results["grid"].depth
    #     C_recharge = pfas['M']["value"] * 1e-15  # g/cm3, equivalent to 1 pmol/L
    #     average_infiltration_rate  =9.51E-7

    #     fig, axes = plt.subplots(3, 1, figsize=(4, 9))    
    #     # fig.suptitle()

    #     for soil in selected_soils:
    #         res   = all_soil_results[soil]
    #         C1    = res["sim"]["C1"]            # mg/cm
    #         Kd    = res["Kd"]                   # cm3/g
    #         Kaw   = res["kawi"]["Kaw"]          # cm3/cm2
    #         Aaw   = res["Aaw"]["thermo"]        # cm2/cm3
    #         bulk_dens = res["soil"]["rho_b"]["value"]

    #         input_mass_per_area = average_infiltration_rate * C_recharge * pulse_duration

    #         Caw_bulk = Aaw * Kaw * C1 
    #         Cs_bulk = bulk_dens * Kd * C1

    #         Caw_normalised = np.trapezoid(Caw_bulk, depth, axis=0) / input_mass_per_area
    #         Cs_normalised  = np.trapezoid(Cs_bulk,  depth, axis=0) / input_mass_per_area
    #         Caq_normalised = C1[-1, :] / C_recharge

    #         axes[0].plot(t, Caw_normalised, color=soil_colours[soil], label=soil)
    #         axes[1].plot(t, Cs_normalised, color=soil_colours[soil])
    #         axes[2].plot(t, Caq_normalised, color=soil_colours[soil])

    #     for ax in axes[:3]:
    #         ax.set_ybound(0, 1)
    #     # for ax in axes:
    #     #     ax.set_xbound(0, 65)
    #     axes[0].set_ylabel("Caw / Crecharge")
    #     axes[1].set_ylabel("∫Cs/Crecharge over 0≤z≤1m (-)")
    #     axes[1].set_ybound(0,1)
    #     axes[2].set_ylabel("Caq/Crecharge (-)")
    #     axes[2].set_xlabel("Time (yrs)")
    #     axes[0].set_title(f"Breakthrough curves for {pfas_name}")
    #     axes[0].legend()

    #     for ax in axes:
    #         ax.grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # plot_breakthrough(all_soil_results, grid_results, pfas)

    def plot_breakthrough(all_soil_results, grid_results, pfas):
        import matplotlib.pyplot as plt
        import numpy as np

        selected_soils = ["Staring-O05", "Staring-O12", "Staring-O15", "Staring-O18"]
        soil_colours = {
            "Staring-O05": "red",
            "Staring-O12": "green",
            "Staring-O15": "blue",
            "Staring-O18": "brown",
        }

        t          = grid_results["grid"].time / (60 * 60 * 24 * 365)
        C_recharge = pfas['M']["value"] * 1e-15  # g/cm3, equivalent to 1 pmol/L

        fig, ax = plt.subplots(figsize=(5, 4))

        for soil in selected_soils:
            res = all_soil_results[soil]
            C1  = res["sim"]["C1"]  # mg/cm
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

    plot_breakthrough(all_soil_results, grid_results, pfas) 
    return


@app.cell
def _(all_soil_results, grid_results, pfas, pfas_name):
    def plot_concentration_profile(all_soil_results, grid_results, pfas):
        import matplotlib.pyplot as plt
        import numpy as np

        # selected_soils = [
        #     "Staring-O01", 
        #     "Staring-O03", 
        #     "Staring-O05", 
        #     "Staring-O09", 
        #     "Staring-O12"]

        # soil_colours   = {
        #     "Staring-O01": "red",
        #     "Staring-O03": "green",
        #     "Staring-O05": "blue",
        #     "Staring-O09": "brown",
        #     "Staring-O12": "orange",
        # }

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

        time       = grid_results["grid"].time / (60*60*24*365)
        depth      = grid_results["grid"].depth
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
            C1     = res["sim"]["C1"]            # mg/cm

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
        axes[0].legend()
        for ax in axes:
            ax.grid(True)

        plt.tight_layout()
        plt.show()        

    plot_concentration_profile(all_soil_results, grid_results, pfas)
    return


if __name__ == "__main__":
    app.run()
