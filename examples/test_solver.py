import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running simulations without TOML
    In this example, we will showcase how we can run a simulation without providing a TOML file.
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
        SWCAdsorptionPreprocessor
    )
    from pfas.component import LinearSPsorption, Le2021_langmuir, Szyszkowski, SWCsorption, Retardation, EquilibriumSolver
    from pfas.data_loader import load_dataset, available_datasets
    from matplotlib import pyplot as plt
    import marimo as mo
    import numpy as np

    print("Available datasets:", available_datasets())
    return (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        Le2021_langmuir,
        LinearSPsorption,
        Retardation,
        SWCsorption,
        Szyszkowski,
        WaterPreprocessor,
        load_dataset,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining shared parameters betweenthe model classes
    Some classes require the same parameters. When providing a TOML file, this is handled correctly, but when providing the parameters seperately to the different classes, we need to take care of this ourselves.

    In the next line of code, we will define bulk density. Furthermore, in our calling of the different classes, we will reuse some of the parameters that we have defined (`i.e. "residual_water_content": water_prep.residual_water_content` )
    """)
    return


@app.cell
def _(GridGenerator, load_dataset):
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


    from pfas.utils import kd_fabregat_palau
    # Step 1: generate grid
    grid_gen = GridGenerator(
        domain_length=100,                      # cm
        spatial_resolution=0.5,                 # cm
        time_resolution=(1/12) * (60*60*24*365),  # seconds
        time_total=250*(60*60*24*365),          # seconds
    )
    grid_results = grid_gen.compute()
    grid = grid_results["grid"]

    pulse_duration = 25 * (60 * 60 * 24 * 365)  # seconds
    return (
        K_oc,
        K_sc,
        T,
        grid,
        grid_results,
        pfas,
        pfas_name,
        pulse_duration,
        sigma0,
        soil_db,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling different classes and running the model

    Now, we will call all the classes and generate our model. If we are missing parameters, the code will tell us.
    """)
    return


@app.cell
def _(
    BoundaryPreprocessor,
    EquilibriumSolver,
    K_oc,
    K_sc,
    Le2021_langmuir,
    LinearSPsorption,
    Retardation,
    SWCsorption,
    Szyszkowski,
    T,
    WaterPreprocessor,
    grid,
    np,
    pfas,
    pulse_duration,
    sigma0,
    soil_db,
):
    staring_soils = [s for s in soil_db.keys() if s.startswith("Staring-O")
        ]

    # If the order needs to be explicitly O01 → O18, use:
    # staring_soils = [f"Staring-O{i:02d}" for i in range(1, 19)]

    print(f"Running {len(staring_soils)} Staring soils:")
    print(staring_soils)

    # Store results for every soil
    all_soil_results = {}

    # -------------------------------------------------------------------------
    # Loop over soils
    # -------------------------------------------------------------------------

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
        d50         = soil["d50"]["value"] / 10000    

        # ---------------------------------------------------------------------
        # 1. Water / hydraulic properties
        # ---------------------------------------------------------------------
        water_prep = WaterPreprocessor(
            average_infiltration_rate=9.51e-7,
            hydraulic_conductivity=K_sat,
            porosity=porosity,
            dispersivity=dispersivity,
            van_genuchten_n=vg_n,
            van_genuchten_l=vg_l,
            residual_water_content=theta_r,
        )

        water_results = water_prep.compute()

        hydro_properties = water_results["hydro_properties"]

        # ---------------------------------------------------------------------
        # 2. Boundary conditions
        # Same PFAS pulse for every soil
        # ---------------------------------------------------------------------
        boundary_prep = BoundaryPreprocessor(
            C_list=[
                pfas["M"]["value"] * 1e-15,
                0.0,
            ],
            T_list=[
                0.0,
                pulse_duration,
            ],
        )

        boundary_results = boundary_prep.compute()

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

        sorption = LinearSPsorption(
            sorption_solid=sorption_solid
        )

        spsorption_result = sorption.compute()

        # ---------------------------------------------------------------------
        # 4. Air-water interfacial area
        # ---------------------------------------------------------------------
        aaw_sorp = SWCsorption(
            hydro_properties=hydro_properties,
            sigma0=sigma0,
            scaling_factor_awi=4.15,
            soil={
                "porosity": porosity,
                "van_genuchten_alpha": vg_alpha,
                "van_genuchten_n": vg_n,
                "residual_water_content": theta_r,
            },
        )

        aaw_result = aaw_sorp.compute()
        aaw = aaw_result["aaw"]

        # ---------------------------------------------------------------------
        # 5. Select Kaw method
        #
        # Use Szyszkowski when the PFAS has the required parameters.
        # Otherwise use Le2021_langmuir.
        # ---------------------------------------------------------------------
        a = pfas["Szyszkowski_params"]["a"]["value"]
        b = pfas["Szyszkowski_params"]["b"]["value"]

        if a is not None and b is not None:

            kawi_sorp = Szyszkowski(
                sigma0=sigma0,
                a=a,
                b=b,
                chi=1,
                T=T,
            )

            kaw_method = "Szyszkowski"

        else:

            kawi_sorp = Le2021_langmuir(
                pfas["structural_properties"]
            )

            kaw_method = "Le2021_langmuir"

        kawi_results = kawi_sorp.compute(
            Cw=1e-12
        )

        # ---------------------------------------------------------------------
        # 6. Retardation
        # ---------------------------------------------------------------------
        ret = Retardation(
            Kd=spsorption_result["Kd"],
            Kaw=kawi_results["Kaw"],
            aaw=aaw,
            kinetic=False,
            bulk_density=bulk_dens,
            hydro_properties=hydro_properties,
        )

        ret_result = ret.compute()

        # ---------------------------------------------------------------------
        # 7. EquilibriumSolver
        #
        # IMPORTANT:
        # Every soil is now simulated using EquilibriumSolver.
        # ---------------------------------------------------------------------
        sim_runner = EquilibriumSolver(
            grid=grid,
            boundary_conditions=boundary_results["boundary_conditions"],
            hydro_properties=hydro_properties,
            adsorption=ret_result["adsorption"],
            initial_contaminant_concentration=np.zeros(
                len(grid.depth)
            ),
        )

        final_results = sim_runner.compute()

        # ---------------------------------------------------------------------
        # Store all results for this soil
        # ---------------------------------------------------------------------
        all_soil_results[soil_name] = {
            "Aaw": aaw,
            "kawi": kawi_results,
            "kaw_method": kaw_method,
            "sim": final_results,
            "water": water_results,
            "soil": soil,
            "retardation": ret_result["adsorption"],
            "Kd": spsorption_result["Kd"],
        }

        print(
            f"{soil_name} completed | "
            f"Kd = {spsorption_result['Kd']:.4g} | "
            f"Kaw = {kawi_results['Kaw']:.4g}"
        )

    return (all_soil_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting results
    """)
    return


@app.cell
def _(all_soil_results):
    def print_retardation_values(all_soil_results):
        for soil_name, results in all_soil_results.items():
            soil = results["soil"]
            hydro = results["water"]["hydro_properties"]
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
def _(all_soil_results, grid_results, pfas, pfas_name):
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
            print(res)
            C1  =  res["sim"]["C1"]
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


if __name__ == "__main__":
    app.run()
