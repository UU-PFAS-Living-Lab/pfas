@app.cell
def _(
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
    np,
):
    # -------------------------------------------------------------------------
    # Load PFAS and soil databases
    # -------------------------------------------------------------------------
    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")

    # -------------------------------------------------------------------------
    # Select PFAS
    # -------------------------------------------------------------------------
    pfas_name = "PFOA"
    pfas = pfas_db[pfas_name]

    n_CFx = pfas["structural_properties"]["n_CFx"]

    K_oc = pfas["K_oc"]["value"]
    K_sc = pfas["K_sc"]["value"]

    # -------------------------------------------------------------------------
    # Shared parameters
    # -------------------------------------------------------------------------
    sigma0 = 72.8
    T = 293.15

    dispersivity = 4.5

    pulse_duration = 25 * (60 * 60 * 24 * 365)

    # -------------------------------------------------------------------------
    # Grid is identical for all soils
    # -------------------------------------------------------------------------
    grid_gen = GridGenerator(
        domain_length=100,
        spatial_resolution=0.5,
        time_resolution=(1 / 12) * (60 * 60 * 24 * 365),
        time_total=250 * (60 * 60 * 24 * 365),
    )

    grid_results = grid_gen.compute()
    grid = grid_results["grid"]

    # -------------------------------------------------------------------------
    # Select all 18 Staring soils
    # -------------------------------------------------------------------------
    staring_soils = [
        s for s in soil_db.keys()
        if s.startswith("Staring-O")
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

        print(f"\n{'=' * 70}")
        print(f"Running soil: {soil_name}")
        print(f"{'=' * 70}")

        soil = soil_db[soil_name]

        # ---------------------------------------------------------------------
        # Soil properties
        # ---------------------------------------------------------------------
        bulk_dens = soil["rho_b"]["value"]
        porosity = soil["porosity"]

        theta_r = soil["theta_r"]
        theta_s = soil["theta_s"]

        K_sat = soil["K_sat"]["value"]

        vg_alpha = soil["van_genuchten"]["alpha"]["value"]
        vg_n = soil["van_genuchten"]["n"]
        vg_l = soil["van_genuchten"]["l"]

        f_oc = soil["f_oc"]["value"] / 100
        f_clay = soil["f_clay"]["value"] / 100
        f_silt = soil["f_silt"]["value"] / 100
        f_silt_clay = f_silt + f_clay

        d50 = soil["d50"]["value"] / 10000

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

    print(f"\nDone — {len(all_soil_results)} soils processed.")

    return all_soil_results, grid_results