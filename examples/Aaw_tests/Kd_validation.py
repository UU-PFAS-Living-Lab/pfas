import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Utilizing data structure
    This tutorial demonstrates how to use the data structure provided by the pfas package, which includes experimental data from peer-reviewed studies and soil property information for various soil types.
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
        SpRetardationPreprocessor,
        WaterPreprocessor,
        load_dataset,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Checking available data

    This code block shows how to access the data that is available in the PFAS data structure.
    """)
    return


@app.cell
def _(load_dataset):
    # ── Load datasets from the pfas library ───────────────────────────────────
    pfas_db = load_dataset("PFASs_neutral_forms")
    soil_db = load_dataset("soils_Ksat_rho_b_d50")

    # ── PFAS: PFOA ────────────────────────────────────────────────────────────
    pfas_name = "PFOS"
    pfas      = pfas_db[pfas_name]
    n_CFx     = pfas["n_CFx"]
    K_oc      = pfas["K_oc"]["value"]   # L/kg
    K_sc      = pfas["K_sc"]["value"]   # L/kg

    # ── Soil: Staring-O## ────────────────────────────────────────────────────────
    soil_name   = "Staring-O05"
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
    return (
        K_oc,
        K_sat,
        K_sc,
        bulk_dens,
        dispersivity,
        n_CFx,
        pfas,
        pfas_name,
        porosity,
        soil,
        soil_name,
        theta_r,
        vg_l,
        vg_n,
    )


@app.cell(hide_code=True)
def _(
    K_oc,
    K_sat,
    K_sc,
    bulk_dens,
    n_CFx,
    pfas_name,
    porosity,
    soil,
    soil_name,
    theta_r,
    vg_l,
    vg_n,
):
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
    return f_oc, f_silt_clay, frac_int, rate_const


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
    return (sp_results,)


@app.cell
def _(K_oc, K_sc, f_oc, f_silt_clay, sp_results):
    print(sp_results)

    print('kd is', K_oc*f_oc + K_sc*f_silt_clay)
    return


app._unparsable_cell(
    r"""
    In thesis H vd Berg, Koc and Ksc when available, otherwise kd_fabregat_palau
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
