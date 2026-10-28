import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Utilizing data structure
    This tutorial demonstrates how to use the data structure provided by the pfas package, which includes experimental data from peer-reviewed studies and soil property information for various soil types. We also demonstrate the functionality of the built-in unit converter.
    """)
    return


@app.cell
def _():
    #loading relevant modules 

    from pfas.model import Model
    from matplotlib import pyplot as plt
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Checking available data

    This code block shows how to access the data that is available in the PFAS data structure.
    """)
    return


@app.cell
def _():
    from pfas.data_loader import load_dataset

    PFASs = load_dataset("PFASs")
    soils = load_dataset("soils")
    spa_matrix = load_dataset("spa_matrix")
    # See what's available
    print("Available PFAS compounds:")
    print(list(PFASs.keys()))

    print("\nAvailable soils:")
    print(list(soils.keys()))

    print("\nSoils with sorption parameter data (spa_matrix):")
    print(list(spa_matrix.keys()))
    return PFASs, soils, spa_matrix


@app.cell
def _(PFASs, soils, spa_matrix):
    # Pick a compound and soil for this run
    pfas_name = "PFOA"
    soil_name = "Accusand"

    pfas = PFASs[pfas_name]
    soil = soils[soil_name]

    # Inspect PFAS properties
    print(f"\nMolar mass       : {pfas['M']}")
    print(f"K_oc             : {pfas['K_oc']}")
    print(f"Diffusivity      : {pfas['diffusivity']}")

    # Inspect soil properties
    print(f"\nBulk density     : {soil['rho_b']}")
    print(f"Porosity         : {soil['porosity']}")
    print(f"K_sat            : {soil['K_sat']}")

    # Unpack van Genuchten parameters (stored as tuple of (field, value) pairs)
    vg_params = dict(soil["van_genuchten"])   # convert to dict for easy access
    print("\nVan Genuchten parameters:")
    print(vg_params)

    # Pull scalar soil values used in the simulation
    bulk_dens   = soil["rho_b"]
    porosity    = soil["porosity"]
    vg_n        = vg_params["n"]
    theta_r     = soil["theta_r"]
    vg_alpha    = vg_params["alpha"]
    dispersivity = 1.5                       # not present for Accusand, use default
    C_rep = 1 #indication of nonlinearity for freundlich sorption, can be between 0 and 1
    # Check for solid phase adsorption parameters available in the dataset:
    if soil_name in spa_matrix and pfas_name in spa_matrix[soil_name]:
        spa = dict(spa_matrix[soil_name][pfas_name])
        freundlich_k = spa["Freundlich_K"]
        freundlich_n = spa["Freundlich_N"]
        frac_int = spa["frac_instant_adsorption"]
        rate_const = spa["kinetic_adsorption_rate"]
        print(f"\nSorption parameters (spa_matrix) for {pfas_name} in {soil_name}:")
        print(f"  Freundlich K : {freundlich_k}")
        print(f"  Freundlich N : {freundlich_n}")
        print(f"  Frac instant : {frac_int}")
        print(f"  Kinetic rate : {rate_const} ")
        use_spa = True
    else:
        print(f"\nNo spa_matrix entry for {pfas_name} in {soil_name}. Using fallback Kd.")
        use_spa = False
    return pfas, rate_const, soil, spa, vg_params


@app.cell
def _(mo):
    mo.md(r"""
    ## Converting Units
    Within the *pfas* package it is important to be consistent with units to ensure the right results. To this end, all data in the database comes with a unit, as is shown in the code above.
    It is not mandatory to keep track of your units, but strongly recommended. To help with converting, we use the [Pint](https://pint.readthedocs.io/) unit registry. Please refer to this documentation for additional options.
    """)
    return


@app.cell
def _(pfas, rate_const, soil, vg_params):

    molar_mass_si = pfas["M"].to_base_units()
    K_oc_si = pfas["K_oc"].to_base_units()
    diffusivity_si = pfas["diffusivity"].to_base_units()

    bulk_dens_si = soil["rho_b"].to_base_units()
    K_sat_si = soil["K_sat"].to_base_units()
    vg_alpha_si = vg_params["alpha"].to_base_units()

    rate_const_si = rate_const.to_base_units()
    return (
        K_oc_si,
        K_sat_si,
        bulk_dens_si,
        diffusivity_si,
        molar_mass_si,
        rate_const_si,
        vg_alpha_si,
    )


@app.cell
def _(
    K_oc_si,
    K_sat_si,
    bulk_dens_si,
    diffusivity_si,
    mo,
    molar_mass_si,
    pfas,
    rate_const_si,
    soil,
    spa,
    vg_alpha_si,
    vg_params,
):
    import pandas as pd

    unit_table = pd.DataFrame([
        {
            "Parameter": "Molar mass",
            "Old value": pfas["M"].magnitude,
            "Old unit": pfas["M"].units,
            "New value": molar_mass_si.magnitude,
            "New unit": molar_mass_si.units,
        },
        {
            "Parameter": "Koc",
            "Old value": pfas["K_oc"].magnitude,
            "Old unit": pfas["K_oc"].units,
            "New value": K_oc_si.magnitude,
            "New unit": K_oc_si.units,
        },
        {
            "Parameter": "Diffusivity",
            "Old value": pfas["diffusivity"].magnitude,
            "Old unit": pfas["diffusivity"].units,
            "New value": diffusivity_si.magnitude,
            "New unit": diffusivity_si.units,
        },
        {
            "Parameter": "Bulk density",
            "Old value": soil["rho_b"].magnitude,
            "Old unit": soil["rho_b"].units,
            "New value": bulk_dens_si.magnitude,
            "New unit": bulk_dens_si.units,
        },
        {
            "Parameter": "Ksat",
            "Old value": soil["K_sat"].magnitude,
            "Old unit": soil["K_sat"].units,
            "New value": K_sat_si.magnitude,
            "New unit": K_sat_si.units,
        },
        {
            "Parameter": "VG alpha",
            "Old value": vg_params["alpha"].magnitude,
            "Old unit": vg_params["alpha"].units,
            "New value": vg_alpha_si.magnitude,
            "New unit": vg_alpha_si.units,
        },
        {
            "Parameter": "Kinetic rate",
            "Old value": spa["kinetic_adsorption_rate"].magnitude,
            "Old unit": spa["kinetic_adsorption_rate"].units,
            "New value": rate_const_si.magnitude,
            "New unit": rate_const_si.units,
        },
    ])

    mo.ui.table(unit_table)
    return


if __name__ == "__main__":
    app.run()
