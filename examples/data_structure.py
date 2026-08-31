import marimo

__generated_with = "0.18.4"
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
    #loading relevant modules 
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner
    from pfas.configuration import read_toml
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
    bulk_dens   = soil["rho_b"]  ["value"]         # numeric value only (g/cm³)
    porosity    = soil["porosity"]
    vg_n        = vg_params["n"]
    theta_r     = soil["theta_r"]
    vg_alpha    = vg_params["alpha"]["value"]    # numeric value (1/cm)
    dispersivity = 1.5                       # not present for Accusand, use default
    C_rep = 1 #indication of nonlinearity for freundlich sorption, can be between 0 and 1
    # Check for solid phase adsorption paraeters available:
    if soil_name in spa_matrix and pfas_name in spa_matrix[soil_name]:
        spa = dict(spa_matrix[soil_name][pfas_name])
        freundlich_k = spa["Freundlich_K"]["value"]   # numeric value
        freundlich_n = spa["Freundlich_N"]
        frac_int = spa["frac_instant_adsorption"]
        rate_const = spa["kinetic_adsorption_rate"]
        print(f"\nSorption parameters (spa_matrix) for {pfas_name} in {soil_name}:")
        print(f"  Freundlich K : {freundlich_k}")
        print(f"  Freundlich N : {freundlich_n}")
        print(f"  Frac instant : {frac_int}")
        print(f"  Kinetic rate : {rate_const} 1/h")
        use_spa = True
    else:
        print(f"\nNo spa_matrix entry for {pfas_name} in {soil_name}. Using fallback Kd.")
        use_spa = False
    return


if __name__ == "__main__":
    app.run()
