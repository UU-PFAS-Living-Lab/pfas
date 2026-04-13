import marimo

__generated_with = "0.19.11"
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
    from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor, GridGenerator, SpRetardationPreprocessor, SWCAdsorptionPreprocessor, SorptionKawiDirectInput, SimulationRunner, SorptionKawCalculated
    from pfas.configuration import read_toml
    from pfas.model import Model
    from matplotlib import pyplot as plt
    import marimo as mo

    return (
        GridGenerator,
        SWCAdsorptionPreprocessor,
        SorptionKawCalculated,
        WaterPreprocessor,
        mo,
        plt,
    )


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

    PFASs = load_dataset("PFASs_neutral_forms")
    soils = load_dataset("soils")
    spa_matrix = load_dataset("spa_matrix")
    # See what's available
    print("Available PFAS compounds:")
    print(list(PFASs.keys()))

    print("\nAvailable soils:")
    print(list(soils.keys()))

    print("\nSoils with sorption parameter data (spa_matrix):")
    print(list(spa_matrix.keys()))
    return PFASs, soils


@app.cell
def _(soils):
    # Pick a compound and soil for this run
    soil_name = "Accusand"

    soil = soils[soil_name]

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
    vg_l        = 0.5
    theta_r     = soil["theta_r"]
    vg_alpha    = vg_params["alpha"]["value"]    # numeric value (1/cm)
    dispersivity = 1.5                       # not present for Accusand, use default
    C_rep = 1 #indication of nonlinearity for freundlich sorption, can be between 0 and 1
    return (
        bulk_dens,
        dispersivity,
        porosity,
        soil,
        theta_r,
        vg_alpha,
        vg_l,
        vg_n,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Running Simulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Testing SorptionKawCalculated
    """)
    return


@app.cell
def _(
    GridGenerator,
    PFASs,
    SWCAdsorptionPreprocessor,
    SorptionKawCalculated,
    WaterPreprocessor,
    bulk_dens,
    dispersivity,
    porosity,
    soil,
    theta_r,
    vg_alpha,
    vg_l,
    vg_n,
):
    ## Running simulation 
    from pfas.utils import kd_freundlich, kaw_Le2021

    # Step 1: Generate the grid
    grid_gen = GridGenerator(
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=5000,
    )
    grid_results = grid_gen.compute()

    # Step 2: Compute water flow / hydraulic properties
    water_prep = WaterPreprocessor(
        average_infiltration_rate=1.5,
        hydraulic_conductivity=soil["K_sat"]["value"],
        porosity=porosity,
        dispersivity=dispersivity,
        van_genuchten_n=vg_n,
        van_genuchten_l=vg_l,
        init_sat=0.2,
        residual_water_content=theta_r,
    )
    water_results = water_prep.compute()

    # Step 3: Air-water interface (AWI) adsorption
    swc_adsorp = SWCAdsorptionPreprocessor(
        hydro_properties=water_results["hydro_properties"],
        sigma0=71,
        scaling_factor_awi=1.0,
        AWI={
            "AWI_type": "SWC-based",
            "SWC-based": {"scaling_factor_awi": 4.15},
        },
        soil={
            "bulk_density": bulk_dens,
            "porosity": water_prep.porosity,
            "van_genuchten_alpha": vg_alpha,
            "van_genuchten_n": water_prep.van_genuchten_n,
            "saturated_water_content": porosity,
            "residual_water_content": water_prep.residual_water_content,
            "hydraulic_conductivity": water_prep.hydraulic_conductivity,
            "dispersivity": water_prep.dispersivity,
        },
    )
    awi_results = swc_adsorp.compute()

    # Step 4: Kaw for all PFAS compounds
    kaw_results = {}
    for pfas_name, pfas in PFASs.items():

        n_CFx  = pfas.get("n_CFx")  or 0
        n_CHx  = pfas.get("n_CHx")  or 0
        n_COO  = pfas.get("n_COO")  or 0
        n_COOH = pfas.get("n_COOH") or 0
        n_SO3  = pfas.get("n_SO3")  or 0
        n_R4N  = pfas.get("n_R4N")  or 0
        n_OH   = pfas.get("n_OH")   or 0
        n_OSO3 = pfas.get("n_OSO3") or 0
        n__O_  = pfas.get("n__O_")  or 0
        n__S_  = pfas.get("n__S_")  or 0

        kawi_sorp = SorptionKawCalculated(
            n_CFx=n_CFx, n_CHx=n_CHx, n_COO=n_COO, n_COOH=n_COOH,
            n_SO3=n_SO3, n_R4N=n_R4N, n_OH=n_OH, n_OSO3=n_OSO3,
            n__O_=n__O_, n__S_=n__S_,
            hydro_properties=water_results["hydro_properties"],
            aaw=awi_results["aaw"],
        )
        kaw_results[pfas_name] = kawi_sorp.kaw

    for pfas_name, kaw in kaw_results.items():
        print(f"{pfas_name:10s}  Kaw = {kaw:.4e}")
    return awi_results, grid_results, kawi_sorp, water_results


@app.cell
def _(awi_results, kawi_sorp, water_results):


    print("Computed Kaw:", kawi_sorp.kaw)
    print("Aaw:", awi_results["aaw"])
    print("Theta:", water_results["hydro_properties"].water_content)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Plotting Results
    """)
    return


@app.cell
def _(final_results, grid_results, plt):
    simulation_grid = grid_results["grid"]
    # Select specific time indices to plot
    t_len = final_results['C_tot'].shape[1]
    time_indices = [0, t_len//4, t_len//2, 3*t_len//4, -1]  # First, and some intermediate, and last time step

    plt.figure(figsize=(8, 6))

    for t_idx in time_indices:
        plt.plot(final_results['C_tot'][:, t_idx], simulation_grid.depth, label=f"t = {simulation_grid.time[t_idx]:.0f} s")

    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
