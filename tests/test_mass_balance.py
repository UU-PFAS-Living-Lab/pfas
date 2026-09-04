def test_mass_balance_steady_state():
    from pfas.component import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        LinearSPsorption,
        Retardation,
        EquilibriumSolver,
    )
    from pfas.model import Model
    import math

    bulk_dens = 1580  # kg/m3

    model = Model()

    model.compute(GridGenerator,
                  domain_length=5,           # 5 m deep
                  spatial_resolution=0.025,  # 200 grid cells
                  time_resolution=0.1,
                  time_total=25)             # 25 yrs

    model.compute(WaterPreprocessor,
                  average_infiltration_rate=0.869,
                  hydraulic_conductivity=1763,
                  porosity=0.363,
                  dispersivity=24.39 / 100,
                  van_genuchten_n=2.72,
                  residual_water_content=0.054)

    # --- Check hydrological properties match expected values ---
    hydro = model.hydro_properties
    assert math.isclose(hydro.water_content, 0.10334785708782432, rel_tol=1e-6)
    assert math.isclose(hydro.pore_velocity, 8.408495584591847, rel_tol=1e-6)
    assert math.isclose(hydro.dispersion_coefficient, 2.0508320730819514, rel_tol=1e-6)

    model.compute(BoundaryPreprocessor,
                  C_list=[10.0e-6, 0],  # 10 ng/L in mg/L, then clean water
                  T_list=[0, 25])       # pulse from t=0 to t=25 yrs

    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "linear": {"Kd_method": "direct_input", "Kd": 5.3e-4},
    }
    model.compute(LinearSPsorption, sorption_solid=sorption_solid)

    # Effective Kaw for PFOS at the reference concentration
    a = 1.499900e-01   # mol/m3
    MW = 414           # g/mol
    C_rep_mgL = 10e-6
    C_rep = C_rep_mgL / MW
    kaw = 5.81e-06
    kaw_eff = kaw / (a + C_rep)

    model.compute(Retardation,
                  Kaw=kaw_eff,
                  aaw=34176,
                  bulk_density=bulk_dens)
    model.compute(EquilibriumSolver)

    assert model.C_tot is not None
    assert model.C_tot.shape[0] == model.grid.depth.size

    # --- Mass balance check at the base of the profile, steady state ---
    depth_idx = -1
    t_idx_ss = -1

    C1_ss = model.C1[depth_idx, t_idx_ss]
    C_tot_solver = model.C_tot[depth_idx, t_idx_ss]

    theta = hydro.water_content
    Kd = model.Kd
    Kaw_val = model.Kaw
    Aaw = model.aaw

    Cl = C1_ss * theta                 # mg/m3b
    Cs = Kd * bulk_dens * C1_ss        # mg/m3b
    Cawi = Kaw_val * Aaw * C1_ss       # mg/m3b
    C_tot_recomputed = Cl + Cs + Cawi

    Cl_m3w = C1_ss                     # mg/m3w
    Cs_kg = Kd * C1_ss                 # mg/kg

    # Expected values at depth = 5.0 m, t = 25.0 yr
    expected = {
        "C_tot_recomputed": 2.241e-05,
        "C_tot_solver": 2.241e-05,
        "Cl": 1.023e-06,
        "Cs": 8.285e-06,
        "Cawi": 1.310e-05,
        "Cl_m3w": 9.894e-06,
        "Cs_kg": 5.244e-09,
    }

    rel_tol = 1e-3  # matches 3-sig-fig precision of the reference values

    assert math.isclose(C_tot_recomputed, expected["C_tot_recomputed"], rel_tol=rel_tol)
    assert math.isclose(C_tot_solver, expected["C_tot_solver"], rel_tol=rel_tol)
    assert math.isclose(Cl, expected["Cl"], rel_tol=rel_tol)
    assert math.isclose(Cs, expected["Cs"], rel_tol=rel_tol)
    assert math.isclose(Cawi, expected["Cawi"], rel_tol=rel_tol)
    assert math.isclose(Cl_m3w, expected["Cl_m3w"], rel_tol=rel_tol)
    assert math.isclose(Cs_kg, expected["Cs_kg"], rel_tol=rel_tol)

    # Recomputed vs solver mass balance should match essentially exactly
    relative_mismatch = abs(C_tot_recomputed - C_tot_solver) / C_tot_solver
    assert relative_mismatch < 1e-6, (
        f"Mass balance mismatch too large: {relative_mismatch:.2%} "
        f"(recomputed={C_tot_recomputed:.6e}, solver={C_tot_solver:.6e})"
    )