def test_full_simulation():
    from pfas.component import WaterPreprocessor, BoundaryPreprocessor, GridGenerator
    from pfas.component import SWCsorption, LinearSPsorption, Retardation, EquilibriumSolver
    from pfas.model import Model

    model = Model()

    model.compute(GridGenerator,
                  domain_length=60,
                  spatial_resolution=1.0,
                  time_resolution=100,
                  time_total=10000)

    model.compute(WaterPreprocessor,
                  average_infiltration_rate=1.5,
                  hydraulic_conductivity=6,
                  porosity=0.34,
                  dispersivity=1.5,
                  van_genuchten_n=1.31,
                  residual_water_content=0.04)

    model.compute(BoundaryPreprocessor,
                  C_list=[10.0, 0],
                  T_list=[0, 2000])

    sorption_solid = {
        "kinetic_sorption": False,
        "sorption_isotherm": "linear",
        "kinetic": {"frac_int": 0.3, "rate_const": 0.01},
        "linear": {"Kd_method": "direct_input", "Kd": 5.0},
    }

    model.compute(LinearSPsorption, sorption_solid=sorption_solid)
    model.compute(SWCsorption, sigma0=71, scaling_factor_awi=1.0, van_genuchten_alpha=0.019)
    model.compute(Retardation, Kaw=0.5, bulk_density=1.6)
    model.compute(EquilibriumSolver)

    assert model.C_tot is not None
    assert model.C_tot.shape[0] == model.grid.depth.size
