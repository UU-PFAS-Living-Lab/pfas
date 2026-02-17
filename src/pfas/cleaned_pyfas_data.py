# cleaned_pyfas_data.py
# Auto‑generated PFAS, soil, and sorption parameter data
# All values are plain numbers or (value, "unit") tuples
# All nested structures are represented as tuples of (field, value) pairs
# All keys are strings
# No Pint, no classes, no calculations

"""
Usage:
from cleaned_pyfas_data import PFASs, soils, spa_matrix

PFASs.keys()
dict_keys(['TFA', 'PFBA', 'PFBS', 'PFPeA', 'PFHxA', 'PFHxS', 'PFHpA', 'PFOA', 'PFOS', 'PFNA', 'PFDA', 'HFPO-DA'])

soils.keys()
dict_keys(['Accusand', 'Vinton soil', 'Schoonenburgse Heuvel - sand', 'Schoonenburgse Heuvel - peat', 'Silva et al. (2020) - loam', 'Silva et al. (2020) - Loamy sand', 'Staring-B01', 'Staring-O01', 'Staring-O02', 'Staring-O03', 'Staring-O04', 'Staring-O05', 'Staring-O06', 'Staring-O07', 'Staring-O08', 'Staring-O09', 'Staring-O10', 'Staring-O11', 'Staring-O12', 'Staring-O13', 'Staring-O14', 'Staring-O15', 'Staring-O16', 'Staring-O17', 'Staring-O18'])

spa_matrix.keys()
dict_keys(['Accusand', 'Vinton soil'])
"""


PFASs = {
    "TFA": {
        "name": "TFA",
        "M": (114.02, "g/mol"),
        "diffusivity": (1, None),
        "K_oc": None,
        "K_sc": None,
        "K_mo": None,
        "n_CFx": 1,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFBA": {
        "name": "PFBA",
        "M": (214.0, "g/mol"),
        "diffusivity": (3, None),
        "K_oc": (2.9, "L/kg"),
        "K_sc": (0.43, "L/kg"),
        "K_mo": None,
        "n_CFx": 3,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFBS": {
        "name": "PFBS",
        "M": (300.1, "g/mol"),
        "diffusivity": (11e-6, "cm**2/s"),
        "K_oc": (11.0, "L/kg"),
        "K_sc": (0.44, "L/kg"),
        "K_mo": None,
        "n_CFx": 4,
        "n_COO": None,
        "n_SO3": 1,
        "n__O_": None
    },
    "PFPeA": {
        "name": "PFPeA",
        "M": (264.05, "g/mol"),
        "diffusivity": (12e-6, "cm**2/s"),
        "K_oc": (15.0, "L/kg"),
        "K_sc": (0.46, "L/kg"),
        "K_mo": None,
        "n_CFx": 4,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFHxA": {
        "name": "PFHxA",
        "M": (314.05, "g/mol"),
        "diffusivity": (7.8e-6, "cm**2/s"),
        "K_oc": (15.0, "L/kg"),
        "K_sc": (0.46, "L/kg"),
        "K_mo": None,
        "n_CFx": 5,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFHxS": {
        "name": "PFHxS",
        "M": (400.12, "g/mol"),
        "diffusivity": (4.5e-6, "cm**2/s"),
        "K_oc": (50.0, "L/kg"),
        "K_sc": (1.2, "L/kg"),
        "K_mo": None,
        "n_CFx": 6,
        "n_COO": None,
        "n_SO3": 1,
        "n__O_": None
    },
    "PFHpA": {
        "name": "PFHpA",
        "M": (364.06, "g/mol"),
        "diffusivity": (9.3e-6, "cm**2/s"),
        "K_oc": (50.0, "L/kg"),
        "K_sc": None,
        "K_mo": None,
        "n_CFx": 6,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFOA": {
        "name": "PFOA",
        "M": (414.07, "g/mol"),
        "diffusivity": (4.9e-6, "cm**2/s"),
        "K_oc": (107.0, "L/kg"),
        "K_sc": (3.3, "L/kg"),
        "K_mo": None,
        "n_CFx": 7,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFOS": {
        "name": "PFOS",
        "M": (500.13, "g/mol"),
        "diffusivity": (5.4e-6, "cm**2/s"),
        "K_oc": (609.0, "L/kg"),
        "K_sc": (9.4, "L/kg"),
        "K_mo": None,
        "n_CFx": 8,
        "n_COO": None,
        "n_SO3": 1,
        "n__O_": None
    },
    "PFNA": {
        "name": "PFNA",
        "M": (464.08, "g/mol"),
        "diffusivity": (2.93e-6, "cm**2/s"),
        "K_oc": (324.0, "L/kg"),
        "K_sc": (2.0, "L/kg"),
        "K_mo": None,
        "n_CFx": 8,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "PFDA": {
        "name": "PFDA",
        "M": (514.08, "g/mol"),
        "diffusivity": (2.27e-6, "cm**2/s"),
        "K_oc": (604.0, "L/kg"),
        "K_sc": (14.0, "L/kg"),
        "K_mo": None,
        "n_CFx": 9,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": None
    },
    "HFPO-DA": {
        "name": "HFPO-DA",
        "M": (330.05, "g/mol"),
        "diffusivity": (5, None),
        "K_oc": None,
        "K_sc": None,
        "K_mo": None,
        "n_CFx": 5,
        "n_COO": 1,
        "n_SO3": None,
        "n__O_": 1
    }
}

soils = {
    "Accusand": {
        "name": "Accusand",
        "rho_b": (1.65, "g/cm**3"),
        "porosity": 0.294,
        "theta_s": 0.294,
        "theta_r": 0.015,
        "K_sat": (2.0964e-2, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.04479, "1/cm")),
            ("n", 4.0),
            ("l", None),
        ),
        "tracer_fit": (
            ("x0", (633.96, "cm**2/cm**3")),
            ("x1", (-1182.5, "cm**2/cm**3")),
            ("x2", (548.54, "cm**2/cm**3")),
        ),
        "soil_roughness_multiplier": 4.15,
        "f_oc": (0.04, "percent"),
        "f_mo": ((14.0 + 2.5 + 12.0), "ug/g"),
        "f_clay": (0.0, "percent"),
        "f_silt": (0.0, "percent"),
    },

    "Vinton soil": {
        "name": "Vinton soil",
        "rho_b": (1.627, "g/cm**3"),
        "porosity": 0.395,
        "theta_s": 0.395,
        "theta_r": 0.056,
        "K_sat": (1.17e-3, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.02178, "1/cm")),
            ("n", 3.451),
            ("l", None),
        ),
        "tracer_fit": (
            ("x0", (1543.6, "cm**2/cm**3")),
            ("x1", (-2848.6, "cm**2/cm**3")),
            ("x2", (1305.0, "cm**2/cm**3")),
        ),
        "soil_roughness_multiplier": 4.15,
        "f_oc": (0.1, "percent"),
        "f_mo": (0.0, "percent"),
        "f_clay": (4.7, "percent"),
        "f_silt": (0.0, "percent"),
    },

    "Schoonenburgse Heuvel - sand": {
        "name": "Schoonenburgse Heuvel - sand",
        "rho_b": (1.5, "g/cm**3"),
        "porosity": 0.427,
        "theta_s": 0.427,
        "theta_r": 0.02,
        "K_sat": (3.61e-4, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.0217, "1/cm")),
            ("n", 1.735),
            ("l", None),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "dispersivity": (
            ("dispersivity", (5.0, "cm")),
        ),
    },

    "Schoonenburgse Heuvel - peat": {
        "name": "Schoonenburgse Heuvel - peat",
        "rho_b": (0.23, "g/cm**3"),
        "porosity": 0.85,
        "theta_s": 0.849,
        "theta_r": 0.01,
        "K_sat": (3.93519e-05, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.0119, "1/cm")),
            ("n", 1.272),
            ("l", None),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "dispersivity": (
            ("dispersivity", (10.0, "cm")),
        ),
    },

    "Silva et al. (2020) - loam": {
        "name": "Silva et al. (2020) - loam",
        "rho_b": (1.33, "g/cm**3"),
        "porosity": 0.47,
        "theta_s": 0.43,
        "theta_r": 0.078,
        "K_sat": (2.89e-4, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.036, "1/cm")),
            ("n", 1.56),
            ("l", None),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "dispersivity": (
            ("dispersivity", (35.0, "cm")),
        ),
    },

    "Silva et al. (2020) - Loamy sand": {
        "name": "Silva et al. (2020) - Loamy sand",
        "rho_b": (1.65, "g/cm**3"),
        "porosity": 0.44,
        "theta_s": 0.41,
        "theta_r": 0.057,
        "K_sat": (1.23e-3, "cm/s"),
        "van_genuchten": (
            ("alpha", (0.0124, "1/cm")),
            ("n", 2.28),
            ("l", None),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "dispersivity": (
            ("dispersivity", (35.0, "cm")),
        ),
    },
    "Staring-B01": {
        "name": "Staring-B01",
        "rho_b": (1.5, "g/cm**3"),
        "porosity": 0.427,
        "theta_s": 0.427,
        "theta_r": 0.02,
        "K_sat": (31.23, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0217, "1/cm")),
            ("n", 1.735),
            ("l", 0.981),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (5.0, "percent"),
        "f_oc": (7.5, "percent"),
    },

    "Staring-O01": {
        "name": "Staring-O01",
        "rho_b": (None, None),  # bulk_density_Poelman1974 cannot be computed
        "porosity": 0.366,
        "theta_s": 0.366,
        "theta_r": 0.01,
        "K_sat": (22.32, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0160, "1/cm")),
            ("n", 2.163),
            ("l", 2.868),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (5.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O02": {
        "name": "Staring-O02",
        "rho_b": (None, None),
        "porosity": 0.387,
        "theta_s": 0.387,
        "theta_r": 0.02,
        "K_sat": (22.76, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0161, "1/cm")),
            ("n", 1.524),
            ("l", 2.440),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (14.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O03": {
        "name": "Staring-O03",
        "rho_b": (None, None),
        "porosity": 0.340,
        "theta_s": 0.340,
        "theta_r": 0.01,
        "K_sat": (12.37, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0172, "1/cm")),
            ("n", 1.703),
            ("l", 0.0),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (25.5, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O04": {
        "name": "Staring-O04",
        "rho_b": (None, None),
        "porosity": 0.364,
        "theta_s": 0.364,
        "theta_r": 0.01,
        "K_sat": (25.81, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0136, "1/cm")),
            ("n", 1.488),
            ("l", 2.179),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (41.5, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O05": {
        "name": "Staring-O05",
        "rho_b": (None, None),
        "porosity": 0.337,
        "theta_s": 0.337,
        "theta_r": 0.01,
        "K_sat": (17.42, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0303, "1/cm")),
            ("n", 2.888),
            ("l", 0.074),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O06": {
        "name": "Staring-O06",
        "rho_b": (None, None),
        "porosity": 0.333,
        "theta_s": 0.333,
        "theta_r": 0.01,
        "K_sat": (32.83, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0160, "1/cm")),
            ("n", 1.289),
            ("l", -1.010),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (25.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O07": {
        "name": "Staring-O07",
        "rho_b": (None, None),
        "porosity": 0.513,
        "theta_s": 0.513,
        "theta_r": 0.01,
        "K_sat": (37.55, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0120, "1/cm")),
            ("n", 1.153),
            ("l", -2.013),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (41.5, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O08": {
        "name": "Staring-O08",
        "rho_b": (None, None),
        "porosity": 0.454,
        "theta_s": 0.454,
        "theta_r": 0.0,
        "K_sat": (8.64, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0113, "1/cm")),
            ("n", 1.346),
            ("l", -0.904),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (10.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O09": {
        "name": "Staring-O09",
        "rho_b": (None, None),
        "porosity": 0.458,
        "theta_s": 0.458,
        "theta_r": 0.0,
        "K_sat": (3.77, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0097, "1/cm")),
            ("n", 1.376),
            ("l", -1.013),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (15.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O10": {
        "name": "Staring-O10",
        "rho_b": (None, None),
        "porosity": 0.472,
        "theta_s": 0.472,
        "theta_r": 0.01,
        "K_sat": (2.30, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0100, "1/cm")),
            ("n", 1.246),
            ("l", -0.793),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (21.5, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O11": {
        "name": "Staring-O11",
        "rho_b": (None, None),
        "porosity": 0.444,
        "theta_s": 0.444,
        "theta_r": 0.0,
        "K_sat": (2.12, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0143, "1/cm")),
            ("n", 1.126),
            ("l", 2.357),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (30.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O12": {
        "name": "Staring-O12",
        "rho_b": (None, None),
        "porosity": 0.561,
        "theta_s": 0.561,
        "theta_r": 0.01,
        "K_sat": (1.08, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0088, "1/cm")),
            ("n", 1.158),
            ("l", -3.172),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (42.5, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O13": {
        "name": "Staring-O13",
        "rho_b": (None, None),
        "porosity": 0.573,
        "theta_s": 0.573,
        "theta_r": 0.01,
        "K_sat": (9.69, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0279, "1/cm")),
            ("n", 1.080),
            ("l", -6.091),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (75.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O14": {
        "name": "Staring-O14",
        "rho_b": (None, None),
        "porosity": 0.394,
        "theta_s": 0.394,
        "theta_r": 0.01,
        "K_sat": (2.50, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0033, "1/cm")),
            ("n", 1.617),
            ("l", -0.514),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (67.5, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O15": {
        "name": "Staring-O15",
        "rho_b": (None, None),
        "porosity": 0.410,
        "theta_s": 0.410,
        "theta_r": 0.01,
        "K_sat": (2.79, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0078, "1/cm")),
            ("n", 1.287),
            ("l", 0.000),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (92.5, "percent"),
        "f_oc": (1.5, "percent"),
    },

    "Staring-O16": {
        "name": "Staring-O16",
        "rho_b": (None, None),
        "porosity": 0.889,
        "theta_s": 0.889,
        "theta_r": 0.0,
        "K_sat": (1.46, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0097, "1/cm")),
            ("n", 1.364),
            ("l", -0.665),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (67.5, "percent"),
    },

    "Staring-O17": {
        "name": "Staring-O17",
        "rho_b": (None, None),
        "porosity": 0.849,
        "theta_s": 0.849,
        "theta_r": 0.01,
        "K_sat": (3.40, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0119, "1/cm")),
            ("n", 1.272),
            ("l", -1.249),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (67.5, "percent"),
    },

    "Staring-O18": {
        "name": "Staring-O18",
        "rho_b": (None, None),
        "porosity": 0.580,
        "theta_s": 0.580,
        "theta_r": 0.01,
        "K_sat": (35.97, "cm/day"),
        "van_genuchten": (
            ("alpha", (0.0127, "1/cm")),
            ("n", 1.316),
            ("l", -0.786),
        ),
        "tracer_fit": None,
        "soil_roughness_multiplier": 4.15,
        "f_clay": (0.0, "percent"),
        "f_silt": (0.0, "percent"),
        "f_oc": (25.0, "percent"),
    },
}

spa_matrix = {
    "Accusand": {
        "PFPeA": (
            ("Freundlich_K", (0.0211, "(mg/kg)/(mg/L)**0.87")),
            ("Freundlich_N", 0.87),
            ("frac_instant_adsorption", 0.4),
            ("kinetic_adsorption_rate", (5.9, "1/h")),
        ),
        "PFHxS": (
            ("Freundlich_K", (0.0213, "(mg/kg)/(mg/L)**0.81")),
            ("Freundlich_N", 0.81),
            ("frac_instant_adsorption", 0.1),
            ("kinetic_adsorption_rate", (3.1, "1/h")),
        ),
        "PFOA": (
            ("Freundlich_K", (0.1, "mg/kg/(mg/L)**0.87")),
            ("Freundlich_N", 0.87),
            ("frac_instant_adsorption", 0.4),
            ("kinetic_adsorption_rate", (5.9, "1/h")),
        ),
        "PFOS": (
            ("Freundlich_K", (0.15, "mg/kg/(mg/L)**0.81")),
            ("Freundlich_N", 0.81),
            ("frac_instant_adsorption", 0.1),
            ("kinetic_adsorption_rate", (3.1, "1/h")),
        ),
    },

    "Vinton soil": {
        "PFPeA": (
            ("Freundlich_K", (0.122, "(mg/kg)/(mg/L)**0.87")),
            ("Freundlich_N", 0.87),
            ("frac_instant_adsorption", 0.16),
            ("kinetic_adsorption_rate", (0.9, "1/h")),
        ),
        "PFHxS": (
            ("Freundlich_K", (0.156, "(mg/kg)/(mg/L)**0.77")),
            ("Freundlich_N", 0.77),
            ("frac_instant_adsorption", 0.16),
            ("kinetic_adsorption_rate", (0.9, "1/h")),
        ),
        "PFOA": (
            ("Freundlich_K", (0.58, "(mg/kg)/(mg/L)**0.87")),
            ("Freundlich_N", 0.87),
            ("frac_instant_adsorption", 0.16),
            ("kinetic_adsorption_rate", (0.9, "1/h")),
        ),
    }
}


