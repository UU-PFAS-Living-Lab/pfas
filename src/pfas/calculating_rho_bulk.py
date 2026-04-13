from pathlib import Path
import json

BASE_DIR = Path(__file__).parent

data_file = BASE_DIR / "data" / "soils_Ksat.json"
with open(data_file, "r") as f:
    soils = json.load(f)

def bulk_density(poro, f_c, f_s, f_oc):
    """
    Calculate bulk density (g/cm^3)

    Parameters
    ----------
    poro : float
        Porosity [-]
    f_c : float
        Clay fraction [%]
    f_c : float
        Silt fraction [%]
    f_oc : float
        Organic carbon fraction [%]    

    Returns
    -------
    float
        Bulk density (g/cm^3)
    """

    # convert percentages to fractions
    f_c  = f_c  / 100.0
    f_s  = f_s  / 100.0
    f_oc = f_oc / 100.0

    # bulk density
    rho_b = (1 - poro) * (1.47*f_oc + 2.88*f_c + 2.66*(1 - f_oc - f_c))
    
    return rho_b

for name, props in soils.items():
    poro  = props["porosity"]
    f_c   = props.get("f_clay", {}).get("value", 0.0)
    f_s   = props.get("f_silt", {}).get("value", 0.0)
    f_oc  = props.get("f_oc",   {}).get("value", 0.0)
    
    rho_b = bulk_density(poro, f_c, f_s, f_oc)
    print(f"{name}: {rho_b:.6f} g/cm³")