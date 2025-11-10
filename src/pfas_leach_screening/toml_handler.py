# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:22:20 2025

@author: 6346650
"""

import toml
from pathlib import Path

def read_toml(path: Path) -> dict:
    toml_file = toml.load(path) 
    return toml_file

def check_experimental_conditions(experimental_conditions_dict: dict) -> bool: 
    """ Validate experimental conditions from TOML config """
    

def check_toml_experimental_conditions(experimental_conditions_dict: dict) -> bool: 
    """ Validate grid parameters from TOML config """
    
    required_keys = ['domain_length', 'spatial_resolution', 'time_total', 'soil_temp']
    # Check if all required keys exist
    for key in required_keys:
        if key not in experimental_conditions_dict:
            print(f"Missing required key: {key}")
            return False
    
    # Check if all values are numeric
    for key in required_keys:
        value = experimental_conditions_dict[key]
        if not isinstance(value, (int, float)):
            print(f"Value for '{key}' must be numeric, got {type(value).__name__}")
            return False
        
        # Check for positive values
        if value <= 0:
            print(f"Value for '{key}' must be positive, got {value}")
            return False
    
    #initial conditions 
    initial_params = experimental_conditions_dict['initial']
    required_initial_keys = ['init_sat', 'initial_solute_concentration'] #TODO initial solute concentration should be one value or vector, how to do it?
    for key in required_initial_keys:
        if key not in initial_params:
            print(f"Missing required key: {key}")
            return False
    boundary_params = experimental_conditions_dict['boundary']
    required_boundary_keys = ["average_infiltration_rate", "pulse_duration", "solute_concentration_influx"]
    for key in required_boundary_keys:
        if key not in boundary_params:
            print(f"Missing required key: {key}")
            return False
    for key in required_boundary_keys:
        value = boundary_params[key]
        if not isinstance(value, (int, float)):
            print(f"Value for '{key}' must be numeric, got {type(value).__name__}")
            return False
        # Check for positive values
        if value <= 0:
            print(f"Value for '{key}' must be positive, got {value}")
            return False
            
    
    
    return True

def check_toml_soil(soil_dict: dict) -> bool:
    """Validate soil parameters from TOML config."""
    
    expected_keys = [
        "soil_name", "soil_type", "bulk_density", "porosity", 
        "van_genuchten_alpha", "van_genuchten_n", "saturated_water_content",
        "hydraulic_conductivity", "dispersivity"
    ]
    
    # Check all keys present
    if set(soil_dict.keys()) != set(expected_keys):
        missing = set(expected_keys) - set(soil_dict.keys())
        extra = set(soil_dict.keys()) - set(expected_keys)
        if missing:
            print(f"Missing keys: {missing}")
        if extra:
            print(f"Extra keys: {extra}")
        return False
    
    # Check string fields
    if not isinstance(soil_dict["soil_name"], str):
        print("soil_name must be a string")
        return False
    if not isinstance(soil_dict["soil_type"], str):
        print("soil_type must be a string")
        return False
    
    # Check numeric fields are correct type
    numeric_fields = [
        "bulk_density", "porosity", "van_genuchten_alpha", 
        "van_genuchten_n", "saturated_water_content",
        "hydraulic_conductivity", "dispersivity"
    ]
    
    for field in numeric_fields:
        if not isinstance(soil_dict[field], (int, float)):
            print(f"{field} must be numeric, got {type(soil_dict[field]).__name__}")
            return False
    
    # Range checks
    if not (0 < soil_dict["porosity"] <= 1):
        print(f"porosity must be between 0 and 1, got {soil_dict['porosity']}")
        return False
    if not (0 < soil_dict["saturated_water_content"] <= 1):
        print(f"saturated_water_content must be between 0 and 1, got {soil_dict['saturated_water_content']}")
        return False
    
    return True

def check_toml_sorption(sorption_dict: dict) -> bool:
    """ Validate sorption parameters from TOML config """
    
    # Check main sorption_solid section
    if 'kinetic_sorption' not in sorption_dict:
        print("Missing required key: kinetic_sorption")
        return False
    
    if not isinstance(sorption_dict['kinetic_sorption'], bool):
        print(f"'kinetic_sorption' must be a boolean, got {type(sorption_dict['kinetic_sorption']).__name__}")
        return False
    
    if 'sorption_isotherm' not in sorption_dict:
        print("Missing required key: sorption_isotherm")
        return False
    
    valid_isotherms = ['linear', 'freundlich', 'langmuir']
    if sorption_dict['sorption_isotherm'] not in valid_isotherms:
        print(f"'sorption_isotherm' must be one of {valid_isotherms}, got '{sorption_dict['sorption_isotherm']}'")
        return False
    
    # Check kinetic parameters if kinetic=true
    if sorption_dict['kinetic']:
        if 'kinetic' not in sorption_dict:
            print("Missing [sorption_solid.kinetic] section when kinetic=true")
            return False
        
        kinetic_params = sorption_dict['kinetic']
        
        if 'frac_int' not in kinetic_params:
            print("Missing required parameter: frac_int")
            return False
        if not isinstance(kinetic_params['frac_int'], (int, float)):
            print(f"'frac_int' must be numeric, got {type(kinetic_params['frac_int']).__name__}")
            return False
        if not 0 <= kinetic_params['frac_int'] <= 1:
            print(f"'frac_int' must be between 0 and 1, got {kinetic_params['frac_int']}")
            return False
        
        if 'rate_const' not in kinetic_params:
            print("Missing required parameter: rate_const")
            return False
        if not isinstance(kinetic_params['rate_const'], (int, float)):
            print(f"'rate_const' must be numeric, got {type(kinetic_params['rate_const']).__name__}")
            return False
        if kinetic_params['rate_const'] <= 0:
            print(f"'rate_const' must be positive, got {kinetic_params['rate_const']}")
            return False
    
    # Check the specific isotherm section
    isotherm = sorption_dict['sorption_isotherm']
    if isotherm not in sorption_dict:
        print(f"Missing section for isotherm: [sorption_solid.{isotherm}]")
        return False
    
    isotherm_params = sorption_dict[isotherm]
    
    # Validate based on isotherm type
    if isotherm == 'linear':
        if 'Kd_method' not in isotherm_params:
            print("Missing required key: Kd_method")
            return False
        
        valid_methods = ['direct_input', 'organic_mineral', 'Fabregat_Palau2021']
        kd_method = isotherm_params['Kd_method']
        
        if kd_method not in valid_methods:
            print(f"'Kd_method' must be one of {valid_methods}, got '{kd_method}'")
            return False
        
        # Check method-specific parameters
        if kd_method == 'direct_input':
            if 'Kd' not in isotherm_params:
                print("Missing required parameter: Kd")
                return False
            if not isinstance(isotherm_params['Kd'], (int, float)):
                print(f"'Kd' must be numeric, got {type(isotherm_params['Kd']).__name__}")
                return False
            if isotherm_params['Kd'] <= 0:
                print(f"'Kd' must be positive, got {isotherm_params['Kd']}")
                return False
        
        elif kd_method == 'organic_mineral':
            required = ['OC_perc', 'Koc', 'Min_perc', 'Kmin']
            for param in required:
                if param not in isotherm_params:
                    print(f"Missing required parameter: {param}")
                    return False
                if not isinstance(isotherm_params[param], (int, float)):
                    print(f"'{param}' must be numeric, got {type(isotherm_params[param]).__name__}")
                    return False
                if isotherm_params[param] <= 0:
                    print(f"'{param}' must be positive, got {isotherm_params[param]}")
                    return False
        
        elif kd_method == 'Fabregat_Palau2021':
            required = ['OC_perc', 'Min_perc', 'chain_length']
            for param in required:
                if param not in isotherm_params:
                    print(f"Missing required parameter: {param}")
                    return False
                if not isinstance(isotherm_params[param], (int, float)):
                    print(f"'{param}' must be numeric, got {type(isotherm_params[param]).__name__}")
                    return False
                if isotherm_params[param] <= 0:
                    print(f"'{param}' must be positive, got {isotherm_params[param]}")
                    return False
            
            if not isinstance(isotherm_params['chain_length'], int):
                print(f"'chain_length' must be an integer, got {type(isotherm_params['chain_length']).__name__}")
                return False
        
        # Check optional c_non_lin if present
        if 'c_non_lin' in isotherm_params:
            if not isinstance(isotherm_params['c_non_lin'], (int, float)):
                print(f"'c_non_lin' must be numeric, got {type(isotherm_params['c_non_lin']).__name__}")
                return False
            if isotherm_params['c_non_lin'] <= 0:
                print(f"'c_non_lin' must be positive, got {isotherm_params['c_non_lin']}")
                return False
    
    elif isotherm == 'freundlich':
        required = ['K_freund', 'n_freund']
        for param in required:
            if param not in isotherm_params:
                print(f"Missing required parameter: {param}")
                return False
            if not isinstance(isotherm_params[param], (int, float)):
                print(f"'{param}' must be numeric, got {type(isotherm_params[param]).__name__}")
                return False
            if isotherm_params[param] <= 0:
                print(f"'{param}' must be positive, got {isotherm_params[param]}")
                return False
    
    elif isotherm == 'langmuir':
        required = ['Q_max', 'K_langmuir']
        for param in required:
            if param not in isotherm_params:
                print(f"Missing required parameter: {param}")
                return False
            if not isinstance(isotherm_params[param], (int, float)):
                print(f"'{param}' must be numeric, got {type(isotherm_params[param]).__name__}")
                return False
            if isotherm_params[param] <= 0:
                print(f"'{param}' must be positive, got {isotherm_params[param]}")
                return False
    
    return True

def check_toml_AWI(awi_dict: dict) -> bool:
    """ Validate air-water interface parameters from TOML config """
    
    # Check AWI_type
    if 'AWI_type' not in awi_dict:
        print("Missing required key: AWI_type")
        return False
    
    valid_types = ['SWC-based', 'Guo']
    if awi_dict['AWI_type'] not in valid_types:
        print(f"'AWI_type' must be one of {valid_types}, got '{awi_dict['AWI_type']}'")
        return False
    
    awi_type = awi_dict['AWI_type']
    
    # Check if the corresponding section exists
    if awi_type not in awi_dict:
        print(f"Missing section for AWI_type: [AWI.{awi_type}]")
        return False
    
    awi_params = awi_dict[awi_type]
    
    # Validate based on AWI type
    if awi_type == 'SWC-based':
        if 'scaling_factor_AWI' not in awi_params:
            print("Missing required parameter: scaling_factor_AWI")
            return False
        if not isinstance(awi_params['scaling_factor_AWI'], (int, float)):
            print(f"'scaling_factor_AWI' must be numeric, got {type(awi_params['scaling_factor_AWI']).__name__}")
            return False
        if awi_params['scaling_factor_AWI'] <= 0:
            print(f"'scaling_factor_AWI' must be positive, got {awi_params['scaling_factor_AWI']}")
            return False
    
    elif awi_type == 'Guo':
        required = ['guo_x0', 'guo_x1', 'guo_x2']
        for param in required:
            if param not in awi_params:
                print(f"Missing required parameter: {param}")
                return False
            if not isinstance(awi_params[param], (int, float)):
                print(f"'{param}' must be numeric, got {type(awi_params[param]).__name__}")
                return False
            if awi_params[param] <= 0:
                print(f"'{param}' must be positive, got {awi_params[param]}")
                return False
    
    return True
    
def check_toml_sorption_awi(sorption_awi_dict: dict) -> bool:
    """ Validate air-water interface sorption parameters from TOML config """
    
    # Check Kawi_method
    if 'Kawi_method' not in sorption_awi_dict:
        print("Missing required key: Kawi_method")
        return False
    
    valid_methods = ['direct_input', 'szyskowski-langmuir']
    if sorption_awi_dict['Kawi_method'] not in valid_methods:
        print(f"'Kawi_method' must be one of {valid_methods}, got '{sorption_awi_dict['Kawi_method']}'")
        return False
    
    kawi_method = sorption_awi_dict['Kawi_method']
    
    # Validate based on method
    if kawi_method == 'direct_input':
        if 'Kaw' not in sorption_awi_dict:
            print("Missing required parameter: Kaw")
            return False
        if not isinstance(sorption_awi_dict['Kaw'], (int, float)):
            print(f"'Kaw' must be numeric, got {type(sorption_awi_dict['Kaw']).__name__}")
            return False
        if sorption_awi_dict['Kaw'] <= 0:
            print(f"'Kaw' must be positive, got {sorption_awi_dict['Kaw']}")
            return False
    
    elif kawi_method == 'szyszkowski-langmuir':
        required = ['szyszkowski_a', 'szyszkowski_b']
        for param in required:
            if param not in sorption_awi_dict:
                print(f"Missing required parameter: {param}")
                return False
            if not isinstance(sorption_awi_dict[param], (int, float)):
                print(f"'{param}' must be numeric, got {type(sorption_awi_dict[param]).__name__}")
                return False
            if sorption_awi_dict[param] <= 0:
                print(f"'{param}' must be positive, got {sorption_awi_dict[param]}")
                return False
    
    return True

def check_toml_pfas(pfas_dict: dict) -> bool:
    """ Validate PFAS parameters from TOML config """
    
    # Check name
    if 'name' not in pfas_dict:
        print("Missing required key: name")
        return False
    if not isinstance(pfas_dict['name'], str):
        print(f"'name' must be a string, got {type(pfas_dict['name']).__name__}")
        return False
    
    # Check molecular_weight
    if 'molecular_weight' not in pfas_dict:
        print("Missing required parameter: molecular_weight")
        return False
    if not isinstance(pfas_dict['molecular_weight'], (int, float)):
        print(f"'molecular_weight' must be numeric, got {type(pfas_dict['molecular_weight']).__name__}")
        return False
    if pfas_dict['molecular_weight'] <= 0:
        print(f"'molecular_weight' must be positive, got {pfas_dict['molecular_weight']}")
        return False
    
    # Check surface_tension
    if 'surface_tension' not in pfas_dict:
        print("Missing required parameter: surface_tension")
        return False
    if not isinstance(pfas_dict['surface_tension'], (int, float)):
        print(f"'surface_tension' must be numeric, got {type(pfas_dict['surface_tension']).__name__}")
        return False
    if pfas_dict['surface_tension'] <= 0:
        print(f"'surface_tension' must be positive, got {pfas_dict['surface_tension']}")
        return False
    
    # Check cas_number if present =
    if 'cas_number' in pfas_dict:
        if not isinstance(pfas_dict['cas_number'], str):
            print(f"'cas_number' must be a string, got {type(pfas_dict['cas_number']).__name__}")
            return False
    
    return True
    
    