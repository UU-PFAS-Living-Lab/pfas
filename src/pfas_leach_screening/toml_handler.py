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

def check_toml_grid(grid_dict: dict) -> bool: 
    """ Validate grid parameters from TOML config """
    
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


    

