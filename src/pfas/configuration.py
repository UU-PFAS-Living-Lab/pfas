"""
TOML configuration file validator for PFAS transport modeling.

This module provides functionality to read and validate TOML configuration files
for soil transport simulations, including validation of experimental conditions,
soil properties, sorption parameters, air-water interface settings, and PFAS properties.
"""

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa


class Configuration():
    """A read-only view over a parsed TOML configuration.

    This class wraps a nested dictionary and provides convenient attribute-style access to configuration
    values. If an attribute is not present on the instance, attribute lookup
    falls back to a recursive search of the underlying dictionary using
    :meth:`find`.

    Attributes
    ----------
    config_dict : Dict[str, Any]
        The underlying nested dictionary containing the parsed configuration.

    Methods
    -------
    find(key, sub_dict=None)
        Recursively search for ``key`` inside the configuration dictionary and
        return the associated value if found, otherwise ``None``.

    Notes
    -----
    This class intentionally implements ``__getattribute__`` to allow
    attribute-style access for keys present in the configuration. Use
    :func:`read_toml` to construct a ``Configuration`` from a TOML file.

    Example
    -------
    >>> config = read_toml(Path("config.toml"))
    >>> config.domain_length
    """

    def __init__(self, config_dict):
        self.config_dict = config_dict

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError:
            val = self.find(key)
            if val is None:
                raise
            return val


    def find(self, key, sub_dict=None):
        # print(key, sub_dict)
        if sub_dict is None:
            sub_dict = self.config_dict
        if key in sub_dict:
            return sub_dict[key]
        if isinstance(sub_dict, dict):
            for new_sub_dict in sub_dict.values():
                if not isinstance(new_sub_dict, dict):
                    continue
                found_value = self.find(key, new_sub_dict)
                if found_value is not None:
                    return found_value
        return None

def read_toml(path: Path) -> Dict[str, Any]:
    """
    Read a TOML configuration file.

    Args:
        path: Path to the TOML file

    Returns
    -------
        Dictionary containing the parsed TOML configuration
    """
    with open(path, "rb") as handle:
        config_dict = tomllib.load(handle)
    return Configuration(config_dict)


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """
    Validate all sections of the configuration dictionary.

    Args:
        config_dict: Configuration dictionary parsed from TOML

    Returns
    -------
        True if all validations pass, False otherwise
    """
    validators = [
        ("experimental_conditions", check_toml_experimental_conditions),
        ("soil", check_toml_soil),
        ("sorption_solid", check_toml_sorption),
        ("AWI", check_toml_awi),
        ("sorption_AWI", check_toml_sorption_awi),
        ("pfas", check_toml_pfas),
    ]

    results = []
    for section_name, validator_func in validators:
        if section_name not in config_dict:
            print(f"Missing required section: {section_name}")
            results.append(False)
        else:
            results.append(validator_func(config_dict[section_name]))

    return all(results)


def check_toml_experimental_conditions(experimental_conditions_dict: Dict[str, Any]) -> bool:
    """
    Validate experimental conditions parameters from TOML config.

    Args:
        experimental_conditions_dict: Dictionary containing experimental conditions

    Returns
    -------
        True if validation passes, False otherwise
    """
    required_keys = ['domain_length', 'spatial_resolution', 'time_total', 'soil_temp']

    # Check if all required keys exist and are valid
    for key in required_keys:
        if key not in experimental_conditions_dict:
            print(f"Missing required key: {key}")
            return False

        value = experimental_conditions_dict[key]
        if not isinstance(value, (int, float)):
            print(f"Value for '{key}' must be numeric, got {type(value).__name__}")
            return False

        if value <= 0:
            print(f"Value for '{key}' must be positive, got {value}")
            return False

    # Validate initial conditions
    if 'initial' not in experimental_conditions_dict:
        print("Missing required section: initial")
        return False

    initial_params = experimental_conditions_dict['initial']
    required_initial_keys = ['init_sat', 'initial_solute_concentration']

    for key in required_initial_keys:
        if key not in initial_params:
            print(f"Missing required key in initial conditions: {key}")
            return False

    # Validate boundary conditions
    if 'boundary' not in experimental_conditions_dict:
        print("Missing required section: boundary")
        return False

    boundary_params = experimental_conditions_dict['boundary']
    required_boundary_keys = [
        "average_infiltration_rate",
        "pulse_duration",
        "solute_concentration_influx"
    ]

    for key in required_boundary_keys:
        if key not in boundary_params:
            print(f"Missing required key in boundary conditions: {key}")
            return False

        value = boundary_params[key]
        if not isinstance(value, (int, float)):
            print(f"Value for '{key}' must be numeric, got {type(value).__name__}")
            return False

        if value <= 0:
            print(f"Value for '{key}' must be positive, got {value}")
            return False

    return True


def check_toml_soil(soil_dict: Dict[str, Any]) -> bool:
    """
    Validate soil parameters from TOML config.

    Args:
        soil_dict: Dictionary containing soil parameters

    Returns
    -------
        True if validation passes, False otherwise
    """
    expected_keys = [
        "soil_name", "soil_type", "bulk_density", "porosity",
        "van_genuchten_alpha", "van_genuchten_n", "saturated_water_content",
        "hydraulic_conductivity", "dispersivity", "residual_water_content",
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
        "hydraulic_conductivity", "dispersivity", "residual_water_content"
    ]

    for field in numeric_fields:
        if not isinstance(soil_dict[field], (int, float)):
            print(f"{field} must be numeric, got {type(soil_dict[field]).__name__}")
            return False

    # Range checks
    if not 0 < soil_dict["porosity"] <= 1:
        print(f"porosity must be between 0 and 1, got {soil_dict['porosity']}")
        return False
    if not 0 < soil_dict["saturated_water_content"] <= 1:
        print(
            f"saturated_water_content must be between 0 and 1, "
            f"got {soil_dict['saturated_water_content']}"
        )
        return False

    return True


def _validate_numeric_param(
    params: Dict[str, Any],
    param_name: str,
    allow_positive_only: bool = True,
    min_value: float = None,
    max_value: float = None
) -> bool:
    """
    Validate a numeric parameter.

    Args:
        params: Dictionary containing parameters
        param_name: Name of parameter to validate
        allow_positive_only: If True, value must be positive
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns
    -------
        True if validation passes, False otherwise
    """
    if param_name not in params:
        print(f"Missing required parameter: {param_name}")
        return False

    value = params[param_name]
    if not isinstance(value, (int, float)):
        print(f"'{param_name}' must be numeric, got {type(value).__name__}")
        return False

    if allow_positive_only and value <= 0:
        print(f"'{param_name}' must be positive, got {value}")
        return False

    if min_value is not None and value < min_value:
        print(f"'{param_name}' must be >= {min_value}, got {value}")
        return False

    if max_value is not None and value > max_value:
        print(f"'{param_name}' must be <= {max_value}, got {value}")
        return False

    return True


def check_toml_sorption(sorption_dict: Dict[str, Any]) -> bool:
    """
    Validate sorption parameters from TOML config.

    Args:
        sorption_dict: Dictionary containing sorption parameters

    Returns
    -------
        True if validation passes, False otherwise
    """
    # Check kinetic_sorption flag
    if 'kinetic_sorption' not in sorption_dict:
        print("Missing required key: kinetic_sorption")
        return False

    if not isinstance(sorption_dict['kinetic_sorption'], bool):
        print(
            f"'kinetic_sorption' must be a boolean, "
            f"got {type(sorption_dict['kinetic_sorption']).__name__}"
        )
        return False

    # Check sorption isotherm type
    if 'sorption_isotherm' not in sorption_dict:
        print("Missing required key: sorption_isotherm")
        return False

    valid_isotherms = ['linear', 'freundlich', 'langmuir']
    if sorption_dict['sorption_isotherm'] not in valid_isotherms:
        print(
            f"'sorption_isotherm' must be one of {valid_isotherms}, "
            f"got '{sorption_dict['sorption_isotherm']}'"
        )
        return False

    # Check kinetic parameters if kinetic=true
    if sorption_dict['kinetic_sorption']:
        if not _validate_kinetic_params(sorption_dict):
            return False

    # Check the specific isotherm section
    isotherm = sorption_dict['sorption_isotherm']
    if isotherm not in sorption_dict:
        print(f"Missing section for isotherm: [sorption_solid.{isotherm}]")
        return False

    # Validate isotherm-specific parameters
    if isotherm == 'linear':
        return _validate_linear_isotherm(sorption_dict[isotherm])
    if isotherm == 'freundlich':
        return _validate_freundlich_isotherm(sorption_dict[isotherm])
    if isotherm == 'langmuir':
        return _validate_langmuir_isotherm(sorption_dict[isotherm])

    return True


def _validate_kinetic_params(sorption_dict: Dict[str, Any]) -> bool:
    """Validate kinetic sorption parameters."""
    if 'kinetic' not in sorption_dict:
        print("Missing [sorption_solid.kinetic] section when kinetic_sorption=true")
        return False

    kinetic_params = sorption_dict['kinetic']

    if not _validate_numeric_param(
        kinetic_params, 'frac_int', allow_positive_only=False, min_value=0, max_value=1
    ):
        return False

    if not _validate_numeric_param(kinetic_params, 'rate_const', allow_positive_only=True):
        return False

    return True


def _validate_linear_isotherm(isotherm_params: Dict[str, Any]) -> bool:
    """Validate linear isotherm parameters."""
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
        return _validate_numeric_param(isotherm_params, 'Kd', allow_positive_only=True)

    if kd_method == 'organic_mineral':
        required = ['OC_perc', 'Koc', 'Min_perc', 'Kmin']
        for param in required:
            if not _validate_numeric_param(isotherm_params, param, allow_positive_only=True):
                return False

    elif kd_method == 'Fabregat_Palau2021':
        required = ['OC_perc', 'Min_perc', 'chain_length']
        for param in required:
            if not _validate_numeric_param(isotherm_params, param, allow_positive_only=True):
                return False

        if not isinstance(isotherm_params['chain_length'], int):
            print(
                f"'chain_length' must be an integer, "
                f"got {type(isotherm_params['chain_length']).__name__}"
            )
            return False

    # Check optional c_non_lin if present
    if 'c_non_lin' in isotherm_params:
        if not _validate_numeric_param(isotherm_params, 'c_non_lin', allow_positive_only=True):
            return False

    return True


def _validate_freundlich_isotherm(isotherm_params: Dict[str, Any]) -> bool:
    """Validate Freundlich isotherm parameters."""
    required = ['K_freund', 'n_freund']
    for param in required:
        if not _validate_numeric_param(isotherm_params, param, allow_positive_only=True):
            return False
    return True


def _validate_langmuir_isotherm(isotherm_params: Dict[str, Any]) -> bool:
    """Validate Langmuir isotherm parameters."""
    required = ['Q_max', 'K_langmuir']
    for param in required:
        if not _validate_numeric_param(isotherm_params, param, allow_positive_only=True):
            return False
    return True


def check_toml_awi(awi_dict: Dict[str, Any]) -> bool:
    """
    Validate air-water interface parameters from TOML config.

    Args:
        awi_dict: Dictionary containing AWI parameters

    Returns
    -------
        True if validation passes, False otherwise
    """
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
        return _validate_numeric_param(
            awi_params, 'scaling_factor_AWI', allow_positive_only=True
        )

    if awi_type == 'Guo':
        required = ['guo_x0', 'guo_x1', 'guo_x2']
        for param in required:
            if not _validate_numeric_param(awi_params, param, allow_positive_only=True):
                return False

    return True


def check_toml_sorption_awi(sorption_awi_dict: Dict[str, Any]) -> bool:
    """
    Validate air-water interface sorption parameters from TOML config.

    Args:
        sorption_awi_dict: Dictionary containing AWI sorption parameters

    Returns
    -------
        True if validation passes, False otherwise
    """
    if 'Kawi_method' not in sorption_awi_dict:
        print("Missing required key: Kawi_method")
        return False

    valid_methods = ['direct_input', 'szyszkowski-langmuir']
    if sorption_awi_dict['Kawi_method'] not in valid_methods:
        print(
            f"'Kawi_method' must be one of {valid_methods}, "
            f"got '{sorption_awi_dict['Kawi_method']}'"
        )
        return False

    kawi_method = sorption_awi_dict['Kawi_method']

    # Validate based on method
    if kawi_method == 'direct_input':
        return _validate_numeric_param(sorption_awi_dict, 'Kaw', allow_positive_only=True)

    if kawi_method == 'szyszkowski-langmuir':
        required = ['szyszkowski_a', 'szyszkowski_b']
        for param in required:
            if not _validate_numeric_param(
                sorption_awi_dict, param, allow_positive_only=True
            ):
                return False

    return True


def check_toml_pfas(pfas_dict: Dict[str, Any]) -> bool:
    """
    Validate PFAS parameters from TOML config.

    Args:
        pfas_dict: Dictionary containing PFAS parameters

    Returns
    -------
        True if validation passes, False otherwise
    """
    # Check name
    if 'name' not in pfas_dict:
        print("Missing required key: name")
        return False
    if not isinstance(pfas_dict['name'], str):
        print(f"'name' must be a string, got {type(pfas_dict['name']).__name__}")
        return False

    # Check molecular_weight
    if not _validate_numeric_param(pfas_dict, 'molecular_weight', allow_positive_only=True):
        return False

    # Check surface_tension
    if not _validate_numeric_param(pfas_dict, 'surface_tension', allow_positive_only=True):
        return False

    # Check cas_number if present
    if 'cas_number' in pfas_dict:
        if not isinstance(pfas_dict['cas_number'], str):
            print(f"'cas_number' must be a string, got {type(pfas_dict['cas_number']).__name__}")
            return False

    return True
