"""
pfas.data_loader.

Utilities for loading and listing the packaged JSON datasets bundled with the
``pfas`` library, plus support for loading user‑supplied JSON files.

Functions
---------
load_dataset(name_or_path)
    Load a packaged dataset by name, or load a JSON file from a filesystem path.
available_datasets()
    Return a list of packaged dataset names.
load_json_file(path)
    Load a JSON file from an arbitrary filesystem path.

Example
-------
>>> from pfas.data_loader import available_datasets, load_dataset
>>> print(available_datasets())
['PFASs', 'soils', 'sp_matrix']

# Load packaged dataset
>>> data = load_dataset('PFASs')

# Load external JSON file
>>> data = load_dataset('/path/to/custom.json')
"""
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

from pfas import ureg


def load_json_file(path):
    """Load a JSON file from an arbitrary filesystem path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add_dict_with_units(cur_dict: Any, ureg):
    if not isinstance(cur_dict, dict):
        return cur_dict
    if "value" in cur_dict and "unit" in cur_dict:
        if cur_dict["value"] is None:
            return None
        elif cur_dict["unit"] == "-":
            return cur_dict["value"]
        try:
            return ureg.Quantity(cur_dict["value"], ureg(cur_dict["unit"]))
        except AssertionError:
            raise ValueError(f"Failed parsing unit: {cur_dict['unit']}.")
    return {k: _add_dict_with_units(v, ureg) for k, v in cur_dict.items()}

def load_dataset(name_or_path, load_units: bool = True):
    """
    Load a packaged dataset by name, or load a JSON file from a filesystem path.

    If `name_or_path` is a valid file path, the JSON file is loaded directly.
    Otherwise, it must match one of the packaged dataset names.
    """
    # Case 1: user provided a filesystem path
    if os.path.isfile(name_or_path):
        path_fp = Path(name_or_path)
    # Case 2: user provided a dataset name
    elif name_or_path not in available_datasets():
        raise ValueError(
            f"Invalid dataset '{name_or_path}'. "
            f"Choose from: {sorted(available_datasets())} "
            f"or provide a valid JSON file path."
        )
    else:
        package = "pfas.data"
        filename = f"{name_or_path}.json"
        path_fp = resources.files(package).joinpath(filename)
    data_dict = load_json_file(path_fp)
    if load_units:

        return _add_dict_with_units(data_dict, ureg)
    return data_dict


def available_datasets():
    """Return a list of all packaged dataset names."""
    package = "pfas.data"
    return [
        p.stem
        for p in resources.files(package).iterdir()
        if p.suffix == ".json"
    ]
