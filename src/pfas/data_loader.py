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


def load_json_file(path):
    """Load a JSON file from an arbitrary filesystem path."""
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(name_or_path):
    """
    Load a packaged dataset by name, or load a JSON file from a filesystem path.

    If `name_or_path` is a valid file path, the JSON file is loaded directly.
    Otherwise, it must match one of the packaged dataset names.
    """
    # Case 1: user provided a filesystem path
    if os.path.isfile(name_or_path):
        return load_json_file(name_or_path)

    # Case 2: user provided a dataset name
    if name_or_path not in available_datasets():
        raise ValueError(
            f"Invalid dataset '{name_or_path}'. "
            f"Choose from: {sorted(available_datasets())} "
            f"or provide a valid JSON file path."
        )

    package = "pfas.data"
    filename = f"{name_or_path}.json"

    with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def available_datasets():
    """Return a list of all packaged dataset names."""
    package = "pfas.data"
    return [
        p.stem
        for p in resources.files(package).iterdir()
        if p.suffix == ".json"
    ]
