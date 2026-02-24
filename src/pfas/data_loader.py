"""
pfas.data_loader.

Utilities for loading and listing the packaged JSON datasets bundled with the
``pfas`` library.

The datasets are stored as JSON files inside the ``pfas.data`` package and are
accessed via :mod:`importlib.resources`, making them available regardless of
how the package is installed (e.g. as a wheel, egg, or editable install).

Functions
---------
load_dataset(name)
    Load a named dataset and return its contents as a Python object.
available_datasets()
    Return a list of all dataset names that can be passed to
    :func:`load_dataset'.

Example
-------
>>> from pfas.data_loader import available_datasets, load_dataset
>>> print(available_datasets())
['PFASs', 'soils', 'sp_matrix']
>>> data = load_dataset('PFASs')
"""
import json
from importlib import resources


def load_dataset(name):
    """Load one of the packaged JSON datasets."""
    if name not in available_datasets():
        raise ValueError(
            f"Invalid dataset '{name}'. "
            f"Choose from: {sorted(available_datasets())}"
        )

    package = "pfas.data"

    with resources.files(package).joinpath(f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def available_datasets():
    """Return a list of all available dataset names."""
    package = "pfas.data"
    return [
        p.stem
        for p in resources.files(package).iterdir()
        if p.suffix == ".json"
    ]
