import json
import os


VALID_DATASETS = {"PFASs", "soils", "spa_matrix"}


def load_dataset(name, data_dir="data"):
    """
    Load one of the JSON datasets.

    Parameters
    ----------
    name : str
        One of: "PFASs", "soils", "spa_matrix"
    data_dir : str
        Folder containing JSON files (default="data")

    Returns
    -------
    dict
    """

    if name not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset name '{name}'. "
            f"Choose from: {sorted(VALID_DATASETS)}"
        )

    path = os.path.join(data_dir, f"{name}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Did you run export_to_json.py?"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
