import json
from importlib import resources


VALID_DATASETS = {"PFASs", "soils", "spa_matrix"}


def load_dataset(name):
    """
    Load one of the packaged JSON datasets.
    """

    if name not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset '{name}'. "
            f"Choose from: {sorted(VALID_DATASETS)}"
        )

    package = "pfas.data"

    with resources.files(package).joinpath(f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)
