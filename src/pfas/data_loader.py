import json
from importlib import resources


def load_dataset(name):
    """
    Load one of the packaged JSON datasets.
    """

    if name not in available_datasets():
        raise ValueError(
            f"Invalid dataset '{name}'. "
            f"Choose from: {sorted(available_datasets())}"
        )

    package = "pfas.data"

    with resources.files(package).joinpath(f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def available_datasets():
    package = "pfas.data"
    return [
        p.stem
        for p in resources.files(package).iterdir()
        if p.suffix == ".json"
    ]
