from pathlib import Path

import pytest

from pfas.configuration import read_toml


@pytest.fixture(scope="session")
def configuration():
    config_path = Path("examples", "data", "config.toml")
    return read_toml(config_path)
