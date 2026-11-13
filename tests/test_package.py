import tomllib
from pathlib import Path

import sasrec2023

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_matches_pyproject():
    with PYPROJECT.open("rb") as f:
        project = tomllib.load(f)["project"]
    assert sasrec2023.__version__ == project["version"]
