"""Locations of the repository and its ``data/`` directory."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent  # .../repo/asmcmc
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = Path(os.environ.get("ASMCMC_DATA_DIR", REPO_ROOT / "data"))


def data_path(*parts) -> Path:
    """Resolve a path under the repo's ``data/`` tree.

    Set ``ASMCMC_DATA_DIR`` to override, e.g. for a non-editable install that
    does not ship ``data/`` alongside the package.
    """
    return DATA_DIR.joinpath(*parts)
