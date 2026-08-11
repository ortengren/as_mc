"""Repo-relative data paths, anchored on the package, not on this file's depth.

``potentials.py``/``initialize.py``/``validation.py`` all need to reach the
repo-root ``data/`` tree, which sits *outside* the package. Anchoring on
``asmcmc.__file__`` (rather than a hardcoded ``Path(__file__).parents[N]`` in
each of those modules) means the resolution stays correct regardless of how
deep any particular module sits inside the package.
"""

import os
from pathlib import Path

import asmcmc

PACKAGE_ROOT = Path(asmcmc.__file__).resolve().parent  # .../repo/asmcmc
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = Path(os.environ.get("ASMCMC_DATA_DIR", REPO_ROOT / "data"))


def data_path(*parts) -> Path:
    """Resolve a path under the repo's ``data/`` tree (overridable via
    ``ASMCMC_DATA_DIR``, e.g. for a non-editable install where ``data/`` isn't
    shipped alongside the package)."""
    return DATA_DIR.joinpath(*parts)
