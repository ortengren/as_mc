import ase
import json
import numpy as np
from abc import ABC, abstractmethod
from ase.neighborlist import neighbor_list
from dataclasses import asdict, dataclass
from numpy import linalg as la
from pathlib import Path
import pandas as pd
import random
from scipy.spatial.transform import Rotation

EPS_0 = 8.8541878188e-22  # F / Å


def gb_shape_function(uhat1, uhat2, rhat, kappa):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    term1 = (np.vecdot(uhat1, rhat) + np.vecdot(uhat2, rhat)) ** 2 / (
        1 + chi * np.vecdot(uhat1, uhat2)
    )
    term2 = (np.vecdot(uhat1, rhat) - np.vecdot(uhat2, rhat)) ** 2 / (
        1 - chi * np.vecdot(uhat1, uhat2)
    )
    sigma = 1 / np.sqrt(1 - (chi / 2) * (term1 + term2))
    return sigma


def gb_axial_energy(uhat1, uhat2, kappa):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    return 1 / np.sqrt(1 - (chi * np.vecdot(uhat1, uhat2)) ** 2)


def gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu):
    chi_prime = (kappa_prime ** (1 / mu) - 1) / (kappa_prime ** (1 / mu) + 1)
    term1 = (np.vecdot(uhat1, rhat) + np.vecdot(uhat2, rhat)) ** 2 / (
        1 + chi_prime * np.vecdot(uhat1, uhat2)
    )
    term2 = (np.vecdot(uhat1, rhat) - np.vecdot(uhat2, rhat)) ** 2 / (
        1 - chi_prime * np.vecdot(uhat1, uhat2)
    )
    return 1 - chi_prime * (term1 + term2) / 2


def gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu):
    eps1 = gb_axial_energy(uhat1, uhat2, kappa)
    eps2 = gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu)
    return eps0 * eps1**nu * eps2**mu


def gb(uhat1, uhat2, r, sigma0, eps0, kappa, kappa_prime, mu, nu, xi):
    rmag = np.expand_dims(la.norm(r, axis=-1), axis=-1)
    rhat = r / rmag
    eps = gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu)
    sigma = gb_shape_function(uhat1, uhat2, rhat, kappa)
    term = xi * sigma0 / (la.norm(r, axis=-1) - (sigma0 * sigma) + (xi * sigma0))
    return 4 * eps * (term**12 - term**6)


def quadrupole(uhat1, uhat2, r, Q):
    rmag = np.expand_dims(la.norm(r, axis=-1), axis=-1)
    rhat = r / rmag
    a1 = np.vecdot(uhat1, rhat)
    a2 = np.vecdot(uhat2, rhat)
    b12 = np.vecdot(uhat1, uhat2)
    prefactor = 0.75 * Q**2 / rmag**5
    prefactor = np.squeeze(prefactor)
    s = (
        1
        + 2 * b12**2
        - 5 * (a1**2 + a2**2)
        - 20 * a1 * a2 * b12
        + 35 * (a1**2) * (a2**2)
    )
    return prefactor * s


def get_total_energy(M, sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q):
    # M should have shape (N, 1431, 3, 3) where N is the number of frames
    E_GB = gb(
        M[:, :, 0, :],
        M[:, :, 1, :],
        M[:, :, 2, :],
        sigma0,
        eps0,
        kappa,
        kappa_prime,
        mu,
        nu,
        xi,
    )
    E_QQ = quadrupole(M[:, :, 0, :], M[:, :, 1, :], M[:, :, 2, :], Q)
    E_QQ = np.squeeze(E_QQ)
    pw_energies = E_GB + E_QQ
    # pw_energies should have shape (N, 1431)
    energies = np.sum(pw_energies, axis=-1)
    return energies


def calc_total_energy(frame, nl_cutoff, potential=None):
    """Total pair energy of ``frame`` under ``potential``.

    ``potential`` is a :class:`Potential`; if ``None`` the package default
    (:data:`DEFAULT_POTENTIAL`) is used.
    """
    if potential is None:
        potential = DEFAULT_POTENTIAL

    # Every interacting pair (i, j) and its shift vector. neighbor_list emits
    # each pair in BOTH directions, so the sum below is halved rather than
    # filtered with i < j: that filter also drops the i == j self-image pairs
    # a molecule has with its own periodic copies, which are real interactions
    # whenever a lattice vector is shorter than the cutoff. (It made the energy
    # of a one-molecule cell exactly zero.) Halving is the same convention as
    # fitting_gbq.data.extract_periodic_pairs / fit.predict_per_mol, and is
    # identical to the old result for boxes larger than the cutoff.
    i, j, s = neighbor_list("ijS", frame, nl_cutoff)

    # calculate displacements
    cell = frame.get_cell()
    shift_vecs = np.dot(s, cell)
    displacements = frame.positions[j] + shift_vecs - frame.positions[i]

    # calculate orientations
    uhat1 = frame.arrays["or_vec"][i]
    uhat2 = frame.arrays["or_vec"][j]

    # calculate pairwise energies
    return 0.5 * np.sum(potential.pair_energy(uhat1, uhat2, displacements))


# TODO: Class structure may need to be updated for AniSOAP implementation.  Currently
# handles only pairwise potentials.  This change would also likely require changes to
# MetropolisCalculator.
class Potential(ABC):
    """Interface the Metropolis sampler depends on: a named, pairwise energy.

    Concrete potentials carry their own parameters and implement
    :meth:`pair_energy`. The ``name`` (provenance, e.g. which fit) is stamped
    into simulation outputs so every run records which potential it used.
    """

    name: str

    @abstractmethod
    def pair_energy(self, uhat1, uhat2, r) -> np.ndarray:
        """Per-pair energies for orientations ``uhat1``/``uhat2`` and
        displacement vectors ``r`` (callers sum over the returned array)."""


# GB parameters in the order ``gb`` expects them (followed by the quadrupole Q).
_GB_PARAM_KEYS = ("sigma0", "eps0", "kappa", "kappa_prime", "mu", "nu", "xi")


@dataclass(frozen=True)
class GBQPotential(Potential):
    """Gay-Berne + quadrupole pair potential with a recorded provenance name."""

    name: str
    sigma0: float
    eps0: float
    kappa: float
    kappa_prime: float
    mu: float
    nu: float
    xi: float
    Q: float

    @classmethod
    def from_json(cls, path, name=None):
        """Build from a fit ``params.json`` (the ``{value, unit}`` schema written
        by ``asmcmc.fitting``). ``name`` defaults to the path tail below the
        ``fitting/`` directory, e.g. ``multiseed/uniform/seed_0/uniform``.

        Outside a ``fitting/`` tree the name comes from the **file stem**, not
        the parent directory: the tracked params files all sit in ``data/``, so
        a directory-based fallback named every one of them ``"data"`` -- making
        the name useless exactly where it matters most, as provenance stamped
        into run configs, dataset frames and benchmark output."""
        path = Path(path)
        data = json.loads(path.read_text())
        if name is None:
            parts = path.parent.parts
            if "fitting" in parts:
                name = "/".join(parts[parts.index("fitting") + 1 :])
            else:
                name = path.stem
        values = {k: data[k]["value"] for k in (*_GB_PARAM_KEYS, "Q")}
        return cls(name=name, **values)

    @property
    def gb_args(self):
        """GB parameters as a tuple in the order ``gb`` accepts them."""
        return tuple(getattr(self, k) for k in _GB_PARAM_KEYS)

    def gb_params_dict(self):
        """GB parameters as a dict (matches the legacy ``GB_PARAMS`` mapping)."""
        return {k: getattr(self, k) for k in _GB_PARAM_KEYS}

    def pair_energy(self, uhat1, uhat2, r):
        gb_e = gb(uhat1, uhat2, r, *self.gb_args)
        qq_e = np.squeeze(quadrupole(uhat1, uhat2, r, self.Q))
        return gb_e + qq_e

    def to_dict(self):
        return {"type": "GBQPotential", **asdict(self)}


_POTENTIALS = {"GBQPotential": GBQPotential}


def potential_from_dict(d):
    d = dict(d)
    cls = _POTENTIALS[d.pop("type")]
    return cls(**d)


# Resolve the default potential relative to this file (not the cwd) so imports
# work regardless of where the interpreter is launched. Points at the tracked
# uniform/seed_0 fit; switch via GBQPotential.from_json(<other params.json>).
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = _REPO_ROOT / "data/my_fitted_gbq_params.json"
CACELLI_PARAMS_PATH = _REPO_ROOT / "data/lit_gbq_params.json"
DEFAULT_POTENTIAL = GBQPotential.from_json(DEFAULT_PARAMS_PATH)
CACELLI_POTENTIAL = GBQPotential.from_json(CACELLI_PARAMS_PATH)

# Backward-compatible aliases, derived from the active default so dependents
# (initialize.py lattice spacing, nvt_scan.py reduced-unit scales) stay
# consistent with whatever potential the sampler uses.
GB_PARAMS = DEFAULT_POTENTIAL.gb_params_dict()
QQ = DEFAULT_POTENTIAL.Q
