"""Gay-Berne + quadrupole pair potentials for uniaxial particles, and a frame's total energy.

Functions take unit symmetry axes ``uhat1``/``uhat2`` and separation vectors ``r``
(A), shaped ``(..., 3)`` and vectorised over pairs, and return energies in eV.
"""

import json
import numpy as np
from abc import ABC, abstractmethod
from ase.neighborlist import neighbor_list
from dataclasses import asdict, dataclass
from numpy import linalg as la
from pathlib import Path

from asmcmc.paths import data_path


def gb_shape_function(uhat1, uhat2, rhat, kappa):
    """sigma / sigma0: the orientation-dependent contact distance, for aspect ratio ``kappa``."""
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
    """Well-depth factor set by the relative orientation of the two axes alone."""
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    return 1 / np.sqrt(1 - (chi * np.vecdot(uhat1, uhat2)) ** 2)


def gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu):
    """Well-depth factor that depends on the separation direction ``rhat``."""
    chi_prime = (kappa_prime ** (1 / mu) - 1) / (kappa_prime ** (1 / mu) + 1)
    term1 = (np.vecdot(uhat1, rhat) + np.vecdot(uhat2, rhat)) ** 2 / (
        1 + chi_prime * np.vecdot(uhat1, uhat2)
    )
    term2 = (np.vecdot(uhat1, rhat) - np.vecdot(uhat2, rhat)) ** 2 / (
        1 - chi_prime * np.vecdot(uhat1, uhat2)
    )
    return 1 - chi_prime * (term1 + term2) / 2


def gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu):
    """The well depth: ``eps0 * gb_axial_energy**nu * gb_directional_energy**mu``."""
    eps1 = gb_axial_energy(uhat1, uhat2, kappa)
    eps2 = gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu)
    return eps0 * eps1**nu * eps2**mu


def gb(uhat1, uhat2, r, sigma0, eps0, kappa, kappa_prime, mu, nu, xi):
    """Gay-Berne pair energy; ``xi`` rescales the range of the shifted 12-6 form."""
    rmag = np.expand_dims(la.norm(r, axis=-1), axis=-1)
    rhat = r / rmag
    eps = gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu)
    sigma = gb_shape_function(uhat1, uhat2, rhat, kappa)
    term = xi * sigma0 / (la.norm(r, axis=-1) - (sigma0 * sigma) + (xi * sigma0))
    return 4 * eps * (term**12 - term**6)


def quadrupole(uhat1, uhat2, r, Q):
    """Energy of two point quadrupoles along the particle axes; ``Q`` enters only as Q**2."""
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


def calc_total_energy(frame, nl_cutoff, potential=None):
    """Total pair energy of ``frame`` under ``potential``.

    ``potential`` is a :class:`Potential`; if ``None`` the package default
    (:data:`DEFAULT_POTENTIAL`) is used.
    """
    if potential is None:
        potential = DEFAULT_POTENTIAL

    # Every interacting pair (i, j) with its periodic shift. neighbor_list lists
    # each pair in both directions, so the sum is halved. Filtering on i < j
    # instead would also drop the pairs a molecule forms with its own periodic
    # images, which are real interactions whenever a lattice vector is shorter
    # than the cutoff. fitting_gbq uses the same convention.
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


class Potential(ABC):
    """A named pair potential: the interface the sampler depends on.

    Concrete potentials carry their own parameters and implement
    :meth:`pair_energy`; ``name`` is stamped into every run's outputs as
    provenance. Only pair potentials are supported, because
    ``MetropolisSampler.particle_energy`` sums ``pair_energy`` over neighbours. A
    many-body model (e.g. a full AniSOAP potential) would need a per-particle
    energy hook here and in the sampler.
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
        """Build from a ``params.json`` in the ``{name: {value, unit}}`` schema that
        ``asmcmc.fitting_gbq`` writes.

        ``name`` defaults to the path below a ``fitting/`` directory (e.g.
        ``multiseed/uniform/seed_0/uniform``), or else to the file stem
        (``lit_gbq_params``).
        """
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
        """GB parameters as keyword arguments for ``gb``."""
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


# The sampler's default: Cacelli et al.'s GBQIII parameterisation. A candidate
# must pass delta_learning.dimer_benchmark before it replaces this; a good fit to
# condensed-phase energies is not enough (see docs/findings.md).
CACELLI_PARAMS_PATH = data_path("lit_gbq_params.json")
CACELLI_POTENTIAL = GBQPotential.from_json(CACELLI_PARAMS_PATH)
DEFAULT_POTENTIAL = CACELLI_POTENTIAL

