import os
from dataclasses import dataclass

import ase.io
from ase.neighborlist import neighbor_list
import numpy as np

# Targets are the ABSOLUTE per-molecule DFT energies (energy_pa * atoms_per_mol).
# The intramolecular energy is frame-independent (rigid benzene), so it enters
# the model as a single fitted constant E_intra, added once per molecule at the
# frame level (fit.predict_per_mol) -- NOT inside the per-pair gbq() below.
#
# extract_periodic_pairs returns every neighbour pair in BOTH directions (and
# each periodic self-image once per direction), so a molecule's lattice energy
# is HALF the summed pair energy; predict_per_mol applies that 1/2.


def extract_periodic_pairs(frame, orientation_key, cutoff):
    """All directed neighbour pairs within ``cutoff`` (PBC, incl. self-images).

    Returns an ``(P, 4)`` array of columns ``(r_mag, a_i, a_j, b_ij)`` where
    ``a_i = r_hat . u_i``, ``a_j = r_hat . u_j``, ``b_ij = u_i . u_j``. Uses
    ``ase.neighbor_list`` (same convention as ``potentials.calc_total_energy``)
    so the per-molecule lattice sum is exactly half the summed pair energy.
    """
    orientations = frame.arrays[orientation_key]
    u = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

    i, j, S = neighbor_list("ijS", frame, cutoff)
    disp = frame.positions[j] + S @ np.asarray(frame.cell) - frame.positions[i]
    r_mag = np.linalg.norm(disp, axis=1)
    r_hat = disp / r_mag[:, None]

    u_i, u_j = u[i], u[j]
    a_i = np.einsum("pk,pk->p", r_hat, u_i)
    a_j = np.einsum("pk,pk->p", r_hat, u_j)
    b_ij = np.einsum("pk,pk->p", u_i, u_j)

    return np.stack([r_mag, a_i, a_j, b_ij], axis=1)


def precompute_dots_gb_shape_func(a_i, a_j, b_ij, sigma0, kappa):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    term1 = ((a_i + a_j) ** 2) / (1 + chi * b_ij)
    term2 = ((a_i - a_j) ** 2) / (1 - chi * b_ij)
    sigma = sigma0 / np.sqrt(1 - (chi / 2) * (term1 + term2))
    return sigma


def precompute_dots_gb_axial_energy(b_ij, kappa):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    return 1 / np.sqrt(1 - (chi * b_ij) ** 2)


def precompute_dots_gb_directional_energy(a_i, a_j, b_ij, kappa_prime, mu):
    chi_prime = (kappa_prime ** (1 / mu) - 1) / (kappa_prime ** (1 / mu) + 1)
    term1 = ((a_i + a_j) ** 2) / (1 + chi_prime * b_ij)
    term2 = ((a_i - a_j) ** 2) / (1 - chi_prime * b_ij)
    return 1 - (chi_prime / 2) * (term1 + term2)


def precompute_dots_gb_en_func(a_i, a_j, b_ij, eps0, kappa, kappa_prime, mu, nu):
    eps1 = precompute_dots_gb_axial_energy(b_ij, kappa)
    eps2 = precompute_dots_gb_directional_energy(a_i, a_j, b_ij, kappa_prime, mu)
    return eps0 * (eps1**nu) * (eps2**mu)


def precompute_dots_gb(r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu):
    eps = precompute_dots_gb_en_func(a_i, a_j, b_ij, eps0, kappa, kappa_prime, mu, nu)
    sigma = precompute_dots_gb_shape_func(a_i, a_j, b_ij, sigma0, kappa)
    term = sigma0 / (r_mag - sigma + sigma0)
    return 4 * eps * (term**12 - term**6)


def precompute_dots_quadrupole(r_mag, a_i, a_j, b_ij, Q):
    prefactor = 0.75 * (Q**2) / (r_mag**5)
    s = (
        1
        + 2 * (b_ij**2)
        - 5 * (a_i**2 + a_j**2)
        - 20 * a_i * a_j * b_ij
        + 35 * (a_i**2) * (a_j**2)
    )
    return prefactor * s


def gbq(r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu, Q):
    """Pairwise GB + quadrupole energy (vectorises over pair arrays).

    E_intra is intentionally absent: it is a per-molecule constant added once
    per molecule in fit.predict_per_mol, not per pair.
    """
    gb = precompute_dots_gb(
        r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu
    )
    q = precompute_dots_quadrupole(r_mag, a_i, a_j, b_ij, Q)
    return gb + q


@dataclass
class FitData:
    """Flattened pair geometry + per-frame targets for the GBQ fit.

    Pair-level arrays (length ``P`` = total directed pairs over all frames) feed
    the vectorised potential; ``frame_index`` segment-sums them back to frames.
    Frame-level arrays (``n_mol``, ``target_per_mol``) have length ``F`` = n_frames.
    """

    r_mag: np.ndarray
    a_i: np.ndarray
    a_j: np.ndarray
    b_ij: np.ndarray
    frame_index: np.ndarray
    n_mol: np.ndarray
    target_per_mol: np.ndarray
    cutoff: float

    @property
    def n_frames(self):
        return int(self.target_per_mol.shape[0])


_CACHE_FIELDS = ("r_mag", "a_i", "a_j", "b_ij", "frame_index", "n_mol", "target_per_mol")


def build_dataset(
    path,
    cutoff,
    orientation_key="or_vec",
    energy_key="energy_pa",
    atoms_per_mol=12,
    index=":",
    cache_dir=None,
):
    """Read frames from ``path`` and flatten them into a :class:`FitData`.

    ``target_per_mol = frame.info[energy_key] * atoms_per_mol`` -- the absolute
    per-molecule DFT energy (eV/molecule); E_intra (fit later) absorbs the
    intramolecular constant. Set ``cache_dir`` to memoise the (whole-file)
    extraction in an ``.npz`` keyed by file name, cutoff and mtime.
    """
    cache_path = None
    if cache_dir is not None and index == ":":
        key = "{}_c{:g}_{}.npz".format(
            os.path.basename(path), cutoff, int(os.path.getmtime(path))
        )
        cache_path = os.path.join(cache_dir, key)
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return FitData(*(d[k] for k in _CACHE_FIELDS), cutoff=float(d["cutoff"]))

    frames = ase.io.read(path, index)
    if not isinstance(frames, list):
        frames = [frames]

    pair_blocks, index_blocks, n_mol, target = [], [], [], []
    for f, frame in enumerate(frames):
        pairs = extract_periodic_pairs(frame, orientation_key, cutoff)
        pair_blocks.append(pairs)
        index_blocks.append(np.full(len(pairs), f, dtype=int))
        n_mol.append(len(frame))
        target.append(frame.info[energy_key] * atoms_per_mol)

    pairs = np.concatenate(pair_blocks, axis=0) if pair_blocks else np.empty((0, 4))
    data = FitData(
        r_mag=pairs[:, 0],
        a_i=pairs[:, 1],
        a_j=pairs[:, 2],
        b_ij=pairs[:, 3],
        frame_index=(
            np.concatenate(index_blocks) if index_blocks else np.empty(0, dtype=int)
        ),
        n_mol=np.asarray(n_mol, dtype=float),
        target_per_mol=np.asarray(target, dtype=float),
        cutoff=float(cutoff),
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            cutoff=data.cutoff,
            **{k: getattr(data, k) for k in _CACHE_FIELDS},
        )
    return data
