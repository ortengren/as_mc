import os
from dataclasses import dataclass
from functools import cached_property

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


# The optional sum_sq/diff_sq/b_sq args are the geometry-only squares
# (a_i+a_j)^2, (a_i-a_j)^2, b_ij^2. They depend only on the pair geometry, not
# on any fitted parameter, so the fit precomputes them once (FitData.gb_geom)
# and threads them through to skip rebuilding them on every evaluation. Default
# None recomputes them, keeping every caller (e.g. gbq) backward-compatible.
def precompute_dots_gb_shape_func(a_i, a_j, b_ij, sigma0, kappa, sum_sq=None, diff_sq=None):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    if sum_sq is None:
        sum_sq = (a_i + a_j) ** 2
    if diff_sq is None:
        diff_sq = (a_i - a_j) ** 2
    term1 = sum_sq / (1 + chi * b_ij)
    term2 = diff_sq / (1 - chi * b_ij)
    sigma = sigma0 / np.sqrt(1 - (chi / 2) * (term1 + term2))
    return sigma


def precompute_dots_gb_axial_energy(b_ij, kappa, b_sq=None):
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    if b_sq is None:
        b_sq = b_ij**2
    # (chi * b_ij)**2 == chi**2 * b_sq
    return 1 / np.sqrt(1 - (chi**2) * b_sq)


def precompute_dots_gb_directional_energy(a_i, a_j, b_ij, kappa_prime, mu, sum_sq=None, diff_sq=None):
    chi_prime = (kappa_prime ** (1 / mu) - 1) / (kappa_prime ** (1 / mu) + 1)
    if sum_sq is None:
        sum_sq = (a_i + a_j) ** 2
    if diff_sq is None:
        diff_sq = (a_i - a_j) ** 2
    term1 = sum_sq / (1 + chi_prime * b_ij)
    term2 = diff_sq / (1 - chi_prime * b_ij)
    return 1 - (chi_prime / 2) * (term1 + term2)


def precompute_dots_gb_en_func(
    a_i, a_j, b_ij, eps0, kappa, kappa_prime, mu, nu, sum_sq=None, diff_sq=None, b_sq=None
):
    eps1 = precompute_dots_gb_axial_energy(b_ij, kappa, b_sq=b_sq)
    eps2 = precompute_dots_gb_directional_energy(
        a_i, a_j, b_ij, kappa_prime, mu, sum_sq=sum_sq, diff_sq=diff_sq
    )
    return eps0 * (eps1**nu) * (eps2**mu)


def precompute_dots_gb(
    r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu, xi,
    sum_sq=None, diff_sq=None, b_sq=None,
):
    if sum_sq is None:
        sum_sq = (a_i + a_j) ** 2
    if diff_sq is None:
        diff_sq = (a_i - a_j) ** 2
    if b_sq is None:
        b_sq = b_ij**2
    eps = precompute_dots_gb_en_func(
        a_i, a_j, b_ij, eps0, kappa, kappa_prime, mu, nu,
        sum_sq=sum_sq, diff_sq=diff_sq, b_sq=b_sq,
    )
    sigma = precompute_dots_gb_shape_func(
        a_i, a_j, b_ij, sigma0, kappa, sum_sq=sum_sq, diff_sq=diff_sq
    )
    # sigma already folds in sigma0 (shape_func returns sigma0 / sqrt(...)), so
    # this matches potentials.gb's xi*sigma0 / (r - sigma0*shape + xi*sigma0).
    term = xi * sigma0 / (r_mag - sigma + xi * sigma0)
    # term**12 - term**6 == t6*(t6 - 1); one expensive array pow instead of two.
    t6 = term**6
    return 4 * eps * t6 * (t6 - 1)


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


def gbq(r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q):
    """Pairwise GB + quadrupole energy (vectorises over pair arrays).

    E_intra is intentionally absent: it is a per-molecule constant added once
    per molecule in fit.predict_per_mol, not per pair.
    """
    gb = precompute_dots_gb(
        r_mag, a_i, a_j, b_ij, sigma0, eps0, kappa, kappa_prime, mu, nu, xi
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

    @cached_property
    def quad_geom_per_frame(self):
        """Per-frame, geometry-only quadrupole factor; quad energy = ``Q**2 *`` it.

        The quadrupole pair energy is ``Q**2`` times a purely geometric factor
        (``0.75 * s / r^5``; see :func:`precompute_dots_quadrupole`), so a
        frame's quadrupole lattice energy factorises as
        ``Q**2 * quad_geom_per_frame``. ``Q`` is the only quadrupole parameter,
        so this array depends on geometry alone and is identical across every
        fit evaluation. Precomputing it once -- with the same
        ``0.5 * (.) / n_mol`` directed-pair / per-molecule reduction that
        :func:`fit.predict_per_mol` applies -- removes the entire per-pair
        quadrupole recomputation from the DE inner loop. Memoised on first use
        (``cached_property``: computed lazily, then stored on the instance).
        """
        s = (
            1
            + 2 * (self.b_ij**2)
            - 5 * (self.a_i**2 + self.a_j**2)
            - 20 * self.a_i * self.a_j * self.b_ij
            + 35 * (self.a_i**2) * (self.a_j**2)
        )
        unit_pair = 0.75 * s / (self.r_mag**5)
        frame = np.bincount(
            self.frame_index, weights=unit_pair, minlength=self.n_frames
        )
        return 0.5 * frame / self.n_mol

    @cached_property
    def gb_geom(self):
        """Geometry-only GB squares ``(sum_sq, diff_sq, b_sq)``, reused per eval.

        ``sum_sq = (a_i + a_j)**2``, ``diff_sq = (a_i - a_j)**2``,
        ``b_sq = b_ij**2`` -- the parameter-independent pieces of the GB shape /
        axial / directional terms. Precomputed once and threaded into
        :func:`precompute_dots_gb` so they are not rebuilt on every fit
        evaluation. Memoised on first use (``cached_property``).
        """
        return (
            (self.a_i + self.a_j) ** 2,
            (self.a_i - self.a_j) ** 2,
            self.b_ij**2,
        )


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
