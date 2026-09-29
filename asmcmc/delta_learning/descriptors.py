"""Turn a dimer dataset into AniSOAP descriptors that line up with the fit targets.

The pipeline has three steps.

First, :func:`load_training_set` reads the campaign once. It pulls only what the
fit needs out of the extxyz shards (centres, principal axes and the Delta target)
into flat arrays, cached as one small ``.npz``. The shards are about 32 MB of
extxyz with ``info`` fields stored as JSON strings, so parsing them repeatedly
would be slow. The geometry is taken from the stored ``molecular_com`` and
``principal_axes`` rather than by coarse-graining the atoms again, as in
``dataset_analysis.pair_records``, so the numbers are exactly the ones the
generator labelled.

Second, :func:`ellipsoid_frames` turns this into AniSOAP input by setting
``c_q`` and ``c_diameter[1..3]`` on each frame. The quaternion comes from the
principal axes via :func:`quaternions_from_axes`, and converting it back with
``asmcmc.mc.trial_moves.quat_to_or_vec`` gives the stored ``or_vec``.

Third, :func:`descriptors` computes the descriptors and puts each row back with
the frame it belongs to. This needs care because AniSOAP silently drops rows;
see :func:`descriptor_rows`.

The ellipsoids are uniaxial: both in-plane semiaxes are equal
(``Hypers.semiaxis_ab``), so rotating a particle about its disc normal doesn't
change its descriptor. That's what allows an MC frame, which only stores
``or_vec``, to be given an arbitrary azimuth (:func:`quaternions_from_normals`).
A test checks this invariance.

This module only computes; it doesn't write any files. All of that lives in
:mod:`asmcmc.delta_learning.sweep`, so moving the sweep's bookkeeping elsewhere
(to signac, say) wouldn't touch this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

from asmcmc.delta_learning.dataset import dataset_frames

# Frames are isolated dimers (pbc=False); the cell only has to be far larger than
# any dimer.
NONPERIODIC_CELL = 100.0

# Names AniSOAP has used for the sample dimension that holds the structure index.
# The installed version calls it "type", which is misleading (every bead here is
# the same species), so the name can't be trusted on its own and
# _structure_column checks whichever column it picks. See descriptor_rows.
STRUCTURE_ALIASES = ("structure", "system", "type")


@dataclass(frozen=True)
class Hypers:
    """One AniSOAP hyperparameter point.

    ``semiaxis_ab`` and ``semiaxis_c`` are semiaxes in Angstrom. AniSOAP reads
    ``c_diameter[i]`` and halves it, so :func:`ellipsoid_frames` writes twice
    these values. Using one value for both in-plane axes keeps the ellipsoids
    uniaxial (see the module docstring).

    :meth:`to_dict` is flat and uses only JSON types. It identifies a sweep
    point, and would also work as a signac state point if the sweep's
    bookkeeping moved there. :attr:`key` is the same identity as a string that
    is safe to use as a directory name. It rounds explicitly rather than using
    ``repr``, so it's the same in every process and when a run is resumed.
    """

    max_angular: int = 9
    max_radial: int = 6
    cutoff_radius: float = 6.0
    radial_gaussian_width: float = 1.5
    semiaxis_ab: float = 2.5
    semiaxis_c: float = 1.0
    radial_basis_name: str = "gto"
    subtract_center_contribution: bool = True

    @property
    def key(self):
        return (
            f"l{self.max_angular}-n{self.max_radial}"
            f"-rc{self.cutoff_radius:.2f}-g{self.radial_gaussian_width:.3f}"
            f"-ab{self.semiaxis_ab:.3f}-c{self.semiaxis_c:.3f}"
            f"-{self.radial_basis_name}-scc{int(self.subtract_center_contribution)}"
        )

    @property
    def diameters(self):
        return (2.0 * self.semiaxis_ab, 2.0 * self.semiaxis_ab, 2.0 * self.semiaxis_c)

    @property
    def n_features(self):
        """Width of the power spectrum at these hypers.

        :func:`descriptor_rows` needs the width even when AniSOAP returns nothing
        to measure it from (when every centre is beyond the cutoff). AniSOAP
        doesn't provide it, so it's calculated here, and
        ``tests/test_anisoap_fit.py`` checks the formula against the actual width
        at the corners of the grid.
        """
        return (self.max_radial + 1) ** 2 * (self.max_angular + 1)

    def to_dict(self):
        return {
            "max_angular": int(self.max_angular),
            "max_radial": int(self.max_radial),
            "cutoff_radius": float(self.cutoff_radius),
            "radial_gaussian_width": float(self.radial_gaussian_width),
            "semiaxis_ab": float(self.semiaxis_ab),
            "semiaxis_c": float(self.semiaxis_c),
            "radial_basis_name": str(self.radial_basis_name),
            "subtract_center_contribution": bool(self.subtract_center_contribution),
        }

    @classmethod
    def from_dict(cls, mapping):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in mapping.items() if k in fields})

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class TrainingSet:
    """A campaign as flat arrays: one row per molecule, one entry per frame.

    CSR layout (molecule ``m`` of frame ``f`` is row ``offsets[f] + m``), so
    frames of any size stay plain 2-D float arrays that an ``.npz`` round-trips
    exactly.

    ``delta`` is the per-frame fit target in eV (``E_UMA - E_GBQ``). ``min_pair_r``
    is the smallest centre-centre separation in the frame: it decides both which
    frames a given cutoff can see at all and which count as "well region" when
    scoring.
    """

    com: np.ndarray
    axes: np.ndarray
    offsets: np.ndarray
    delta: np.ndarray
    e_uma: np.ndarray
    e_gbq: np.ndarray
    min_pair_r: np.ndarray
    n_molecules: np.ndarray

    def __len__(self):
        return len(self.delta)

    def counts(self):
        return np.diff(self.offsets)

    def frame_slice(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def subset(self, idx):
        """A :class:`TrainingSet` holding only frames ``idx``, re-offset."""
        idx = np.asarray(idx, dtype=int)
        counts = self.counts()[idx]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        rows = (
            np.concatenate(
                [np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]
            ).astype(int)
            if len(idx)
            else np.zeros(0, dtype=int)
        )
        return TrainingSet(
            com=self.com[rows],
            axes=self.axes[rows],
            offsets=offsets,
            delta=self.delta[idx],
            e_uma=self.e_uma[idx],
            e_gbq=self.e_gbq[idx],
            min_pair_r=self.min_pair_r[idx],
            n_molecules=self.n_molecules[idx],
        )


def _campaign_signature(campaign_dir):
    """Fingerprint of the shards on disk, so an out-of-date parse cache is rebuilt.

    This identifies the dataset, not a sweep point. It makes sure that after a
    campaign is extended or regenerated, the cache built from the old shards
    isn't silently reused.
    """
    campaign_dir = Path(campaign_dir)
    parts = []
    for path in sorted(campaign_dir.glob("clusters_shard*.xyz")):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    if not parts:
        raise FileNotFoundError(f"no clusters_shard*.xyz under {campaign_dir}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_training_set(campaign_dir, cache_dir=None, refresh=False):
    """Read a campaign into a :class:`TrainingSet`, caching the extraction."""
    campaign_dir = Path(campaign_dir)
    signature = _campaign_signature(campaign_dir)

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"geometry_{campaign_dir.name}_{signature}.npz"
        if cache_path.exists() and not refresh:
            with np.load(cache_path) as handle:
                return TrainingSet(**{k: handle[k] for k in handle.files})

    com, axes = [], []
    delta, e_uma, e_gbq, min_pair_r, n_mol = [], [], [], [], []

    for frame in dataset_frames(campaign_dir):
        centres = np.asarray(frame.info["molecular_com"], dtype=float)
        principal = np.asarray(frame.info["principal_axes"], dtype=float)
        uma = float(frame.info["interaction_energy"])
        gbq = float(frame.info["gbq_interaction_energy"])

        i, j = np.triu_indices(len(centres), k=1)
        separations = np.linalg.norm(centres[j] - centres[i], axis=1)

        com.append(centres)
        axes.append(principal)
        e_uma.append(uma)
        e_gbq.append(gbq)
        delta.append(uma - gbq)
        min_pair_r.append(float(separations.min()))
        n_mol.append(len(centres))

    counts = np.array(n_mol, dtype=int)
    training_set = TrainingSet(
        com=np.concatenate(com).astype(float),
        axes=np.concatenate(axes).astype(float),
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(int),
        delta=np.asarray(delta, dtype=float),
        e_uma=np.asarray(e_uma, dtype=float),
        e_gbq=np.asarray(e_gbq, dtype=float),
        min_pair_r=np.asarray(min_pair_r, dtype=float),
        n_molecules=counts,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **training_set.__dict__)
    return training_set


def quaternions_from_axes(axes):
    """``(w, x, y, z)`` quaternions from stacked principal-axis matrices.

    ``axes`` is ``(n, 3, 3)`` with the axes as columns, as ``dataset`` writes
    them, and the third column is the disc normal. A principal-axis matrix is
    orthogonal but not necessarily a rotation: the eigenvector signs are
    arbitrary, so about half of them come out left-handed. Flipping the third
    column of those fixes the handedness without moving the normal's axis (the
    disc is head-tail symmetric, so the normal's sign has no physical meaning).

    ``Rotation.as_quat`` returns ``(x, y, z, w)``, but AniSOAP's
    ``rotation_type="quaternion"`` expects the scalar first, hence the roll.
    Getting this wrong silently breaks rotational invariance, which
    ``test_descriptor_is_rigid_rotation_invariant`` checks for.
    """
    axes = np.asarray(axes, dtype=float)
    if axes.ndim == 2:
        axes = axes[None]
    matrices = axes.copy()
    left_handed = np.linalg.det(matrices) < 0
    matrices[left_handed, :, 2] *= -1.0
    return np.roll(Rotation.from_matrix(matrices).as_quat(), 1, axis=1)


def quaternions_from_normals(normals, rng=None):
    """``(w, x, y, z)`` quaternions that rotate the body z-axis onto each disc normal.

    This is what MC frames need, since they only store ``or_vec``. The azimuth
    about the normal is undefined, so it's chosen arbitrarily (at random if
    ``rng`` is given, otherwise deterministically). That's only valid for
    uniaxial ellipsoids; see the module docstring.
    """
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Any vector not parallel to the normal seeds the in-plane frame.
    seed = np.where(
        np.abs(normals[:, 0:1]) < 0.9,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    e1 = np.cross(seed, normals)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    if rng is not None:
        angle = rng.uniform(0.0, 2.0 * np.pi, size=len(normals))[:, None]
        e1 = np.cos(angle) * e1 + np.sin(angle) * np.cross(normals, e1)
    e2 = np.cross(normals, e1)

    matrices = np.stack([e1, e2, normals], axis=2)
    return np.roll(Rotation.from_matrix(matrices).as_quat(), 1, axis=1)


def stamp_ellipsoid(frame, quaternions, hypers):
    """Attach ``c_q`` + ``c_diameter[1..3]`` to ``frame``, in place."""
    frame.arrays["c_q"] = np.asarray(quaternions, dtype=float)
    for axis, diameter in zip((1, 2, 3), hypers.diameters):
        frame.arrays[f"c_diameter[{axis}]"] = np.full(len(frame), float(diameter))
    return frame


def make_ellipsoid_frame(centres, quaternions, hypers, cell=None, pbc=False):
    """One AniSOAP-ready frame from centres + orientations."""
    centres = np.asarray(centres, dtype=float)
    if cell is None:
        cell = np.eye(3) * NONPERIODIC_CELL
    frame = Atoms("X" * len(centres), positions=centres, cell=cell, pbc=pbc)
    return stamp_ellipsoid(frame, quaternions, hypers)


def ellipsoid_frames(training_set, hypers):
    """A :class:`TrainingSet` as AniSOAP-ready one-bead-per-molecule frames.

    Cheap enough to redo per hyperparameter point: the quaternions do not depend
    on the hypers but the diameters do, and rebuilding costs far less than
    caching a frame list per point.
    """
    frames = []
    for i in range(len(training_set)):
        rows = training_set.frame_slice(i)
        frames.append(
            make_ellipsoid_frame(
                training_set.com[rows],
                quaternions_from_axes(training_set.axes[rows]),
                hypers,
            )
        )
    return frames


def anisoap_projection(hypers):
    """The configured ``EllipsoidalDensityProjection``.

    Imported lazily: ``anisoap`` pulls in a compiled extension and metatensor,
    and importing ``asmcmc`` must not require either.
    """
    from anisoap.representations import EllipsoidalDensityProjection

    return EllipsoidalDensityProjection(
        max_angular=int(hypers.max_angular),
        max_radial=int(hypers.max_radial),
        radial_basis_name=hypers.radial_basis_name,
        # Both must be float: AniSOAP rejects an int cutoff outright (overflow
        # risk in the moment recursion).
        cutoff_radius=float(hypers.cutoff_radius),
        radial_gaussian_width=float(hypers.radial_gaussian_width),
        rotation_key="c_q",
        rotation_type="quaternion",
        subtract_center_contribution=bool(hypers.subtract_center_contribution),
        basis_rcond=1e-8,
        basis_tol=1e-3,
    )


def _structure_column(samples, n_frames, bead_counts):
    """Index of the sample dimension holding the structure id, validated.

    The installed AniSOAP calls this dimension ``"type"`` even though it holds
    the structure index, so the name is only a hint and the column is checked:
    every structure id must be in range, and no structure can have more rows than
    it has beads. If the column were really a species label (every bead here is
    species X), it would be constant, and the row-count check would fail with an
    error instead of silently adding the whole dataset into frame 0.
    """
    names = list(samples.names)
    ordered = [names.index(n) for n in STRUCTURE_ALIASES if n in names]
    ordered += [i for i in range(len(names)) if i not in ordered]

    values = np.asarray(samples.values)
    for column in ordered:
        ids = values[:, column].astype(int)
        if len(ids) and (ids.min() < 0 or ids.max() >= n_frames):
            continue
        if np.all(np.bincount(ids, minlength=n_frames) <= bead_counts):
            return column
    raise RuntimeError(
        "cannot identify the structure index among AniSOAP sample dimensions "
        f"{names}; descriptors cannot be aligned to targets"
    )


def _any_pair_within_cutoff(frames, cutoff):
    """True if any frame has two centres closer than ``cutoff``.

    The frames are isolated clusters (``pbc=False``), so plain pairwise
    distances are enough, with no minimum-image convention.
    """
    for frame in frames:
        positions = frame.get_positions()
        if len(positions) < 2:
            continue
        i, j = np.triu_indices(len(positions), k=1)
        if (
            len(i)
            and np.min(np.linalg.norm(positions[j] - positions[i], axis=1)) < cutoff
        ):
            return True
    return False


def descriptor_rows(frames, hypers, projection=None):
    """Raw per-centre power-spectrum rows and the frame each belongs to.

    Returns ``(values, structure_ids, n_features)``.

    AniSOAP returns no row for a centre with no neighbour inside the cutoff, and
    no rows at all for a frame where every centre is isolated. Nothing in the
    result says that rows are missing, so pairing the values with a target array
    by position would misalign the dataset and the fit would train on mismatched
    pairs. The structure ids returned here are the original frame indices (they
    skip dropped frames rather than renumbering), which is what lets
    :func:`descriptors` put the rows back in the right place.

    When no frame has a neighbour inside the cutoff, AniSOAP raises an error from
    inside ``cg_combine``, because it takes a maximum over angular channels that
    don't exist. What we want in that case is an all-zero descriptor.
    """
    projection = projection or anisoap_projection(hypers)
    if not _any_pair_within_cutoff(frames, hypers.cutoff_radius):
        return (
            np.zeros((0, hypers.n_features), dtype=float),
            np.zeros(0, dtype=int),
            hypers.n_features,
        )

    tensor = projection.power_spectrum(frames, mean_over_samples=False)
    block = tensor.block()

    # (rows, components, features), with a single l=0 component under lcut=0.
    values = np.asarray(block.values, dtype=float)
    values = values.reshape(values.shape[0], -1)

    bead_counts = np.array([len(f) for f in frames], dtype=int)
    column = _structure_column(block.samples, len(frames), bead_counts)
    ids = np.asarray(block.samples.values)[:, column].astype(int)
    return values, ids, values.shape[1]


def descriptors(frames, hypers, projection=None):
    """Per-frame descriptors, ``(len(frames), n_features)``.

    Per-centre rows are summed over each frame rather than averaged. The target
    is an interaction energy, which is extensive, so a linear model on the sum is
    a sum of per-centre energies. This is the usual local-energy decomposition,
    and it's the form the sampler would use.

    A frame with nothing inside the cutoff gets an all-zero row, since the model
    must predict exactly zero interaction there. Together with a fit that has no
    intercept, this gives the model the correct dissociation limit automatically.
    """
    values, ids, n_features = descriptor_rows(frames, hypers, projection=projection)
    matrix = np.zeros((len(frames), n_features), dtype=float)
    np.add.at(matrix, ids, values)
    return matrix


def campaign_descriptors(training_set, hypers):
    """:func:`ellipsoid_frames` + :func:`descriptors` for a whole campaign."""
    return descriptors(ellipsoid_frames(training_set, hypers), hypers)


