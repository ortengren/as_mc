"""Dimer dataset -> ellipsoid frames -> AniSOAP descriptors, aligned to the fit targets.

Three jobs, in the order the pipeline uses them.

**1. Read the campaign once.** :func:`load_training_set` pulls only what the fit
needs out of the extxyz shards (centres, principal axes, the Delta target) into flat
CSR-style arrays cached as one small ``.npz``. The shards are ~32 MB of extxyz whose
``info`` fields are ``_JSON`` blobs, so parsing them repeatedly would be inefficient.

Geometry is read from the stored ``molecular_com``/``principal_axes`` rather
than re-coarse-graining the atoms, for the same reason
``dataset_analysis.pair_records`` does: the numbers are then exactly the ones the
generator labelled, not a re-derivation that might differ in the last digits.

**2. Turn it into AniSOAP input.** :func:`ellipsoid_frames` stamps ``c_q``
and ``c_diameter[1..3]``. The quaternion comes from the principal-axis frame via
:func:`quaternions_from_axes`, which round-trips through
``asmcmc.mc.trial_moves.quat_to_or_vec`` back to the stored ``or_vec``.

**3. Compute descriptors that line up with the targets.**
:func:`descriptors` is where the one genuinely dangerous property of the AniSOAP
API is handled -- see :func:`descriptor_rows`.

**Uniaxial ellipsoids.** Both in-plane semiaxes are equal
(``Hypers.semiaxis_ab``), so rotating a particle about its disc normal changes no
descriptor. That is what lets an MC frame, which stores only ``or_vec``, be
featurised with an arbitrary azimuth (:func:`quaternions_from_normals`); a test
pins the invariance.

This module is pure computation and has no output directories or result files. All
persistence lives in :mod:`asmcmc.delta_learning.sweep`, so swapping the sweep's
bookkeeping (e.g. onto signac) touches that module and nothing here.
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

# Sample-dimension names under which AniSOAP has shipped the structure index.
# The installed version labels it "type", which is not what it holds -- every
# bead here is the same species -- so the name alone is not trustworthy and
# _structure_column validates whatever it picks. See descriptor_rows.
STRUCTURE_ALIASES = ("structure", "system", "type")


@dataclass(frozen=True)
class Hypers:
    """One AniSOAP hyperparameter point.

    ``semiaxis_ab``/``semiaxis_c`` are **semiaxes** in Angstrom; AniSOAP reads
    ``c_diameter[i]`` and halves it, so :func:`ellipsoid_frames` stamps twice
    these. A single in-plane value enforces the uniaxial restriction the module
    docstring explains.

    :meth:`to_dict` is deliberately flat and JSON-primitive: it is the identity
    of a sweep point, and doubles as a valid signac state point should the
    sweep's bookkeeping move there. :attr:`key` is the same identity as a
    filesystem-safe string, rounded explicitly rather than via ``repr`` so it is
    stable across processes and resumable runs.
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

        Needed because :func:`descriptor_rows` has to know the shape even when
        AniSOAP returns nothing at all to measure it from (every centre beyond
        the cutoff). AniSOAP does not expose this, so it is *derived*, and
        ``tests/test_anisoap_fit.py`` pins it against the realised width
        across the grid corners rather than trusting the arithmetic.
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
    """Fingerprint of the shards on disk, so a stale parse cache invalidates.

    This identifies the *dataset*, not a sweep point -- it exists so that
    extending or regenerating a campaign cannot silently be served from a cache
    built against the old shards.
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

    ``axes`` is ``(n, 3, 3)`` with **columns** as axes, the layout
    ``dataset`` writes, whose third column is the disc normal. A
    principal-axis matrix is orthogonal but not necessarily a *rotation*: the
    eigenvector signs are arbitrary, so roughly half come out left-handed.
    Flipping the third column on those fixes the handedness without moving the
    normal's axis (the disc is head-tail symmetric, so its sign is not physical).

    ``Rotation.as_quat`` returns ``(x, y, z, w)``; AniSOAP's
    ``rotation_type="quaternion"`` wants scalar-first, hence the roll. Getting
    this backwards silently breaks rotational invariance, which is what
    ``test_descriptor_is_rigid_rotation_invariant`` exists to catch.
    """
    axes = np.asarray(axes, dtype=float)
    if axes.ndim == 2:
        axes = axes[None]
    matrices = axes.copy()
    left_handed = np.linalg.det(matrices) < 0
    matrices[left_handed, :, 2] *= -1.0
    return np.roll(Rotation.from_matrix(matrices).as_quat(), 1, axis=1)


def quaternions_from_normals(normals, rng=None):
    """``(w, x, y, z)`` quaternions taking body **z** onto each disc normal.

    The deployment path: an MC frame stores ``or_vec`` only, so the azimuth about
    the normal is undefined and is chosen arbitrarily (randomly when ``rng`` is
    given, deterministically otherwise). Valid only for uniaxial ellipsoids --
    see the module docstring.
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

    The installed AniSOAP names this dimension ``"type"`` even though it holds
    the structure index, so the name is a hint and the validation is the real
    evidence: a structure id must be in range, and a structure cannot contribute
    more rows than it has beads. Were the column actually a species label (every
    bead here is species X) it would be constant, and the row-count check fails
    loudly instead of silently summing the whole dataset into frame 0.
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

    The frames are isolated clusters (``pbc=False``), so this is a plain
    pairwise distance -- no minimum-image convention to get wrong.
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

    AniSOAP emits no row for a centre with no neighbour inside the cutoff, and no
    rows at all for a frame where every centre is isolated. Nothing in the returned
    object flags the omission, so zipping the values against a target array
    positionally misaligns the dataset and the fit trains on mismatched pairs. The
    structure ids recovered here are the original frame indices (they skip dropped
    frames rather than renumbering), which is what lets :func:`descriptors` put the
    rows back where they belong.

    When no frame has a neighbour inside the cutoff, AniSOAP raises from inside
    ``cg_combine``, because it takes a max over angular channels that do not exist.
    The right answer for our use case is an all-zero descriptor.
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

    Per-centre rows are summed into their frame, not averaged: the target is
    an interaction energy, which is extensive, so a linear model on the sum is a
    sum of per-centre energies. This is the standard local-energy decomposition, and
    the form a deployment inside the sampler would use.

    A frame with nothing inside the cutoff gets an all-zero row, since a model must
    predict exactly zero interaction there. Paired with a no-intercept fit that gives
    the model the correct dissociation limit by construction.
    """
    values, ids, n_features = descriptor_rows(frames, hypers, projection=projection)
    matrix = np.zeros((len(frames), n_features), dtype=float)
    np.add.at(matrix, ids, values)
    return matrix


def campaign_descriptors(training_set, hypers):
    """:func:`ellipsoid_frames` + :func:`descriptors` for a whole campaign."""
    return descriptors(ellipsoid_frames(training_set, hypers), hypers)


