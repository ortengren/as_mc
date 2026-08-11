"""Geometry helpers for turning atomistic molecular frames into ellipsoids.

Kept free of ``anisoap``/``metatensor`` (unlike :mod:`asmcmc.generate_cg_reps`,
which pulls in the optional ``[anisoap]`` extra) so the coarse-graining can be
imported and tested from the base install.

**Why this exists: molecules that straddle a periodic boundary.** ASE's
connectivity is PBC-aware, so identifying *which* atoms form a molecule works
on any frame. Their *positions*, however, come back wrapped into the cell, and
a molecule split across a face then has atoms at both edges — its naive
centroid lands near the cell centre and its principal axes are meaningless.
On the experimental Pbca benzene crystal
(``data/benzene_pbca_cod_7238223.cif``) every molecule wraps, and the
unguarded mapping collapses all four ring centres onto the same point.
:func:`molecule_fragments` avoids this by walking the bond graph and
accumulating the true bond displacement vectors, yielding contiguous
(possibly outside-the-cell) coordinates that are safe to average.
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.spatial.transform import Rotation

# Bond detection: covalent radii scaled by this factor. 1.2 is the usual ASE
# working value -- comfortably above C-H (1.09 A) and aromatic C-C (1.39 A)
# while staying below benzene's shortest intermolecular contacts.
BOND_CUTOFF_MULT = 1.2


def molecule_fragments(frame, mult=BOND_CUTOFF_MULT):
    """Split ``frame`` into molecules, with positions unwrapped across the PBC.

    Returns a list of ``(indices, positions)`` pairs, one per connected
    component of the bond graph, ordered by lowest atom index. ``positions``
    is contiguous -- it may lie outside the cell -- so centroids and principal
    axes computed from it are meaningful even when the molecule straddles a
    cell face. ``indices`` indexes back into ``frame``.
    """
    i, j, D = neighbor_list("ijD", frame, natural_cutoffs(frame, mult=mult))

    adjacency = [[] for _ in range(len(frame))]
    for a, b, d in zip(i, j, D):
        adjacency[a].append((b, d))

    fragments = []
    visited = np.zeros(len(frame), dtype=bool)
    for root in range(len(frame)):
        if visited[root]:
            continue
        # Offsets from the root, accumulated along bonds. D is the true
        # displacement (image shift already applied), so summing it along a
        # spanning walk reassembles the molecule regardless of wrapping.
        offsets = {root: np.zeros(3)}
        visited[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            for v, d in adjacency[u]:
                if v not in offsets:
                    offsets[v] = offsets[u] + d
                    visited[v] = True
                    stack.append(v)
        indices = np.array(sorted(offsets))
        positions = frame.positions[root] + np.array([offsets[k] for k in indices])
        fragments.append((indices, positions))
    return fragments


def disc_normal(positions, masses=None):
    """Unit normal of a planar (or near-planar) set of ``positions``.

    The smallest-variance principal direction: exact for a planar molecule
    such as benzene, and the natural disc axis otherwise. Sign is arbitrary --
    these particles are head-tail symmetric (``u = -u``).
    """
    centre = centre_of_mass(positions, masses)
    # Rows of Vt are principal directions, ordered by decreasing variance.
    return np.linalg.svd(positions - centre)[2][2]


def centre_of_mass(positions, masses=None):
    """Mass-weighted centre; the plain centroid when ``masses`` is None.

    For benzene the two coincide (D6h symmetry), so the choice only matters
    for lower-symmetry molecules.
    """
    if masses is None:
        return positions.mean(axis=0)
    return (masses[:, None] * positions).sum(axis=0) / masses.sum()


def quat_to_or_vec(quats):
    """Convert stored ``c_q`` quaternions to ``or_vec`` disc normals.

    The datasets store ``c_q`` in ``(w, x, y, z)`` order and take the body
    **z** axis -- the short semiaxis of the oblate ellipsoid -- as the disc
    normal. Verified against ``ellipsoids_with_axes_and_energies.xyz``
    (dot product 1.000000).
    """
    quats = np.atleast_2d(np.asarray(quats, dtype=float))
    # scipy wants (x, y, z, w); the files store (w, x, y, z).
    rotations = Rotation.from_quat(np.roll(quats, -1, axis=1))
    return rotations.apply(np.array([0.0, 0.0, 1.0]))


def coarse_grain_frame(frame, mult=BOND_CUTOFF_MULT, mass_weighted=True):
    """Map an atomistic ``frame`` to one ellipsoid centre per molecule.

    Returns an :class:`ase.Atoms` of ``X`` sites carrying an ``or_vec`` array
    (unit disc normals), sharing ``frame``'s cell and pbc -- the layout
    :func:`asmcmc.base.potentials.calc_total_energy` and
    ``fitting_gbq.data.extract_periodic_pairs`` expect.
    """
    masses = frame.get_masses() if mass_weighted else None
    centres, normals = [], []
    for indices, positions in molecule_fragments(frame, mult=mult):
        m = None if masses is None else masses[indices]
        centres.append(centre_of_mass(positions, m))
        normals.append(disc_normal(positions, m))

    cg = Atoms("X" * len(centres), positions=np.array(centres), cell=frame.cell,
               pbc=frame.pbc)
    cg.arrays["or_vec"] = np.array(normals)
    # Unwrapping can place a centre outside the cell; fold it back so output
    # matches the stored ellipsoid files. Physically a no-op under PBC.
    cg.wrap()
    return cg
