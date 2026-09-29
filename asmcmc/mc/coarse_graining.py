"""Map atomistic frames to one ellipsoid (centre and disc normal) per molecule.

This module has no optional dependencies, so it can be imported and tested from
the base install.

The main subtlety is molecules that straddle a periodic boundary. ASE's
connectivity is PBC-aware, so working out which atoms belong to each molecule
works on any frame. The positions, however, come back wrapped into the cell, so a
molecule that crosses a cell face has atoms on both sides of the box. Its naive
centroid then lands near the middle of the cell and its principal axes are
meaningless. In the experimental Pbca benzene crystal
(``data/benzene_pbca_cod_7238223.cif``) every molecule is wrapped like this, and
a naive mapping puts all four ring centres on the same point.
:func:`molecule_fragments` avoids the problem by walking the bond graph and adding
up the true bond vectors, which gives contiguous coordinates (possibly outside
the cell) that are safe to average.
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list

# Bonds are detected from covalent radii scaled by this factor. 1.2 is the usual
# ASE value: well above C-H (1.09 A) and aromatic C-C (1.39 A) bond lengths, and
# still below benzene's shortest intermolecular contacts.
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


def coarse_grain_frame(frame, mult=BOND_CUTOFF_MULT, mass_weighted=True):
    """Map an atomistic ``frame`` to one ellipsoid centre per molecule.

    Returns an :class:`ase.Atoms` of ``X`` sites carrying an ``or_vec`` array
    (unit disc normals), sharing ``frame``'s cell and pbc -- the layout
    :func:`asmcmc.mc.potentials.calc_total_energy` and
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
