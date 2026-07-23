import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from asmcmc.utils import (
    coarse_grain_frame,
    disc_normal,
    molecule_fragments,
    quat_to_or_vec,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
PBCA_CIF = _REPO_ROOT / "data/benzene_pbca_cod_7238223.cif"
BENZENES_XYZ = _REPO_ROOT / "data/anisoap_data/benzenes/benzenes.xyz"
ELLIPSOIDS_XYZ = _REPO_ROOT / "data/anisoap_data/benzenes/ellipsoids.xyz"


def _benzene_ring(centre=(0.0, 0.0, 0.0), normal="z"):
    """A flat 6-carbon ring of radius 1.39 A, normal along ``normal``."""
    t = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.stack([1.39 * np.cos(t), 1.39 * np.sin(t), np.zeros(6)], axis=1)
    if normal == "x":
        ring = ring[:, [2, 0, 1]]
    elif normal == "y":
        ring = ring[:, [1, 2, 0]]
    return ring + np.asarray(centre)


def test_fragments_split_a_two_molecule_cell():
    pos = np.vstack([_benzene_ring((5.0, 5.0, 3.0)), _benzene_ring((5.0, 5.0, 9.0))])
    frame = Atoms("C12", positions=pos, cell=[10, 10, 12], pbc=True)
    frags = molecule_fragments(frame)
    assert len(frags) == 2
    assert sorted(len(idx) for idx, _ in frags) == [6, 6]


# The regression this module exists for: an unguarded centroid over wrapped
# positions collapses toward the cell centre. Here the ring is centred on a
# cell face, so its atoms sit at both edges along x.
def test_wrapped_molecule_is_unwrapped_before_averaging():
    # Ring plane contains x, centred on the x=0 face, so wrapping splits it.
    ring = _benzene_ring((0.0, 5.0, 5.0), normal="z")
    frame = Atoms("C6", positions=ring, cell=[10, 10, 10], pbc=True)
    frame.wrap()
    # Wrapping really did split it: raw positions straddle both x faces.
    raw = frame.get_positions()
    assert raw[:, 0].max() - raw[:, 0].min() > 5.0
    assert np.abs(raw.mean(axis=0)[0] - 5.0) < 1.0  # naive centroid: cell centre

    (idx, unwrapped), = molecule_fragments(frame)
    assert len(idx) == 6
    assert unwrapped[:, 0].max() - unwrapped[:, 0].min() < 3.0
    # True centre is on the x=0 face, not the middle of the cell.
    centre = unwrapped.mean(axis=0)
    assert min(abs(centre[0]), abs(centre[0] - 10.0)) < 1e-6
    np.testing.assert_allclose(centre[1:], [5.0, 5.0], atol=1e-6)
    # ... and the naive average over the wrapped positions gets it wrong.
    assert abs(raw.mean(axis=0)[0] - 5.0) < 1e-6


def test_disc_normal_recovers_the_ring_axis():
    for axis, expected in [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])]:
        u = disc_normal(_benzene_ring((1.0, 2.0, 3.0), normal=axis))
        assert abs(abs(np.dot(u, expected)) - 1.0) < 1e-9


@pytest.mark.skipif(not PBCA_CIF.exists(), reason="Pbca CIF not present")
def test_pbca_crystal_maps_to_four_distinct_ellipsoids():
    frame = ase.io.read(PBCA_CIF)
    cg = coarse_grain_frame(frame)
    assert len(cg) == 4

    # Every Pbca molecule straddles a boundary; the unguarded mapping put all
    # four centres on the same point (the cell centre). Assert they are apart.
    d = cg.get_all_distances(mic=True)
    assert d[np.triu_indices(4, k=1)].min() > 2.0

    assert frame.get_volume() / len(cg) == pytest.approx(123.58, abs=0.01)
    # Herringbone: the four intra-cell normals fall into two families, a
    # near-perpendicular edge-to-face pair (~86 deg) and a ~28 deg pair.
    # (A 0 deg family exists too, but only between periodic self-images.)
    b = np.abs(cg.arrays["or_vec"] @ cg.arrays["or_vec"].T)
    angles = np.degrees(np.arccos(np.clip(b[np.triu_indices(4, k=1)], 0, 1)))
    assert ((angles > 80.0) & (angles < 90.0)).sum() == 4
    assert ((angles > 25.0) & (angles < 32.0)).sum() == 2


@pytest.mark.skipif(
    not (BENZENES_XYZ.exists() and ELLIPSOIDS_XYZ.exists()),
    reason="anisoap_data drop not present",
)
def test_mapping_reproduces_the_reference_ellipsoid_file():
    atomistic = ase.io.read(BENZENES_XYZ, ":20")
    reference = ase.io.read(ELLIPSOIDS_XYZ, ":20")

    for frame, ref in zip(atomistic, reference):
        cg = coarse_grain_frame(frame)
        assert len(cg) == len(ref)

        # Bead order need not match; pair up by minimum-image distance.
        delta = cg.get_positions()[:, None, :] - ref.get_positions()[None, :, :]
        frac = np.linalg.solve(np.array(frame.cell).T, delta.reshape(-1, 3).T).T
        frac -= np.round(frac)
        dist = np.linalg.norm(frac @ np.array(frame.cell), axis=1).reshape(len(cg), -1)
        match = dist.argmin(axis=1)
        assert dist[np.arange(len(cg)), match].max() < 1e-4

        # Orientation: |u . u_ref| = 1 up to the head-tail sign. The ~1e-3
        # floor is anisoap's own ellipsoid fit in the stored c_q, not a
        # convention difference here -- mass-weighted, unweighted and
        # carbons-only normals agree with each other to 1e-6, and the
        # molecules are planar to 1.1e-4 A.
        ref_normals = quat_to_or_vec(ref.arrays["c_q"])[match]
        dots = np.abs(np.einsum("ij,ij->i", cg.arrays["or_vec"], ref_normals))
        assert np.abs(dots - 1.0).max() < 2e-3
