import numpy as np
import ase.io
from ase.geometry import find_mic
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from asmcmc.base.trial_moves import calc_or_vec

CIF_PATH = "../data/benzene_pbca_cod_7238223.cif"
OUT_PATH = "../data/benzene_herringbone_cg.xyz"
# Oblate disc semi-axes used for the coarse-grained benzene (viz/shape only;
# the uniaxial GB potential reads orientation from or_vec, not these).
SEMI_AXES = [2.5, 2.5, 1.0]


def split_molecules(frame):
    """Split a frame into its bonded molecules.

    A neighbor list with covalent-radius cutoffs (scaled up slightly so the
    ~1.09 A C-H bonds are caught) bonds only intramolecular pairs; each
    connected component of that bond graph is one molecule. Each returned
    ``ase.Atoms`` keeps the parent cell/pbc, so a molecule wrapped across the
    periodic boundary can still be unwrapped when we coarse-grain it.
    """
    cutoffs = natural_cutoffs(frame, mult=1.2)
    nl = build_neighbor_list(frame, cutoffs, self_interaction=False, bothways=True)
    n_mol, labels = connected_components(nl.get_connectivity_matrix(sparse=True))
    return [frame[labels == i] for i in range(n_mol)]


def coarse_grain_molecule(molecule):
    """Reduce one molecule to ``(center_of_mass, ring_normal)``.

    ``ring_normal`` is the principal axis of the mass-weighted inertia tensor
    with the *largest* moment: for a planar molecule the mass sits in-plane, so
    the axis perpendicular to the plane (the benzene C6 axis) carries the
    largest moment of inertia. That unit vector becomes the oblate particle's
    ``or_vec``.
    """
    mass = molecule.get_masses()
    # unwrap: a molecule can straddle the periodic boundary, so rebuild it as a
    # contiguous cluster by adding the minimum-image displacement from atom 0.
    pos = molecule.positions
    pos = pos[0] + find_mic(pos - pos[0], molecule.cell.array, molecule.pbc)[0]

    com = mass @ pos / mass.sum()
    r = pos - com
    r2 = np.einsum("ki,ki->k", r, r)
    inertia = (mass[:, None, None] * (r2[:, None, None] * np.eye(3)
                                      - r[:, :, None] * r[:, None, :])).sum(0)
    evals, evecs = np.linalg.eigh(inertia)
    ring_normal = evecs[:, evals.argmax()]
    return com, ring_normal


def orientation_quaternion(ring_normal):
    """Scalar-first quaternion ``[w, x, y, z]`` for asmcmc's convention.

    asmcmc stores orientation as ``or_vec = R @ [0, 0, 1]`` (see
    ``trial_moves.calc_or_vec``), so we want the rotation ``R`` that carries the
    body z-axis onto ``ring_normal``. The shortest-arc rotation does this; the
    azimuth about the normal is a free gauge for a uniaxial disc (the MC samples
    it and the potential ignores it), so any such ``R`` is valid.
    """
    z = np.array([0.0, 0.0, 1.0])
    n = ring_normal / np.linalg.norm(ring_normal)
    axis = np.cross(z, n)
    sin, cos = np.linalg.norm(axis), np.clip(z @ n, -1.0, 1.0)
    if sin < 1e-12:  # already (anti)parallel to z
        rot = Rotation.identity() if cos > 0 else Rotation.from_rotvec([np.pi, 0, 0])
    else:
        rot = Rotation.from_rotvec(axis / sin * np.arccos(cos))
    # scipy is scalar-last [x,y,z,w]; roll to scalar-first [w,x,y,z] for c_q
    return np.roll(rot.as_quat(), 1)


def main():
    # ase.io.read applies the Pbca symmetry ops, so `frame` is the full unit
    # cell (the CIF's asymmetric unit is only half a benzene molecule).
    frame = ase.io.read(CIF_PATH)
    print(f"unit cell: {len(frame)} atoms  ({frame.get_chemical_formula()})")

    molecules = split_molecules(frame)
    print(f"molecules: {len(molecules)}  sizes {sorted(len(m) for m in molecules)}")

    coms, or_vecs = zip(*(coarse_grain_molecule(m) for m in molecules))
    coms, or_vecs = np.array(coms), np.array(or_vecs)
    quats = np.array([orientation_quaternion(n) for n in or_vecs])

    # the c_q we store must reproduce or_vec through asmcmc's own decoder
    round_trip = np.array([calc_or_vec(q).squeeze() for q in quats])
    assert np.allclose(round_trip, or_vecs, atol=1e-9), "c_q does not round-trip or_vec"

    # Verify the coarse-grained normals form a herringbone: the nematic order
    # of the 4 ring normals should be low (~0.25, T-shaped), not ~1 (parallel).
    Q = np.mean([1.5 * np.outer(u, u) - 0.5 * np.eye(3) for u in or_vecs], axis=0)
    S = np.linalg.eigvalsh(Q)[-1]
    print(f"nematic S of the 4 normals = {S:.3f}   (herringbone ~0.25, parallel ~1)")

    # Assemble the coarse-grained unit cell in the sampler's representation and
    # save it as the reusable herringbone motif the initializer will tile.
    cg = ase.Atoms("X" * len(molecules), positions=coms, cell=frame.cell, pbc=True)
    cg.new_array("c_q", quats)
    cg.new_array("or_vec", or_vecs)
    cg.new_array("axes", np.tile(SEMI_AXES, (len(molecules), 1)))
    cg.wrap()
    print("COMs (fractional):")
    print(np.round(cg.get_scaled_positions(), 3))

    ase.io.write(OUT_PATH, cg, format="extxyz")
    print(f"wrote {len(cg)} coarse-grained particles -> {OUT_PATH}")


if __name__ == "__main__":
    main()
