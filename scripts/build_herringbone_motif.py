"""Coarse-grain a benzene Pbca cif into the 4-molecule herringbone motif.

`HerringboneLatticeInitializer` tiles this motif into a supercell, so the motif's
cell *is* the starting density of every herringbone run. That makes the choice of
cif a physical input, not bookkeeping: the tracked default
(`data/benzene_herringbone_cg.xyz`, COD 7238223) is an in-situ cryo-grown crystal
reported at 150 K whose cell is ~4% larger than benzene actually is at that
temperature -- 123.58 A^3/molecule, rho 1.050, against an accepted rho of
~1.09-1.10. Every run started from it therefore has to collapse ~4% further to
reach the Cacelli potential's preferred density.

Cacelli et al. (J. Chem. Phys. 120, 3648) start from the 138 K neutron structure
of their ref 39 (Bacon, Curry & Wilson, Proc. R. Soc. A 279, 98 (1964) = CSD
BENZEN01 = CCDC 1108750), which is `data/benzene_Pbca_csd_1108750.cif` here.

    python scripts/build_herringbone_motif.py \
        --cif data/benzene_Pbca_csd_1108750.cif \
        --out data/benzene_herringbone_cg_138K.xyz

The density check is an assertion, not a print: a cif whose coarse-grained
density disagrees with its own `_exptl_crystal_density_diffrn` means the symmetry
expansion or the fragment split went wrong, and that must not reach a run.
"""

import argparse
import warnings
from pathlib import Path

import ase.io
import numpy as np
from scipy.spatial.transform import Rotation

from asmcmc.utils import coarse_grain_frame

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (2.5, 2.5, 1.0)  # ellipsoid semiaxes, as in export_ideal_crystals.py
AVOGADRO = 6.02214076e23
BENZENE_G_PER_MOL = 78.11


def or_vec_to_quat(u):
    """Minimal rotation carrying z-hat onto each unit normal ``u``.

    Returned scalar-first [w,x,y,z], the convention `c_q` uses, so that
    `trial_moves.calc_or_vec(q)` (which is R @ z-hat) round-trips back to ``u``.
    The spin about the particle's own normal is unobservable for a uniaxial
    disc, so picking the minimal rotation costs nothing.
    """
    quats = []
    for v in u:
        axis = np.cross([0.0, 0.0, 1.0], v)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:  # already parallel or antiparallel to z
            rot = Rotation.identity() if v[2] > 0 else Rotation.from_rotvec([np.pi, 0, 0])
        else:
            rot = Rotation.from_rotvec(axis / norm * np.arccos(np.clip(v[2], -1.0, 1.0)))
        quats.append(np.roll(rot.as_quat(), 1))
    return np.array(quats)


def cif_density(path):
    """`_exptl_crystal_density_diffrn` from the cif, or None if absent."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("_exptl_crystal_density_diffrn"):
            return float(line.split()[1])
    return None


def build(cif, out):
    with warnings.catch_warnings():
        # ASE warns that it cannot interpret the orthorhombic setting of #61;
        # the assertions below (density, centre count) are what actually
        # establish the expansion was right.
        warnings.simplefilter("ignore")
        atoms = ase.io.read(str(cif))

    cg = coarse_grain_frame(atoms)
    n = len(cg)
    volume = cg.get_volume()
    rho = BENZENE_G_PER_MOL * n / (AVOGADRO * volume * 1e-24)

    assert n == 4, f"expected 4 molecules per Pbca cell, got {n}"
    reported = cif_density(cif)
    if reported is not None:
        assert abs(rho - reported) < 0.005, (
            f"coarse-grained density {rho:.4f} disagrees with the cif's "
            f"{reported:.4f} g/cm^3 -- symmetry expansion or fragment split is wrong"
        )

    u = cg.arrays["or_vec"]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    cg.arrays["or_vec"] = u
    cg.new_array("c_q", or_vec_to_quat(u))
    cg.new_array("axes", np.tile(SHAPE, (n, 1)).astype(float))

    quat_check = np.array(
        [Rotation.from_quat(np.roll(q, -1)).as_matrix() @ [0, 0, 1] for q in cg.arrays["c_q"]]
    )
    assert np.abs(quat_check - u).max() < 1e-9, "c_q does not reproduce or_vec"

    ase.io.write(str(out), cg, format="extxyz")

    order = 1.5 * np.einsum("ni,nj->ij", u, u) / n - 0.5 * np.eye(3)
    angles = np.degrees(np.arccos(np.clip(np.abs(u @ u.T), 0, 1)))
    print(f"{cif}  ->  {out}")
    print(f"  molecules            {n}")
    print(f"  cell edges (A)       {np.round(np.diag(cg.cell.array), 4)}")
    print(f"  V / molecule (A^3)   {volume / n:.2f}")
    print(f"  density (g/cm^3)     {rho:.4f}" + ("" if reported is None else f"  (cif: {reported})"))
    print(f"  nematic S            {np.linalg.eigvalsh(order)[-1]:.4f}")
    print(f"  pair angles (deg)    {np.round(np.sort(angles[np.triu_indices(n, 1)]), 1)}")
    print(f"  fractional centres   {np.round(cg.get_scaled_positions(), 3).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", default=ROOT / "data/benzene_Pbca_csd_1108750.cif")
    parser.add_argument("--out", default=ROOT / "data/benzene_herringbone_cg_138K.xyz")
    args = parser.parse_args()
    build(Path(args.cif), Path(args.out))


if __name__ == "__main__":
    main()
