"""Generate a UMA-labelled benzene dimer/trimer dataset for AniSOAP training.

The labeller is Meta FAIR Chemistry's OMol-trained UMA MLIP, which clears the
physics gate in :mod:`asmcmc.utils.validation` first: r = 0.994 / RMSE 0.55 kcal/mol
against all 197 Cacelli MP2 dimer rows, all three wells bound and placed within
~0.1 A, having never seen that data.

Two design points are worth stating because they are not obvious:

**Range.** ``max_com_distance`` is 15 A, not the 9 A that spans the dimer
wells, because that is what consumes the data: ``MetropolisCalculator`` uses
``nl_radius = 15.0`` and ``generate_cg_reps.get_rep_raw`` a 15 A descriptor
cutoff. In a real 100 K frame, 55 pairs per molecule sit beyond 9 A carrying
-0.96 kcal/mol/molecule (9% of cohesion); a mere 0.01 kcal/mol systematic
error on each would sum to 0.55 kcal/mol/molecule. The tail is individually
negligible and collectively is not.

**Rigid monomers** (``rigid=True``). AniSOAP represents a molecule as a rigid
ellipsoid and the MC sampler is rigid-body, so neither can see intramolecular
distortion -- vibrating the monomers adds scatter (measured: +/-0.023 kcal/mol
on the cofacial well, 1% of its depth) that the model structurally cannot fit.
It also forces a separate monomer evaluation per cluster. Rigid makes the
monomer reference a single constant: a dimer costs 1 MLIP call instead of 3,
a trimer 4 instead of 7. ``rigid=False`` restores per-cluster distortion for a
future flexible model.

Each saved frame carries:
  * total energy and atomic forces in a ``SinglePointCalculator``
  * ``arrays["molecule_id"]``      : molecule membership per atom
  * ``info["molecular_com"]``      : (n_mol, 3), Angstrom
  * ``info["inertia_tensor"]``     : (n_mol, 3, 3), amu Angstrom^2
  * ``info["principal_moments"]``  : (n_mol, 3)
  * ``info["principal_axes"]``     : (n_mol, 3, 3)
  * ``info["molecular_force"]``    : (n_mol, 3), eV/Angstrom
  * ``info["molecular_torque"]``   : (n_mol, 3), eV
  * ``info["or_vec"]``             : (n_mol, 3) disc normals, the GB+Q orientation
  * ``info["monomer_energies"]``   : isolated monomer references, eV
  * ``info["interaction_energy"]`` : E(cluster) - sum E(monomers), eV
  * ``info["gbq_interaction_energy"]`` : the same quantity under CACELLI_POTENTIAL,
    so the Delta-learning target E_UMA - E_GBQ needs no geometry re-derivation
  * for trimers under ``decomposition="full"``: ``pair_energies``,
    ``pair_interaction_energies``, ``three_body_energy``

Usage::

    python -m asmcmc.data_preparation.cluster_dataset --n-configs 500 --out-dir results/clusters/pilot

Output is **sharded by seed** and written incrementally, so an interrupted
campaign resumes by re-running the same command.

Notes
-----
1. OMol/UMA requires total charge and spin multiplicity in ``Atoms.info``.
   Neutral benzene clusters are singlets: charge=0, spin=1.
2. Frames are non-periodic and centred on the origin. The monomer reference is
   ASE's idealized D6h benzene from the g2 set.
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Cap BLAS/OMP before numpy is imported: every worker is a separate UMA process
# and must not oversubscribe the machine (same reason as npt_equilibration.py).
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import sys
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from tqdm import tqdm

from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.utils.uma import DEFAULT_UMA_MODEL, load_uma_calculator
from asmcmc.utils.geometry import coarse_grain_frame

# Used only to set relative amplitudes of random internal displacement; the
# generator projects out rigid translation and rotation afterward. The g2 atom
# order is fixed: C0..C5 around the ring, H6..H11 bonded to C0..C5.
BOND_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),  # aromatic ring, 1.395 A
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (4, 10),
    (5, 11),  # C-H, 1.087 A
)

RADIAL_SAMPLINGS = ("volume-uniform", "mixture")
CONFIG_NAME = "dataset_config.json"


@dataclass(frozen=True)
class SamplingSettings:
    """Geometry knobs for cluster construction.

    ``max_com_distance`` matches ``MetropolisCalculator``'s ``nl_radius`` and
    the AniSOAP descriptor cutoff -- see the module docstring on why the 9 A
    that spans the dimer wells is not enough.

    ``trimer_max_com_distance`` bounds the *third* molecule's placement
    separately. Under volume-uniform sampling ~79% of placements land beyond
    9 A, where the three-body term is numerically zero; tightening this one
    number concentrates the trimer budget on geometries that carry three-body
    physics without touching the dimer sampling. **Measured caveat:** it is not
    what limits three-body sampling -- the hard core is (see
    ``orientation_sampling``) -- and it also sets where trimer-derived *pair*
    labels land, which is most of them. Tighten it only deliberately.

    ``orientation_sampling`` chooses how orientations are proposed.
    ``"uniform"`` is the original behaviour and remains the default, so an
    existing campaign stays reproducible bit-for-bit. ``"motif"`` proposes
    position and orientation **jointly** from
    :func:`asmcmc.data_preparation.proposals.default_mixture` -- at 3.6 A a
    uniform proposal clears the hard core ~3% of the time against ~86% for a
    cofacial-seeded one, so under ``"uniform"`` most of the short-range budget
    is spent on rejections. Motif runs additionally stamp ``log_q`` on every
    frame.
    """

    min_com_distance: float = 3.4
    max_com_distance: float = 15.0
    trimer_max_com_distance: float = 15.0
    min_atom_distance: float = 2.4
    radial_sampling: str = "volume-uniform"
    orientation_sampling: str = "uniform"
    uniform_weight: float = 0.40
    rigid: bool = True
    vibration_rms: float = 0.05
    vibration_max_atom: float = 0.10
    compact_probability: float = 0.70
    max_placement_attempts: int = 500


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation in SO(3), generated from a unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.average(positions, axis=0, weights=masses)


def inertia_tensor(
    positions: np.ndarray, masses: np.ndarray, com: np.ndarray | None = None
) -> np.ndarray:
    """Return the Cartesian inertia tensor in amu Angstrom^2."""
    if com is None:
        com = center_of_mass(positions, masses)
    r = positions - com
    rr = np.einsum("ni,nj->nij", r, r)
    r2 = np.einsum("ni,ni->n", r, r)
    eye = np.eye(3)
    return np.sum(masses[:, None, None] * (r2[:, None, None] * eye - rr), axis=0)


def remove_rigid_components(
    displacement: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Project mass-weighted rigid translation and infinitesimal rotation out
    of a Cartesian displacement."""
    disp = displacement.copy()
    total_mass = masses.sum()

    # Remove center-of-mass translation.
    disp -= np.sum(masses[:, None] * disp, axis=0) / total_mass

    com = center_of_mass(positions, masses)
    r = positions - com

    # Find the infinitesimal rotation omega minimizing
    # sum_i m_i |disp_i - omega x r_i|^2.
    inertia = inertia_tensor(positions, masses, com)
    angular_rhs = np.sum(np.cross(r, masses[:, None] * disp), axis=0)
    omega = np.linalg.pinv(inertia, rcond=1e-12) @ angular_rhs
    disp -= np.cross(omega[None, :], r)

    # Numerical cleanup of translation.
    disp -= np.sum(masses[:, None] * disp, axis=0) / total_mass
    return disp


def vibrate_monomer(
    reference: Atoms,
    rng: np.random.Generator,
    target_rms: float,
    max_atom_displacement: float,
) -> Atoms:
    """Add a small internal distortion. Heavy atoms move less than hydrogens,
    and rigid translation/rotation are projected out."""
    mol = reference.copy()
    masses = mol.get_masses()
    positions = mol.get_positions()

    # Approximately mass-weighted random vibration. Add a correlated component
    # along bonds so bond stretches and bends are represented.
    disp = rng.normal(size=positions.shape) / np.sqrt(masses[:, None])

    for i, j in BOND_PAIRS:
        direction = positions[j] - positions[i]
        direction /= np.linalg.norm(direction)
        amplitude = rng.normal()
        disp[i] -= 0.5 * amplitude * direction / math.sqrt(masses[i])
        disp[j] += 0.5 * amplitude * direction / math.sqrt(masses[j])

    disp = remove_rigid_components(disp, positions, masses)

    rms = math.sqrt(np.mean(np.sum(disp**2, axis=1)))
    if rms > 0:
        # Draw a half-normal amplitude around the requested RMS.
        requested = min(
            abs(rng.normal(loc=target_rms, scale=0.35 * target_rms)),
            2.0 * target_rms,
        )
        disp *= requested / rms

    largest = np.max(np.linalg.norm(disp, axis=1))
    if largest > max_atom_displacement:
        disp *= max_atom_displacement / largest

    mol.set_positions(positions + disp)

    # Preserve the original COM exactly before cluster placement.
    old_com = center_of_mass(positions, masses)
    new_com = center_of_mass(mol.get_positions(), masses)
    mol.translate(old_com - new_com)
    return mol


def build_reference_benzene() -> Atoms:
    """Idealized D6h benzene from ASE's g2 set (C-C 1.395 A, C-H 1.087 A)."""
    mol = molecule("C6H6")
    mol.set_pbc(False)
    mol.translate(-mol.get_center_of_mass())
    mol.info.update({"charge": 0, "spin": 1})
    return mol


def sample_radius(
    rng: np.random.Generator,
    settings: SamplingSettings,
    max_distance: float | None = None,
) -> float:
    """Draw one centre-of-mass separation.

    ``volume-uniform`` samples uniformly in r^3 over the whole range, so the
    configuration density per unit volume is flat and the shell occupancy
    matches what a bulk pair census looks like. ``mixture`` instead spends
    ``compact_probability`` of its mass on the dimer wells (3.4-6 A) -- denser
    where the cohesive energy lives, at the cost of a shape that has to be
    chosen rather than derived.
    """
    hi = settings.max_com_distance if max_distance is None else max_distance
    lo = settings.min_com_distance
    if hi <= lo:
        raise ValueError(f"max_com_distance {hi} must exceed min {lo}")

    if (
        settings.radial_sampling == "mixture"
        and rng.random() < settings.compact_probability
    ):
        # Truncated normal spanning benzene's dimer wells: the sandwich and
        # parallel-displaced minima near 3.9 A and the T-shaped one near 5.0 A.
        for _ in range(100):
            r = rng.normal(loc=4.70, scale=0.60)
            if lo <= r <= min(6.0, hi):
                return float(r)
    return float((lo**3 + rng.random() * (hi**3 - lo**3)) ** (1.0 / 3.0))


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def minimum_inter_molecular_distance(
    positions_a: np.ndarray, positions_b: np.ndarray
) -> float:
    delta = positions_a[:, None, :] - positions_b[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=-1))))


ORIENTATION_SAMPLINGS = ("uniform", "motif")


def build_proposal(settings: SamplingSettings):
    """The motif mixture implied by ``settings``.

    Lives here rather than in :mod:`asmcmc.data_preparation.proposals` so that
    module never imports this one -- ``make_cluster`` depends on the proposals,
    not the other way round.
    """
    from asmcmc.data_preparation.proposals import RadialProposal, default_mixture

    return default_mixture(
        radial=RadialProposal(
            lo=settings.min_com_distance,
            hi=settings.max_com_distance,
            mode=settings.radial_sampling,
            compact_probability=settings.compact_probability,
        ),
        uniform_weight=settings.uniform_weight,
    )


def _place(reference: Atoms, rotation: np.ndarray, target_com: np.ndarray) -> Atoms:
    """A copy of ``reference`` rotated by ``rotation`` and centred on ``target_com``."""
    mol = reference.copy()
    masses = mol.get_masses()
    pos = mol.get_positions() @ rotation.T
    pos -= center_of_mass(pos, masses)
    mol.set_positions(pos + target_com)
    return mol


def _motif_placement(n_molecules, reference, rng, settings, proposal):
    """One motif-seeded attempt: molecules, their centres, and ``log q``.

    Only the (0, 1) pair is motif-seeded; a trimer's third molecule is placed by
    the original scheme. Its density is the **anchor mixture** -- ``make_cluster``
    picks anchor 0 or 1 with probability 1/2 and which it used is not
    recoverable from the stored frame, so both terms are summed.
    """
    from asmcmc.data_preparation.proposals import rotation_to_normal

    u0, u1, r_vec, component = proposal.draw(rng, 1)
    log_q = float(proposal.log_density(u0, u1, r_vec)[0])

    normals = [u0[0], u1[0]]
    coms = [np.zeros(3), r_vec[0]]

    if n_molecules == 3:
        radial = proposal.components[-1].radial
        anchor = 0 if rng.random() < 0.5 else 1
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        distance = float(radial.sample(rng, 1)[0])
        third = coms[anchor] + distance * direction

        separations = np.array([np.linalg.norm(third - c) for c in coms])
        # 1/2 from the anchor choice, 1/(4 pi d^2) from a uniform direction.
        density = float(
            np.sum(0.5 * radial.pdf(separations) / (4.0 * np.pi * separations**2))
        )
        if density <= 0.0:
            return None
        log_q += float(np.log(density)) - float(np.log(4.0 * np.pi))  # u2 uniform

        normals.append(_uniform_normal(rng))
        coms.append(third)

    spins = rng.uniform(0.0, 2.0 * np.pi, size=n_molecules)
    rotations = rotation_to_normal(np.asarray(normals), spins)
    molecules = [
        _place(reference, rotations[k], coms[k]) for k in range(n_molecules)
    ]
    return molecules, log_q, int(component[0])


def _uniform_normal(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _clash_free(molecules, min_atom_distance: float) -> bool:
    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            if (
                minimum_inter_molecular_distance(
                    molecules[i].get_positions(), molecules[j].get_positions()
                )
                < min_atom_distance
            ):
                return False
    return True


def _assemble(molecules, extra_info: dict) -> Atoms:
    cluster = molecules[0].copy()
    molecule_id = np.zeros(len(molecules[0]), dtype=np.int32)
    for mol_index, mol in enumerate(molecules[1:], start=1):
        cluster += mol
        molecule_id = np.concatenate(
            [molecule_id, np.full(len(mol), mol_index, dtype=np.int32)]
        )

    cluster.translate(-cluster.get_center_of_mass())
    cluster.set_pbc(False)
    cluster.set_cell(np.zeros((3, 3)))
    cluster.arrays["molecule_id"] = molecule_id
    cluster.info.update(
        {"charge": 0, "spin": 1, "n_molecules": len(molecules), **extra_info}
    )
    return cluster


def make_cluster(
    n_molecules: int,
    reference: Atoms,
    rng: np.random.Generator,
    settings: SamplingSettings,
    proposal=None,
) -> Atoms:
    """Generate one non-periodic dimer or trimer with rejection of hard clashes.

    Under ``settings.orientation_sampling == "motif"`` the mixture component is
    redrawn on **every** attempt, including after a clash rejection. That is what
    makes the recorded ``log_q`` usable: with a single top-level rejection the
    realised density is ``q(x)*1[no clash]/Z`` for one global ``Z``, which
    cancels from every relative weight. Retrying inside a component would leave
    per-component normalisations that do not.
    """
    if n_molecules not in (2, 3):
        raise ValueError("Only dimers and trimers are supported.")

    if settings.orientation_sampling == "motif":
        if not settings.rigid:
            # Vibration perturbs the realised normal away from the proposed one,
            # so the stored log_q would describe a geometry that is not quite the
            # one written out. Refuse rather than record a subtly wrong density.
            raise ValueError(
                "orientation_sampling='motif' requires rigid=True: a vibrated "
                "monomer's normal differs from the proposed one, which would "
                "make the recorded log_q wrong."
            )
        proposal = proposal or build_proposal(settings)
        for _ in range(settings.max_placement_attempts):
            placed = _motif_placement(n_molecules, reference, rng, settings, proposal)
            if placed is None:
                continue
            molecules, log_q, component = placed
            if _clash_free(molecules, settings.min_atom_distance):
                return _assemble(
                    molecules,
                    {"log_q": log_q, "proposal_component": proposal.names[component]},
                )
        raise RuntimeError(
            "Failed to place a clash-free cluster under motif sampling. Consider "
            "reducing --min-atom-distance or widening the motif windows."
        )

    for _ in range(settings.max_placement_attempts):
        monomers: list[Atoms] = []
        target_coms = [np.zeros(3)]

        if n_molecules == 2:
            target_coms.append(sample_radius(rng, settings) * random_unit_vector(rng))
        else:
            # Place molecule 2 relative to molecule 1.
            c2 = sample_radius(
                rng, settings, settings.trimer_max_com_distance
            ) * random_unit_vector(rng)

            # Place molecule 3 relative to either molecule 1 or 2. This samples
            # triangular, chain-like, and partially dissociated trimers.
            anchor = np.zeros(3) if rng.random() < 0.5 else c2
            c3 = anchor + sample_radius(
                rng, settings, settings.trimer_max_com_distance
            ) * random_unit_vector(rng)
            target_coms.extend([c2, c3])

        valid = True
        for target_com in target_coms:
            if settings.rigid:
                mol = reference.copy()
            else:
                mol = vibrate_monomer(
                    reference,
                    rng,
                    target_rms=settings.vibration_rms,
                    max_atom_displacement=settings.vibration_max_atom,
                )
            rotation = random_rotation_matrix(rng)
            pos = mol.get_positions() @ rotation.T

            masses = mol.get_masses()
            pos -= center_of_mass(pos, masses)
            pos += target_com
            mol.set_positions(pos)

            for previous in monomers:
                if (
                    minimum_inter_molecular_distance(
                        mol.get_positions(), previous.get_positions()
                    )
                    < settings.min_atom_distance
                ):
                    valid = False
                    break
            if not valid:
                break
            monomers.append(mol)

        if valid:
            cluster = monomers[0].copy()
            molecule_id = np.zeros(len(monomers[0]), dtype=np.int32)
            for mol_index, mol in enumerate(monomers[1:], start=1):
                cluster += mol
                molecule_id = np.concatenate(
                    [molecule_id, np.full(len(mol), mol_index, dtype=np.int32)]
                )

            # Recenter the entire cluster for numerical convenience.
            cluster.translate(-cluster.get_center_of_mass())
            cluster.set_pbc(False)
            cluster.set_cell(np.zeros((3, 3)))
            cluster.arrays["molecule_id"] = molecule_id
            cluster.info.update({"charge": 0, "spin": 1, "n_molecules": n_molecules})
            return cluster

    raise RuntimeError(
        "Failed to place a clash-free cluster. Consider reducing "
        "--min-atom-distance or --min-com-distance."
    )


def molecule_indices(atoms: Atoms) -> list[np.ndarray]:
    ids = np.asarray(atoms.arrays["molecule_id"], dtype=int)
    return [np.flatnonzero(ids == i) for i in range(int(ids.max()) + 1)]


def molecular_geometry_metadata(atoms: Atoms) -> dict[str, np.ndarray]:
    positions = atoms.get_positions()
    masses = atoms.get_masses()

    coms, inertias, moments, axes = [], [], [], []
    for idx in molecule_indices(atoms):
        com = center_of_mass(positions[idx], masses[idx])
        tensor = inertia_tensor(positions[idx], masses[idx], com)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        coms.append(com)
        inertias.append(tensor)
        moments.append(eigenvalues)
        # Columns are principal axes in the laboratory Cartesian frame.
        axes.append(eigenvectors)

    return {
        "molecular_com": np.asarray(coms),
        "inertia_tensor": np.asarray(inertias),
        "principal_moments": np.asarray(moments),
        "principal_axes": np.asarray(axes),
    }


def molecular_force_and_torque(
    atoms: Atoms, forces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    net_forces, torques = [], []

    for idx in molecule_indices(atoms):
        com = center_of_mass(positions[idx], masses[idx])
        rel = positions[idx] - com
        f = forces[idx]
        net_forces.append(np.sum(f, axis=0))
        torques.append(np.sum(np.cross(rel, f), axis=0))

    return np.asarray(net_forces), np.asarray(torques)


def gbq_baseline(atoms: Atoms, potential=CACELLI_POTENTIAL) -> dict:
    """The Delta-learning baseline: this cluster's energy under GB+Q.

    Stored per frame so a fit can form ``E_UMA - E_GBQ`` without re-deriving
    cluster geometry. Uses :func:`asmcmc.utils.geometry.coarse_grain_frame`, the same
    atomistic-to-ellipsoid map the MC and the GB+Q fit already agree on.

    Returned in eV, matching the MLIP energies, and summed over the cluster's
    distinct molecule pairs (all of them -- clusters are small and
    non-periodic, so there is no cutoff or minimum-image question).
    """
    cg = coarse_grain_frame(atoms)
    com = cg.get_positions()
    normals = np.asarray(cg.arrays["or_vec"])
    n = len(com)

    i, j = np.triu_indices(n, k=1)
    if len(i) == 0:
        return {"or_vec": normals, "gbq_interaction_energy": 0.0}

    energy = potential.pair_energy(normals[i], normals[j], com[j] - com[i])
    return {
        "or_vec": normals,
        "gbq_interaction_energy": float(np.sum(energy)),
        "gbq_potential": potential.name,
    }


def subset_atoms(atoms: Atoms, molecule_numbers: Sequence[int]) -> Atoms:
    ids = np.asarray(atoms.arrays["molecule_id"], dtype=int)
    mask = np.isin(ids, np.asarray(molecule_numbers, dtype=int))
    sub = atoms[mask]
    sub.set_pbc(False)
    sub.set_cell(np.zeros((3, 3)))
    sub.info.update({"charge": 0, "spin": 1})
    return sub


def evaluate_energy_forces(atoms: Atoms, calculator) -> tuple[float, np.ndarray]:
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces


def energy_decomposition(
    atoms: Atoms,
    calculator,
    cluster_energy: float,
    mode: str,
    rigid_monomer_energy: float | None = None,
) -> dict:
    """Isolated-monomer references and optional trimer pair/three-body terms.

    For a trimer::

        pair_interaction_ij = E_ij - E_i - E_j
        E_3body = E_123 - sum(E_ij) + sum(E_i)

    ``rigid_monomer_energy`` short-circuits the per-molecule evaluations: with
    rigid monomers every molecule is a rotated copy of the same geometry, so
    its energy is one constant (rotation-invariant) rather than n calls.
    """
    n_mol = int(atoms.info["n_molecules"])
    if mode == "none":
        return {}

    if rigid_monomer_energy is not None:
        monomer_energies = np.full(n_mol, float(rigid_monomer_energy))
    else:
        monomer_energies = np.array(
            [
                evaluate_energy_forces(subset_atoms(atoms, [i]), calculator)[0]
                for i in range(n_mol)
            ]
        )

    result: dict = {
        "monomer_energies": monomer_energies,
        "interaction_energy": float(cluster_energy - monomer_energies.sum()),
    }

    if n_mol == 3 and mode == "full":
        pairs = ((0, 1), (0, 2), (1, 2))
        pair_energies = np.array(
            [
                evaluate_energy_forces(subset_atoms(atoms, pair), calculator)[0]
                for pair in pairs
            ]
        )
        pair_interactions = np.array(
            [
                pair_energies[k] - monomer_energies[i] - monomer_energies[j]
                for k, (i, j) in enumerate(pairs)
            ]
        )
        three_body = cluster_energy - pair_energies.sum() + monomer_energies.sum()
        result.update(
            {
                "pair_molecule_ids": np.asarray(pairs, dtype=np.int32),
                "pair_energies": pair_energies,
                "pair_interaction_energies": pair_interactions,
                "three_body_energy": float(three_body),
            }
        )

    return result


def attach_stored_results(
    atoms: Atoms, energy: float, forces: np.ndarray, extra_info: dict
) -> Atoms:
    """Detach the live MLIP and attach portable ASE single-point results."""
    stored = atoms.copy()
    stored.info.update(extra_info)
    stored.calc = SinglePointCalculator(stored, energy=energy, forces=forces)
    return stored


def verify_saved_frame(atoms: Atoms) -> None:
    """Cheap consistency checks before writing."""
    n_mol = int(atoms.info["n_molecules"])
    required_shapes = {
        "molecular_com": (n_mol, 3),
        "inertia_tensor": (n_mol, 3, 3),
        "principal_moments": (n_mol, 3),
        "principal_axes": (n_mol, 3, 3),
        "molecular_force": (n_mol, 3),
        "molecular_torque": (n_mol, 3),
        "or_vec": (n_mol, 3),
    }
    for key, shape in required_shapes.items():
        arr = np.asarray(atoms.info[key])
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} has invalid shape/data: {arr.shape}")

    forces = atoms.get_forces()
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("Invalid atomic forces.")
    if not np.isfinite(atoms.get_potential_energy()):
        raise ValueError("Invalid potential energy.")


# --- sharded, resumable generation -------------------------------------------


def shard_path(out_dir: Path, shard: int) -> Path:
    return Path(out_dir) / f"clusters_shard{shard:02d}.xyz"


def shard_count(path: Path) -> int:
    """Frames already complete in a shard, tolerating a truncated final one.

    A run killed mid-write can leave a partial frame, which makes
    ``ase.io.read`` raise on the whole file. Counting extxyz frame headers
    (natoms line + comment line + natoms body lines) instead means a resume
    drops only the torn frame rather than the entire shard.
    """
    path = Path(path)
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    n_frames, cursor = 0, 0
    while cursor < len(lines):
        try:
            n_atoms = int(lines[cursor].strip())
        except (ValueError, IndexError):
            break
        if cursor + 1 + n_atoms >= len(lines):
            break  # header present but body truncated
        cursor += 2 + n_atoms
        n_frames += 1
    return n_frames


def config_rng(seed: int, index: int, attempt: int = 0) -> np.random.Generator:
    """Independent generator for one configuration.

    Seeding per *configuration* rather than per shard is what makes resume
    exact: config ``index`` is byte-identical whether it was produced in the
    first pass or after an interruption. Advancing a single shard-wide stream
    could not do that -- each configuration draws a variable number of values
    (rejection sampling in ``make_cluster``), so there is no fixed amount to
    skip.

    ``attempt`` salts the seed. Without it a configuration that fails to place
    would be retried from an identical stream and fail identically forever.
    """
    return np.random.default_rng([int(seed), int(index), int(attempt)])


def generate_shard(
    out_dir,
    shard: int,
    n_configs: int,
    seed: int,
    settings_dict: dict,
    model: str,
    device: str,
    decomposition: str,
    trimer_fraction: float,
    flush_every: int,
    progress: bool = False,
    progress_queue=None,
) -> dict:
    """Generate one shard, appending incrementally so a crash loses ~nothing.

    Takes ``settings_dict`` rather than a ``SamplingSettings`` so the payload
    pickles cleanly into a spawned worker.

    Progress is reported one of two ways because the two call paths differ.
    ``progress_queue`` (a manager queue) is for the pooled path: a spawned
    worker cannot draw to the parent's terminal without shards fighting over
    the same lines, so it posts counts and the parent owns the single bar.
    ``progress`` alone drives a local bar for the in-process single-shard path.
    """
    out_dir = Path(out_dir)
    settings = SamplingSettings(**settings_dict)
    path = shard_path(out_dir, shard)

    done = shard_count(path)
    # Frames already on disk still count toward the campaign, so a resumed run's
    # bar starts where the data does rather than at zero.
    if progress_queue is not None and done:
        progress_queue.put(min(done, n_configs))
    if done >= n_configs:
        return {"shard": shard, "written": 0, "total": done, "skipped": True}

    reference = build_reference_benzene()
    calculator = load_uma_calculator(model, device=device)

    # Built once per shard: the mixture parses the Cacelli reference file and
    # runs a quadrature per concentration, neither of which belongs per-cluster.
    proposal = (
        build_proposal(settings) if settings.orientation_sampling == "motif" else None
    )

    rigid_monomer_energy = None
    if settings.rigid and decomposition != "none":
        # One evaluation for the whole shard: every molecule is a rotated copy
        # of this geometry and the energy is rotation-invariant.
        rigid_monomer_energy = evaluate_energy_forces(reference.copy(), calculator)[0]

    buffer: list[Atoms] = []
    written = 0
    failures = 0

    def flush():
        nonlocal buffer
        if buffer:
            write(path, buffer, append=True)
            buffer = []

    # disable=None silences the bar whenever stderr is not a terminal, so pytest
    # and piped runs stay clean without a flag.
    local_bar = (
        tqdm(
            total=n_configs,
            initial=done,
            unit="cfg",
            desc=f"shard {shard:02d}",
            disable=None,
        )
        if progress and progress_queue is None
        else None
    )

    attempt = 0
    while done + written < n_configs:
        index = done + written
        rng = config_rng(seed, index, attempt)
        n_molecules = 3 if rng.random() < trimer_fraction else 2
        try:
            cluster = make_cluster(n_molecules, reference, rng, settings, proposal)
            energy, forces = evaluate_energy_forces(cluster, calculator)
            net_force, torque = molecular_force_and_torque(cluster, forces)
            info = {
                **molecular_geometry_metadata(cluster),
                **gbq_baseline(cluster),
                **energy_decomposition(
                    cluster,
                    calculator,
                    cluster_energy=energy,
                    mode=decomposition,
                    rigid_monomer_energy=rigid_monomer_energy,
                ),
                "molecular_force": net_force,
                "molecular_torque": torque,
                "config_index": index,
                "shard": shard,
                "generator_seed": seed,
                "mlip_model": model,
                "mlip_task": "omol",
                "rigid_monomers": bool(settings.rigid),
                "radial_sampling": settings.radial_sampling,
                "orientation_sampling": settings.orientation_sampling,
                "energy_units": "eV",
                "force_units": "eV/Angstrom",
                "torque_units": "eV",
                "length_units": "Angstrom",
                "inertia_units": "amu*Angstrom^2",
            }
            stored = attach_stored_results(cluster, energy, forces, info)
            verify_saved_frame(stored)
            buffer.append(stored)
            written += 1
            attempt = 0

            # Per configuration, not per flush: a 5000-config campaign flushes
            # every 10, and a bar that only moved every tenth would read as
            # stalled for minutes at a time.
            if progress_queue is not None:
                progress_queue.put(1)
            elif local_bar is not None:
                local_bar.update(1)

            if len(buffer) >= flush_every:
                flush()
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures += 1
            attempt += 1  # re-salt the seed; the same stream would fail identically
            print(f"shard {shard:02d} skipping attempt: {exc}", file=sys.stderr)
            if failures > max(1000, 10 * n_configs):
                flush()
                if local_bar is not None:
                    local_bar.close()
                raise RuntimeError("Too many failed sample attempts.") from exc

    flush()
    if local_bar is not None:
        local_bar.close()
    return {
        "shard": shard,
        "written": written,
        "total": done + written,
        "skipped": False,
    }


def _shard_sizes(n_configs: int, n_shards: int) -> list[int]:
    """Split ``n_configs`` as evenly as possible across shards."""
    base, extra = divmod(n_configs, n_shards)
    return [base + (1 if k < extra else 0) for k in range(n_shards)]


def main(
    n_configs: int = 500,
    out_dir="results/clusters/pilot",
    n_shards: int | None = None,
    seed0: int = 20260731,
    settings: SamplingSettings | None = None,
    model: str = DEFAULT_UMA_MODEL,
    device: str = "cpu",
    decomposition: str = "full",
    trimer_fraction: float = 0.5,
    flush_every: int = 10,
    max_workers: int = 4,
) -> list[dict]:
    """Run a sharded, resumable generation campaign.

    Idempotent: a shard already holding its target count is skipped, so
    re-running the same command finishes an interrupted campaign.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or SamplingSettings()

    if settings.radial_sampling not in RADIAL_SAMPLINGS:
        raise ValueError(
            f"radial_sampling must be one of {RADIAL_SAMPLINGS}, "
            f"got {settings.radial_sampling!r}"
        )
    if settings.orientation_sampling not in ORIENTATION_SAMPLINGS:
        raise ValueError(
            f"orientation_sampling must be one of {ORIENTATION_SAMPLINGS}, "
            f"got {settings.orientation_sampling!r}"
        )

    n_shards = n_shards or min(max_workers, os.cpu_count() or 1)
    n_shards = max(1, min(n_shards, n_configs))
    sizes = _shard_sizes(n_configs, n_shards)

    config = {
        "n_configs": n_configs,
        "n_shards": n_shards,
        "shard_sizes": sizes,
        "seed0": seed0,
        "model": model,
        "decomposition": decomposition,
        "trimer_fraction": trimer_fraction,
        "settings": asdict(settings),
    }
    config_path = out_dir / CONFIG_NAME
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2))

    jobs = [
        dict(
            out_dir=str(out_dir),
            shard=k,
            n_configs=sizes[k],
            seed=seed0 + k,
            settings_dict=asdict(settings),
            model=model,
            device=device,
            decomposition=decomposition,
            trimer_fraction=trimer_fraction,
            flush_every=flush_every,
        )
        for k in range(n_shards)
    ]

    if n_shards == 1:
        return [generate_shard(**jobs[0], progress=True)]

    cpu_count_lim = 1 if os.cpu_count() is None else os.cpu_count() / 2
    num_workers = min(max_workers, cpu_count_lim, n_shards)
    results = []
    context = get_context("spawn")

    # A manager queue rather than a plain mp.Queue: ProcessPoolExecutor pickles
    # its arguments, and a raw queue refuses that ("should only be shared
    # through inheritance"). The proxy pickles fine.
    with context.Manager() as manager:
        queue = manager.Queue()
        # spawn (not the Linux-default fork): forking a process that has already
        # started BLAS/torch threads can deadlock the child.
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=context
        ) as pool:
            futures = {
                pool.submit(generate_shard, **job, progress_queue=queue): job["shard"]
                for job in jobs
            }
            pending = set(futures)
            with tqdm(
                total=n_configs, unit="cfg", desc="clusters", disable=None
            ) as bar:
                while pending:
                    # Short timeout so the bar keeps moving while shards run;
                    # draining between waits is what keeps the queue bounded.
                    finished, pending = wait(pending, timeout=0.5)
                    bar.update(_drain(queue))
                    for future in finished:
                        results.append(future.result())
                bar.update(_drain(queue))

    for res in sorted(results, key=lambda r: r["shard"]):
        state = "skipped (complete)" if res["skipped"] else f"+{res['written']}"
        print(f"shard {res['shard']:02d}: {state}, {res['total']} frames", flush=True)

    return sorted(results, key=lambda r: r["shard"])


def _drain(progress_queue) -> int:
    """Total of everything currently queued, without blocking.

    Catches only ``Empty``: a broader except would swallow a dead manager
    connection and leave the bar silently frozen while the run continued.
    """
    total = 0
    while True:
        try:
            total += progress_queue.get_nowait()
        except Empty:
            return total


def dataset_frames(out_dir) -> list[Atoms]:
    """Every frame across a campaign's shards, in shard order."""
    frames = []
    for path in sorted(Path(out_dir).glob("clusters_shard*.xyz")):
        frames.extend(read(path, index=":"))
    return frames


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m asmcmc.data_preparation.cluster_dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-configs", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("results/clusters/pilot"))
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=20260731)
    parser.add_argument("--model", default=DEFAULT_UMA_MODEL)
    parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--trimer-fraction",
        type=float,
        default=0.5,
        help="Probability that a generated configuration is a trimer.",
    )
    parser.add_argument(
        "--decomposition",
        choices=("none", "monomers", "full"),
        default="full",
        help=(
            "none: cluster E/F only; monomers: monomer and total interaction "
            "energies; full: also trimer pair and three-body energies."
        ),
    )
    parser.add_argument(
        "--radial-sampling",
        choices=RADIAL_SAMPLINGS,
        default="volume-uniform",
        help=(
            "volume-uniform: flat in r^3 over the whole range. mixture: "
            "concentrate compact-probability of the mass on the 3.4-6 A wells."
        ),
    )
    parser.add_argument(
        "--orientation-sampling",
        choices=ORIENTATION_SAMPLINGS,
        default="uniform",
        help=(
            "uniform: random SO(3) per molecule (the original behaviour). "
            "motif: propose position and orientation jointly from cofacial / "
            "slipped / T-shaped components plus a defensive uniform one, and "
            "stamp log_q on every frame. Requires --rigid (the default)."
        ),
    )
    parser.add_argument(
        "--uniform-weight",
        type=float,
        default=0.40,
        help="Mixture weight of the defensive uniform component under "
        "--orientation-sampling motif. Lowering it sharpens the proposal but "
        "widens the importance weights; check the realised spread before doing so.",
    )
    parser.add_argument("--min-com-distance", type=float, default=3.4)
    parser.add_argument(
        "--max-com-distance",
        type=float,
        default=15.0,
        help="Matches MetropolisCalculator's nl_radius and the AniSOAP cutoff.",
    )
    parser.add_argument(
        "--trimer-max-com-distance",
        type=float,
        default=15.0,
        help="Separate ceiling for trimer placement; tighten to concentrate "
        "the trimer budget where three-body terms are non-zero.",
    )
    parser.add_argument("--min-atom-distance", type=float, default=2.4)
    parser.add_argument(
        "--no-rigid",
        dest="rigid",
        action="store_false",
        help="Vibrate each monomer instead of using the rigid reference. Costs "
        "~2x the MLIP calls; a rigid-ellipsoid model cannot fit the difference.",
    )
    parser.add_argument("--vibration-rms", type=float, default=0.05)
    parser.add_argument("--vibration-max-atom", type=float, default=0.10)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Append to the shard file after this many configurations.",
    )
    return parser.parse_args(argv)


def cli(argv=None) -> None:
    args = parse_args(argv)

    if args.n_configs <= 0:
        raise SystemExit("--n-configs must be positive.")
    if not 0.0 <= args.trimer_fraction <= 1.0:
        raise SystemExit("--trimer-fraction must be between 0 and 1.")
    if args.min_com_distance >= args.max_com_distance:
        raise SystemExit("--min-com-distance must be less than --max-com-distance.")

    settings = SamplingSettings(
        min_com_distance=args.min_com_distance,
        max_com_distance=args.max_com_distance,
        trimer_max_com_distance=args.trimer_max_com_distance,
        min_atom_distance=args.min_atom_distance,
        radial_sampling=args.radial_sampling,
        orientation_sampling=args.orientation_sampling,
        uniform_weight=args.uniform_weight,
        rigid=args.rigid,
        vibration_rms=args.vibration_rms,
        vibration_max_atom=args.vibration_max_atom,
    )

    results = main(
        n_configs=args.n_configs,
        out_dir=args.out_dir,
        n_shards=args.n_shards,
        seed0=args.seed0,
        settings=settings,
        model=args.model,
        device=args.device,
        decomposition=args.decomposition,
        trimer_fraction=args.trimer_fraction,
        flush_every=args.flush_every,
        max_workers=args.max_workers,
    )
    total = sum(r["total"] for r in results)
    print(f"\n{total} frames across {len(results)} shards in {args.out_dir}")


if __name__ == "__main__":
    cli()
