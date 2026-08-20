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

# proposals imports nothing from this package at module scope -- it reaches the
# classifier's thresholds through a deferred import -- so this direction is safe
# and the old function-body import is no longer needed.
from asmcmc.data_preparation.proposals import (
    MOTIFS,
    RadialProposal,
    UNIFORM,
    draw_motif,
    rotation_to_normal,
)

RADIAL_SAMPLINGS = ("volume-uniform", "mixture")
CONFIG_NAME = "dataset_config.json"


@dataclass(frozen=True)
class SamplingSettings:
    """Geometry knobs for cluster construction.

    ``max_com_distance`` matches ``MetropolisCalculator``'s ``nl_radius`` and
    the AniSOAP descriptor cutoff -- see the module docstring on why the 9 A
    that spans the dimer wells is not enough. It bounds the ``uniform`` motif;
    the four named motifs live inside ``cluster_analysis.WELL_RANGE`` by
    definition and are unaffected by it.

    ``trimer_max_com_distance`` bounds the *third* molecule's placement
    separately. Under volume-uniform sampling ~79% of placements land beyond
    9 A, where the three-body term is numerically zero; tightening this one
    number concentrates the trimer budget on geometries that carry three-body
    physics without touching the dimer sampling. **Measured caveat:** it also
    sets where trimer-derived *pair* labels land, which is most of them.
    Tighten it only deliberately.

    ``trimer_fraction`` lives here rather than on the campaign because it
    decides geometry: it is the probability that a configuration is a trimer
    rather than a dimer.

    Monomers are always rigid -- see the module docstring for why, and note that
    the flexible path was removed rather than left dormant.
    """

    min_com_distance: float = 3.4
    max_com_distance: float = 15.0
    trimer_max_com_distance: float = 15.0
    min_atom_distance: float = 2.4
    radial_sampling: str = "volume-uniform"
    compact_probability: float = 0.70
    trimer_fraction: float = 0.70
    max_placement_attempts: int = 500

    def __post_init__(self):
        """Validate here, not at four scattered call sites.

        A frozen dataclass is the one place every construction path goes
        through, so a setting cannot reach the generator unchecked.
        """
        if self.radial_sampling not in RADIAL_SAMPLINGS:
            raise ValueError(
                f"radial_sampling must be one of {RADIAL_SAMPLINGS}, "
                f"got {self.radial_sampling!r}"
            )
        if not self.min_com_distance < self.max_com_distance:
            raise ValueError("min_com_distance must be below max_com_distance")
        if not 0.0 <= self.trimer_fraction <= 1.0:
            raise ValueError("trimer_fraction must be in [0, 1]")
        if not 0.0 <= self.compact_probability <= 1.0:
            raise ValueError("compact_probability must be in [0, 1]")
        if self.max_placement_attempts < 1:
            raise ValueError("max_placement_attempts must be positive")

    def radial(self):
        """The :class:`~asmcmc.data_preparation.proposals.RadialProposal` these imply."""
        return RadialProposal(
            lo=self.min_com_distance,
            hi=self.max_com_distance,
            mode=self.radial_sampling,
            compact_probability=self.compact_probability,
        )

    def trimer_radial(self):
        """As :meth:`radial`, capped at ``trimer_max_com_distance``.

        The third molecule gets its own ceiling. The previous motif path ignored
        this setting entirely -- it reached into the mixture's uniform component,
        whose ceiling came from ``max_com_distance`` -- so a campaign that asked
        for a tighter trimer never got one.
        """
        return RadialProposal(
            lo=self.min_com_distance,
            hi=self.trimer_max_com_distance,
            mode=self.radial_sampling,
            compact_probability=self.compact_probability,
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


def build_reference_benzene() -> Atoms:
    """Idealized D6h benzene from ASE's g2 set (C-C 1.395 A, C-H 1.087 A)."""
    mol = molecule("C6H6")
    mol.set_pbc(False)
    mol.translate(-mol.get_center_of_mass())
    mol.info.update({"charge": 0, "spin": 1})
    return mol


def minimum_inter_molecular_distance(
    positions_a: np.ndarray, positions_b: np.ndarray
) -> float:
    delta = positions_a[:, None, :] - positions_b[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=-1))))


def _place(reference: Atoms, rotation: np.ndarray, target_com: np.ndarray) -> Atoms:
    """A copy of ``reference`` rotated by ``rotation`` and centred on ``target_com``."""
    mol = reference.copy()
    masses = mol.get_masses()
    pos = mol.get_positions() @ rotation.T
    pos -= center_of_mass(pos, masses)
    mol.set_positions(pos + target_com)
    return mol


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
    motif: str,
) -> Atoms:
    """One non-periodic dimer or trimer seeded on ``motif``, free of hard clashes.

    The motif fixes the (0, 1) pair. A trimer's third molecule is placed
    isotropically about either of the first two, so the motif is a statement
    about the seeded contact rather than about the whole cluster -- which is why
    a trimer's other two pairs are incidental and land wherever they land.

    The geometry is redrawn in full on every attempt, motif included, so a
    rejection cannot bias the realised distribution toward whichever corner of
    the motif happens to clear the hard core most easily.
    """
    if n_molecules not in (2, 3):
        raise ValueError("Only dimers and trimers are supported.")

    for _ in range(settings.max_placement_attempts):
        u0, u1, r_vec = draw_motif(motif, rng, settings.radial())
        normals = [u0, u1]
        coms = [np.zeros(3), r_vec]

        if n_molecules == 3:
            anchor = coms[0] if rng.random() < 0.5 else coms[1]
            direction = _isotropic_normal(rng)
            distance = float(settings.trimer_radial().sample(rng, 1)[0])
            coms.append(anchor + distance * direction)
            # Its own orientation is independent of where it was placed.
            normals.append(_isotropic_normal(rng))

        spins = rng.uniform(0.0, 2.0 * np.pi, size=n_molecules)
        rotations = rotation_to_normal(np.asarray(normals), spins)
        molecules = [_place(reference, rotations[k], coms[k]) for k in range(n_molecules)]

        if _clash_free(molecules, settings.min_atom_distance):
            return _assemble(molecules, {"motif": motif})

    raise RuntimeError(
        f"Failed to place a clash-free {motif!r} cluster in "
        f"{settings.max_placement_attempts} attempts. Consider reducing "
        "--min-atom-distance."
    )


def _isotropic_normal(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def motif_plan(seed0: int, n_configs: int, quotas: dict) -> np.ndarray:
    """One motif label per configuration index, honouring ``quotas`` exactly.

    Composition is decided here, once, rather than emerging from a sampling
    density -- so a campaign that asks for 30% cofacial gets 30% cofacial, and
    the number is checkable before any MLIP time is spent.

    Counts come from largest-remainder apportionment, so they sum to
    ``n_configs`` exactly with no motif rounded away. The shuffle is seeded from
    ``seed0`` alone, which makes the label at index ``i`` a pure function of the
    campaign: shard ``k`` reads its own slice, and a resumed shard reproduces
    the labels it had before.
    """
    if not quotas:
        raise ValueError("at least one motif quota is required")
    unknown = set(quotas) - set(MOTIFS)
    if unknown:
        raise ValueError(
            f"unknown motif(s) {sorted(unknown)}; expected from {list(MOTIFS)}"
        )
    shares = {m: float(w) for m, w in quotas.items() if float(w) > 0.0}
    if not shares:
        raise ValueError("motif quotas must include at least one positive share")
    if min(shares.values()) < 0.0:
        raise ValueError("motif quotas must be non-negative")

    total = sum(shares.values())
    exact = {m: n_configs * w / total for m, w in shares.items()}
    counts = {m: int(math.floor(v)) for m, v in exact.items()}
    # Hand out the remaining slots to the largest fractional parts.
    order = sorted(exact, key=lambda m: (-(exact[m] - counts[m]), m))
    for m in order[: n_configs - sum(counts.values())]:
        counts[m] += 1

    plan = np.array(
        [m for motif, k in counts.items() for m in [motif] * k], dtype=object
    )
    np.random.default_rng(seed0).shuffle(plan)
    return plan


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
    motifs: Sequence[str],
    flush_every: int,
    progress: bool = False,
    progress_queue=None,
) -> dict:
    """Generate one shard, appending incrementally so a crash loses ~nothing.

    Takes ``settings_dict`` rather than a ``SamplingSettings`` so the payload
    pickles cleanly into a spawned worker.

    ``motifs`` is this shard's slice of :func:`motif_plan` -- one label per
    configuration, in index order, so a resumed shard picks up the same labels
    it would have generated had it never stopped.

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

    rigid_monomer_energy = None
    if decomposition != "none":
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
        n_molecules = 3 if rng.random() < settings.trimer_fraction else 2
        try:
            cluster = make_cluster(n_molecules, reference, rng, settings, motifs[index])
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
                "rigid_monomers": True,
                "radial_sampling": settings.radial_sampling,
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
    quotas: dict | None = None,
    flush_every: int = 10,
    max_workers: int = 4,
) -> list[dict]:
    """Run a sharded, resumable generation campaign.

    ``quotas`` maps motif name to a share of the campaign; shares are
    normalised, so ``{"cofacial": 1, "uniform": 1}`` and
    ``{"cofacial": 0.5, "uniform": 0.5}`` mean the same thing. The default is
    the pre-motif behaviour: everything uniform.

    Idempotent: a shard already holding its target count is skipped, so
    re-running the same command finishes an interrupted campaign.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or SamplingSettings()  # validates itself
    quotas = quotas or {UNIFORM: 1.0}

    n_shards = n_shards or min(max_workers, os.cpu_count() or 1)
    n_shards = max(1, min(n_shards, n_configs))
    sizes = _shard_sizes(n_configs, n_shards)

    # Decided up front so the manifest can state the composition the campaign
    # will actually have, rather than the one it was asked for.
    plan = motif_plan(seed0, n_configs, quotas)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    shard_motifs = [
        [str(m) for m in plan[bounds[k] : bounds[k + 1]]] for k in range(n_shards)
    ]

    config = {
        "n_configs": n_configs,
        "n_shards": n_shards,
        "shard_sizes": sizes,
        "seed0": seed0,
        "model": model,
        "decomposition": decomposition,
        "quotas": {str(k): float(v) for k, v in quotas.items()},
        "motif_counts": {
            m: int(np.sum(plan == m)) for m in MOTIFS if int(np.sum(plan == m))
        },
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
            motifs=shard_motifs[k],
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
    # Geometry defaults are read off the dataclass rather than restated, so the
    # two cannot drift -- the previous parser hard-coded nine literals that were
    # already written down in SamplingSettings.
    default_settings = SamplingSettings()

    parser.add_argument("--n-configs", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("results/clusters/pilot"))
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=20260731)
    parser.add_argument("--model", default=DEFAULT_UMA_MODEL)
    parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--decomposition",
        choices=("none", "monomers", "full"),
        default="full",
        help=(
            "none: cluster E/F only; monomers: monomer and total interaction "
            "energies; full: also trimer pair and three-body energies."
        ),
    )

    quota = parser.add_argument_group(
        "motif quotas",
        "Share of the campaign seeded on each contact motif. Shares are "
        "normalised, so they need not sum to 1. With none given the campaign is "
        "entirely uniform, which is the pre-motif behaviour.",
    )
    for motif in MOTIFS:
        quota.add_argument(
            f"--{motif.lower().replace('-', '-')}",
            type=float,
            default=None,
            dest=f"quota_{motif.lower().replace('-', '_')}",
            metavar="SHARE",
            help=f"Share of configurations seeded on the {motif} contact.",
        )

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument(
        "--radial-sampling",
        choices=RADIAL_SAMPLINGS,
        default=default_settings.radial_sampling,
        help=(
            "Applies to the uniform motif. volume-uniform: flat in r^3 over the "
            "whole range. mixture: concentrate compact-probability of the mass "
            "on the 3.4-6 A wells."
        ),
    )
    geometry.add_argument(
        "--compact-probability",
        type=float,
        default=default_settings.compact_probability,
        help="Mass placed on the wells under --radial-sampling mixture.",
    )
    geometry.add_argument(
        "--trimer-fraction",
        type=float,
        default=default_settings.trimer_fraction,
        help="Probability that a generated configuration is a trimer.",
    )
    geometry.add_argument(
        "--min-com-distance", type=float, default=default_settings.min_com_distance
    )
    geometry.add_argument(
        "--max-com-distance",
        type=float,
        default=default_settings.max_com_distance,
        help="Matches MetropolisCalculator's nl_radius and the AniSOAP cutoff.",
    )
    geometry.add_argument(
        "--trimer-max-com-distance",
        type=float,
        default=default_settings.trimer_max_com_distance,
        help="Separate ceiling for trimer placement; tighten to concentrate "
        "the trimer budget where three-body terms are non-zero.",
    )
    geometry.add_argument(
        "--min-atom-distance", type=float, default=default_settings.min_atom_distance
    )
    geometry.add_argument(
        "--max-placement-attempts",
        type=int,
        default=default_settings.max_placement_attempts,
        help="Clash-rejection budget per configuration.",
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Append to the shard file after this many configurations.",
    )
    return parser.parse_args(argv)


def quotas_from_args(args) -> dict:
    """The ``{motif: share}`` map implied by the ``--<motif>`` flags."""
    quotas = {}
    for motif in MOTIFS:
        value = getattr(args, f"quota_{motif.lower().replace('-', '_')}")
        if value is not None:
            quotas[motif] = value
    return quotas


def cli(argv=None) -> None:
    args = parse_args(argv)

    if args.n_configs <= 0:
        raise SystemExit("--n-configs must be positive.")

    # SamplingSettings validates its own fields; surface that as a CLI error
    # rather than a traceback.
    try:
        settings = SamplingSettings(
            min_com_distance=args.min_com_distance,
            max_com_distance=args.max_com_distance,
            trimer_max_com_distance=args.trimer_max_com_distance,
            min_atom_distance=args.min_atom_distance,
            radial_sampling=args.radial_sampling,
            compact_probability=args.compact_probability,
            trimer_fraction=args.trimer_fraction,
            max_placement_attempts=args.max_placement_attempts,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    results = main(
        n_configs=args.n_configs,
        out_dir=args.out_dir,
        n_shards=args.n_shards,
        seed0=args.seed0,
        settings=settings,
        model=args.model,
        device=args.device,
        decomposition=args.decomposition,
        quotas=quotas_from_args(args) or None,
        flush_every=args.flush_every,
        max_workers=args.max_workers,
    )
    total = sum(r["total"] for r in results)
    print(f"\n{total} frames across {len(results)} shards in {args.out_dir}")


if __name__ == "__main__":
    cli()
