"""Generate a UMA-labelled benzene dimer dataset for AniSOAP training.

The labeller is Meta FAIR Chemistry's OMol-trained UMA MLIP, which clears the
physics gate in :mod:`asmcmc.delta_learning.dimer_benchmark` first: r = 0.994 / RMSE 0.55 kcal/mol
against all 197 Cacelli MP2 dimer rows, all three wells bound and placed within
~0.1 A, having never seen that data.

Dimers only (2026-09; trimers dropped -- see CLAUDE.md's TRIMER note). Every
pair is drawn uniformly in orientation and volume-in-r^3, filtered by geometry
alone (``SamplingSettings.min_atom_distance``/``max_atom_distance``): no motif
taxonomy, no quotas. Deployment is pair-decomposed (see
:mod:`asmcmc.delta_learning.model`'s ``AniSOAPDeltaPotential``), so a dimer
campaign matches the training distribution to the deployment distribution by
construction.

Each saved frame carries:
  * total energy in a ``SinglePointCalculator``
  * ``arrays["molecule_id"]``      : molecule membership per atom
  * ``info["molecular_com"]``      : (2, 3), Angstrom
  * ``info["principal_axes"]``     : (2, 3, 3)
  * ``info["or_vec"]``             : (2, 3) disc normals, the GB+Q orientation
  * ``info["monomer_energies"]``   : isolated monomer references, eV
  * ``info["interaction_energy"]`` : E(cluster) - sum E(monomers), eV
  * ``info["gbq_interaction_energy"]`` : the same quantity under CACELLI_POTENTIAL,
    so the Delta-learning target E_UMA - E_GBQ needs no geometry re-derivation

Usage::

    python -m asmcmc.delta_learning.dataset --n-configs 500 --out-dir results/clusters/pilot

Output is sharded by seed and written incrementally, so an interrupted
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
import os

# Cap BLAS/OMP before numpy is imported: every worker is a separate UMA process
# and must not oversubscribe the machine.
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

from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.delta_learning.uma import DEFAULT_UMA_MODEL, load_uma_calculator
from asmcmc.mc.coarse_graining import centre_of_mass, coarse_grain_frame

RADIAL_SAMPLINGS = ("volume-uniform", "mixture")
CONFIG_NAME = "dataset_config.json"


# --- dimer geometry sampling -------------------------------------------------

# The reference benzene from ``build_reference_benzene`` lies in the xy-plane.
REFERENCE_NORMAL = np.array([0.0, 0.0, 1.0])


def orthonormal_basis(u):
    """Two unit vectors completing ``u`` (shape ``(n, 3)``) to a right-handed frame."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    seed = np.where(
        (np.abs(u[:, 0]) > 0.9)[:, None],
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = seed - np.einsum("ni,ni->n", seed, u)[:, None] * u
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(u, e1)


def rotation_to_normal(normals, spin, reference_normal=REFERENCE_NORMAL):
    """Rotations carrying ``reference_normal`` onto each of ``normals``.

    ``spin`` is an additional rotation about the target normal. It leaves the
    coarse-grained system unchanged, but it does move the atoms, so it is a genuine
    degree of freedom of the atomistic configuration and is drawn uniformly.
    """
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    spin = np.atleast_1d(np.asarray(spin, dtype=float))
    n0 = np.asarray(reference_normal, dtype=float)

    axis = np.cross(np.broadcast_to(n0, normals.shape), normals)
    cos = normals @ n0
    eye = np.broadcast_to(np.eye(3), (len(normals), 3, 3))

    K = _skew(axis)
    # Rodrigues in the (I + K + K^2/(1+cos)) form. The 1+cos denominator blows
    # up for an exactly antiparallel target, handled separately below.
    with np.errstate(divide="ignore", invalid="ignore"):
        align = eye + K + K @ K / (1.0 + cos)[:, None, None]

    flipped = cos < -1.0 + 1e-12
    if np.any(flipped):
        # A 180 degree turn about any axis perpendicular to n0.
        perp = orthonormal_basis(np.broadcast_to(n0, (int(flipped.sum()), 3)))[0]
        align[flipped] = 2.0 * np.einsum("ni,nj->nij", perp, perp) - np.eye(3)

    degenerate = cos > 1.0 - 1e-12
    if np.any(degenerate):
        align[degenerate] = np.eye(3)

    K_spin = _skew(normals)
    spin_rot = (
        eye
        + np.sin(spin)[:, None, None] * K_spin
        + (1.0 - np.cos(spin))[:, None, None] * (K_spin @ K_spin)
    )
    return spin_rot @ align


def _skew(v):
    zero = np.zeros(len(v))
    return np.stack(
        [
            np.stack([zero, -v[:, 2], v[:, 1]], axis=-1),
            np.stack([v[:, 2], zero, -v[:, 0]], axis=-1),
            np.stack([-v[:, 1], v[:, 0], zero], axis=-1),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class RadialProposal:
    """Centre-centre separations: volume-uniform, or concentrated on the wells."""

    lo: float = 3.4
    hi: float = 15.0
    mode: str = "mixture"
    compact_probability: float = 0.70
    mu: float = 4.70
    sigma: float = 0.60
    cap: float = 6.0

    @property
    def _window(self):
        return self.lo, min(self.cap, self.hi)

    def sample(self, rng, size):
        out = (self.lo**3 + rng.random(size) * (self.hi**3 - self.lo**3)) ** (1.0 / 3.0)
        if self.mode != "mixture":
            return out
        a, b = self._window
        use = rng.random(size) < self.compact_probability
        for i in np.flatnonzero(use):
            for _ in range(100):
                r = rng.normal(self.mu, self.sigma)
                if a <= r <= b:
                    out[i] = r
                    break
        return out


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def draw_uniform(rng, radial=None):
    """Both orientations and the separation drawn uniformly."""
    radial = radial or RadialProposal()
    return _unit(rng), _unit(rng), float(radial.sample(rng, 1)[0]) * _unit(rng)


@dataclass(frozen=True)
class SamplingSettings:
    """Geometry knobs for cluster construction.

    ``max_com_distance`` bounds the centre-centre draw. ``min_atom_distance``/
    ``max_atom_distance`` bound the realised minimum atom-atom separation
    between the two molecules, which is what actually governs both ends of the
    UMA label: below ~3.0 A the pair is a hard-core clash UMA scores strongly
    repulsive (measured: pairs at 2.40-2.73 A dominate the Delta^2 fit
    objective by 40-75% depending on campaign, while the MC sampler never
    approaches closer than ~3.6 A), and past UMA's ~6 A minimum-atom-atom
    interaction horizon the label is a truncation artifact (Delta = -E_GBQ
    exactly). Tightening this window is what disarms both failure modes at
    once.
    """

    min_com_distance: float = 3.4
    max_com_distance: float = 11.0
    min_atom_distance: float = 3.0
    max_atom_distance: float = 5.5
    radial_sampling: str = "volume-uniform"
    compact_probability: float = 0.70
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
        if not self.min_atom_distance < self.max_atom_distance:
            raise ValueError("min_atom_distance must be below max_atom_distance")
        if not 0.0 <= self.compact_probability <= 1.0:
            raise ValueError("compact_probability must be in [0, 1]")
        if self.max_placement_attempts < 1:
            raise ValueError("max_placement_attempts must be positive")

    def radial(self):
        """The :class:`~asmcmc.delta_learning.dataset.RadialProposal` these imply."""
        return RadialProposal(
            lo=self.min_com_distance,
            hi=self.max_com_distance,
            mode=self.radial_sampling,
            compact_probability=self.compact_probability,
        )


def inertia_tensor(
    positions: np.ndarray, masses: np.ndarray, com: np.ndarray | None = None
) -> np.ndarray:
    """Return the Cartesian inertia tensor in amu Angstrom^2."""
    if com is None:
        com = centre_of_mass(positions, masses)
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
    pos -= centre_of_mass(pos, masses)
    mol.set_positions(pos + target_com)
    return mol


def _placements_valid(molecules, min_atom_dist: float, max_atom_dist: float) -> bool:
    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            min_dist = minimum_inter_molecular_distance(
                molecules[i].get_positions(), molecules[j].get_positions()
            )
            if not min_atom_dist <= min_dist <= max_atom_dist:
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
    reference: Atoms,
    rng: np.random.Generator,
    settings: SamplingSettings,
) -> Atoms:
    """One non-periodic dimer, free of hard clashes.

    Orientations and separation are uniform (:func:`draw_uniform`);
    the geometry window (``min_atom_distance``/``max_atom_distance``) is what
    shapes the campaign, not a taxonomy. A rejected draw costs no MLIP call --
    only the geometry below is evaluated -- so the low clearance rate of a
    uniform orientation at close range (measured ~3% at 3.6 A, vs ~86% for a
    face-to-face-seeded one) costs CPU attempts, not budget.
    """
    for _ in range(settings.max_placement_attempts):
        u0, u1, r_vec = draw_uniform(rng, settings.radial())
        normals = [u0, u1]
        coms = [np.zeros(3), r_vec]

        spins = rng.uniform(0.0, 2.0 * np.pi, size=2)
        rotations = rotation_to_normal(np.asarray(normals), spins)
        molecules = [_place(reference, rotations[k], coms[k]) for k in range(2)]

        if _placements_valid(
            molecules, settings.min_atom_distance, settings.max_atom_distance
        ):
            return _assemble(molecules, {})

    raise RuntimeError(
        f"Failed to place a clash-free dimer in "
        f"{settings.max_placement_attempts} attempts. Consider widening "
        "--min-atom-distance/--max-atom-distance."
    )


def molecule_indices(atoms: Atoms) -> list[np.ndarray]:
    ids = np.asarray(atoms.arrays["molecule_id"], dtype=int)
    return [np.flatnonzero(ids == i) for i in range(int(ids.max()) + 1)]


def molecular_geometry_metadata(atoms: Atoms) -> dict[str, np.ndarray]:
    positions = atoms.get_positions()
    masses = atoms.get_masses()

    coms, axes = [], []
    for idx in molecule_indices(atoms):
        com = centre_of_mass(positions[idx], masses[idx])
        tensor = inertia_tensor(positions[idx], masses[idx], com)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        coms.append(com)
        # Columns are principal axes in the laboratory Cartesian frame.
        axes.append(eigenvectors)

    return {
        "molecular_com": np.asarray(coms),
        "principal_axes": np.asarray(axes),
    }


def gbq_baseline(atoms: Atoms, potential=CACELLI_POTENTIAL) -> dict:
    """The Delta-learning baseline: this cluster's energy under GB+Q.

    Stored per frame so a fit can form ``E_UMA - E_GBQ`` without re-deriving
    cluster geometry. Uses :func:`asmcmc.mc.coarse_graining.coarse_grain_frame`.

    Returned in eV, matching the MLIP energies, and summed over the cluster's
    distinct molecule pairs.
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


def evaluate_energy(atoms: Atoms, calculator) -> float:
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    return energy


def energy_decomposition(
    atoms: Atoms,
    calculator,
    cluster_energy: float,
    mode: str,
    rigid_monomer_energy: float | None = None,
) -> dict:
    """Isolated-monomer references and the pair interaction energy.

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
                evaluate_energy(subset_atoms(atoms, [i]), calculator)
                for i in range(n_mol)
            ]
        )

    return {
        "monomer_energies": monomer_energies,
        "interaction_energy": float(cluster_energy - monomer_energies.sum()),
    }


def attach_stored_results(atoms: Atoms, energy: float, extra_info: dict) -> Atoms:
    """Detach the live MLIP and attach portable ASE single-point results."""
    stored = atoms.copy()
    stored.info.update(extra_info)
    stored.calc = SinglePointCalculator(stored, energy=energy)
    return stored


def verify_saved_frame(atoms: Atoms) -> None:
    """Cheap consistency checks before writing."""
    n_mol = int(atoms.info["n_molecules"])
    required_shapes = {
        "molecular_com": (n_mol, 3),
        "principal_axes": (n_mol, 3, 3),
        "or_vec": (n_mol, 3),
    }
    for key, shape in required_shapes.items():
        arr = np.asarray(atoms.info[key])
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} has invalid shape/data: {arr.shape}")

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
    could not do that.
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
    flush_every: int,
    progress: bool = False,
    progress_queue=None,
) -> dict:
    """Generate one shard, appending incrementally so a crash loses ~nothing.

    Takes ``settings_dict`` rather than a ``SamplingSettings`` so the payload
    pickles cleanly into a spawned worker.

    Progress is reported one of two ways because the two call paths differ.
    ``progress_queue`` is for the pooled path: a spawned worker cannot draw to the
    terminal without shards fighting over the same lines, so it posts counts and the
    parent owns the single bar. ``progress`` alone drives a local bar for the
    in-process single-shard path.
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
        # One evaluation for the whole shard: energy is rotation-invariant.
        rigid_monomer_energy = evaluate_energy(reference.copy(), calculator)

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
        try:
            cluster = make_cluster(reference, rng, settings)
            energy = evaluate_energy(cluster, calculator)
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
                "config_index": index,
                "shard": shard,
                "generator_seed": seed,
                "mlip_model": model,
                "mlip_task": "omol",
                "rigid_monomers": True,
                "radial_sampling": settings.radial_sampling,
                "energy_units": "eV",
                "length_units": "Angstrom",
            }
            stored = attach_stored_results(cluster, energy, info)
            verify_saved_frame(stored)
            buffer.append(stored)
            written += 1
            attempt = 0

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
    decomposition: str = "monomers",
    flush_every: int = 10,
    max_workers: int = 4,
) -> list[dict]:
    """Run a sharded, resumable generation campaign.

    Idempotent: a shard already holding its target count is skipped, so
    re-running the same command finishes an interrupted campaign.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or SamplingSettings()  # validates itself

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
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
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


def load_dataset_config(out_dir) -> dict:
    """A campaign's ``dataset_config.json``, or ``{}`` if it has none."""
    path = Path(out_dir) / CONFIG_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m asmcmc.delta_learning.dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        choices=("none", "monomers"),
        default="monomers",
        help="none: cluster energy only; monomers: monomer and interaction energies.",
    )

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument(
        "--radial-sampling",
        choices=RADIAL_SAMPLINGS,
        default=default_settings.radial_sampling,
        help=(
            "volume-uniform: flat in r^3 over the whole range. mixture: "
            "concentrate compact-probability of the mass on the 3.4-6 A wells."
        ),
    )
    geometry.add_argument(
        "--compact-probability",
        type=float,
        default=default_settings.compact_probability,
        help="Mass placed on the wells under --radial-sampling mixture.",
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
        "--min-atom-distance",
        type=float,
        default=default_settings.min_atom_distance,
        help="Floor on realised minimum atom-atom separation; keeps hard-core "
        "clash geometries (which dominate the UMA-vs-GBQ loss) out of the "
        "campaign.",
    )
    geometry.add_argument(
        "--max-atom-distance",
        type=float,
        default=default_settings.max_atom_distance,
        help="Ceiling on realised minimum atom-atom separation; UMA returns "
        "exactly zero interaction past its ~6 A horizon, so nothing beyond it "
        "is learnable.",
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
            min_atom_distance=args.min_atom_distance,
            max_atom_distance=args.max_atom_distance,
            radial_sampling=args.radial_sampling,
            compact_probability=args.compact_probability,
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
        flush_every=args.flush_every,
        max_workers=args.max_workers,
    )
    total = sum(r["total"] for r in results)
    print(f"\n{total} frames across {len(results)} shards in {args.out_dir}")


if __name__ == "__main__":
    cli()
