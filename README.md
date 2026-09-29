# asmcmc

asmcmc (AniSOAP Markov-chain Monte Carlo) runs Monte Carlo simulations of benzene
in which each molecule is coarse-grained to a rigid oblate ellipsoid. It's also
where we're trying to improve the potential those ellipsoids interact through,
using a machine-learned AniSOAP correction.

## Where things stand

The sampler works and has been validated. It does NPT and NVT Metropolis MC of
rigid ellipsoids, and with Cacelli et al.'s (2004) Gay-Berne plus quadrupole
potential (GBQIII) it reproduces their 100 K crystal quantitatively.

The problem is the potential itself, which is wrong in ways we understand. It
melts at least 130 K too low, its liquid is about 28% too dense, and it prefers
the wrong crystal polymorph.

Refitting GB+Q to DFT crystal energies didn't fix this: the refit matches the
energies well but makes the stacked dimer repulsive. So the current approach is a
Δ-learning correction, which is a ridge regression on AniSOAP descriptors trained
on dimers labelled by the machine-learned potential UMA. The best model so far
cuts the error in the dimer wells by 23% (from 0.465 to 0.359 kcal/mol). It
hasn't been used in MC yet.

[`docs/findings.md`](docs/findings.md) has the reasoning and the numbers behind
all of this. Please read it before changing any of the physics.

## Install

The code runs in a conda or mamba environment with Python 3.11 or newer. It's
tested with Python 3.12, NumPy 2.2, ASE 3.29 and SciPy 1.16. Install the package
in editable mode:

```bash
pip install -e . --no-deps    # into an environment that already has the dependencies
pip install -e ".[dev]"       # or let pip install the core dependencies and pytest
```

`asmcmc.mc` only needs the core dependencies. There are two optional extras:

- **AniSOAP**, for the `delta_learning` descriptors and model. Install AniSOAP
  from a clone of the lab's AniSOAP repository (it builds a Rust extension), then
  install the `anisoap` extra (`metatensor`, `scikit-learn`).
- **UMA**, for labelling datasets. The `uma` extra installs `fairchem-core`
  (tested with 2.12). The UMA checkpoint is gated on Hugging Face, so you'll need
  to run `huggingface-cli login` once.

## Test

```bash
pytest
```

There are about 330 tests, and they take around a minute and a half. They don't
need a GPU or fairchem (UMA is replaced by a stub), and everything they read is
tracked in git. The AniSOAP tests are skipped if AniSOAP isn't installed.

## Quick start

Units are eV, Å and K throughout, and pressure is in eV/Å³
(`asmcmc.units.ATM_TO_EV_PER_A3` is 1 atm).

```python
from asmcmc.mc.initialize import HerringboneLatticeInitializer
from asmcmc.mc.measurements import NematicOrderParameter, TrajectoryAnalyzer
from asmcmc.mc.metropolis import MetropolisSampler
from asmcmc.units import ATM_TO_EV_PER_A3

sampler = MetropolisSampler(
    temp=150.0,
    pressure=1.0 * ATM_TO_EV_PER_A3,
    initializer=HerringboneLatticeInitializer(n_particles=128, seed=1),
    nl_radius=6.8,
    output_dir="results/simulations/demo",
)
sampler.calculate_trajectory(num_steps=20_000, num_eq_steps=20_000, max_or_delt=0.25)
# -> results/simulations/demo/{equilibration.db, simulation.db, run_config.json}

analyzer = TrajectoryAnalyzer("results/simulations/demo/simulation.db")
analyzer.add_measurement("order", NematicOrderParameter())
print(analyzer.run_analysis()["order"]["S"])
```

To plot what a run did (structure, phase, acceptance and energy):

```bash
python scripts/plot_run.py results/simulations/demo --db simulation.db
```

A run directory holds `equilibration.db` and/or `simulation.db`, which are ASE
databases with one row per recorded block, plus a `run_config.json` that is
written once when the run starts. `MetropolisSampler.from_equilibration(run_dir)`
rebuilds a sampler from these files, and `continue_equilibration(run_dir,
extra_steps)` extends a run in place.

## Layout

```text
asmcmc/
  paths.py, units.py     data_path(); physical constants and unit conversions
  mc/                    the sampler, and analysis of its runs
    metropolis.py          MetropolisSampler, continue_equilibration
    potentials.py          GBQPotential, CACELLI_POTENTIAL, calc_total_energy
    trial_moves.py         translation, rotation and volume moves
    initialize.py          random, columnar and herringbone starting lattices
    run_config.py          RunConfig (run_config.json)
    measurements.py        observables: g(r), order parameters, heat capacity, ESS...
    diagnostics.py         figures and xyz export for one run directory
    coarse_graining.py     atomistic frames -> one ellipsoid per molecule
  delta_learning/        the AniSOAP correction
    dimer_benchmark.py     the dimer benchmark every candidate potential must pass
    uma.py                 the UMA labeller
    dataset.py             UMA-labelled dimer dataset generation
    dataset_analysis.py    dataset QA and summaries
    descriptors.py         AniSOAP descriptors
    model.py               the ridge Δ-model; AniSOAPDeltaPotential
    sweep.py               hyperparameter sweep
  fitting_gbq/           GB+Q refit to DFT crystal energies (not used for MC)
scripts/                 command-line drivers (below)
tests/                   pytest suite, one file per module
data/                    inputs; see data/README.md
docs/                    findings.md, cacelli_protocol_diff.md
notebooks/               analyses (most read local results/)
results/                 run outputs: local and gitignored, except results/fitting/
TP-sweeps/               signac workspace for the planned (T, P) sweep; its code is
                         on the signac-flow-htc-impl branch
```

`delta_learning` and `fitting_gbq` build on `mc`, but `mc` doesn't import either
of them.

## Workflows

### A validation state point

This starts from the herringbone crystal at 1 atm and follows the protocol in
`docs/findings.md` §2. There are five temperatures, each with a fixed seed and
its own run directory under `results/validation/`:

```bash
python scripts/run_herringbone.py --temp 150 equilibrate   # 1e6 steps from the crystal
python scripts/run_herringbone.py --temp 150 resume        # 9.2e6 more
python scripts/run_herringbone.py --temp 150               # 1.5e7 production steps, then measure
```

These runs take hours. Look at the `plot_run.py` figures before moving on to the
next stage.

### The Δ-learning pipeline

```bash
python -m asmcmc.delta_learning.dataset --n-configs 5000 --out-dir results/cluster_new   # UMA labels
python -m asmcmc.delta_learning.sweep --campaign results/cluster_new --out-dir results/anisoap_fit_new
```

`dataset` needs fairchem and runs for hours. Keep `--max-workers` at 4 on a
machine with 14 GB of RAM. When a campaign finishes, check it with
`dataset_analysis.qa_report`. The sweep writes `comparison.csv`; rank its points
by `improves_on_baseline` first and `well_rmse_kcal` second. If the reference
data changes, `--regate` re-scores a finished sweep without refitting it.

### The GB+Q refit

Run `python -m asmcmc.fitting_gbq.run --help` to see the options. The campaign in
`results/fitting/` was made with `scripts/run_fits.sh` and
`scripts/run_fit_seeds.sh`, and `python -m asmcmc.fitting_gbq.summary` redraws its
figures.

### Regenerating the UMA dimer reference

`python scripts/uma_cacelli_dimers.py` rewrites `data/uma_dimers/` and takes
about two minutes on a CPU. `docs/findings.md` §3 explains how it reads the
supplement's Euler angles.

### Other scripts

`export_xyz.py` writes a run's db as extended XYZ for viewing in OVITO, and
`build_herringbone_motif.py` coarse-grains a Pbca cif into the starting motif.

## Branches

- `main` has this layout.
- `archive/pre-cleanup` has the tree as it was before the September 2026
  cleanup, including the (T, P) grid scan, the replica-statistics tools, the
  historical notebooks and the archived scripts.
- `signac-flow-htc-impl` has the signac/CHTC setup for the planned (T, P) ×
  potential sweep. It predates this layout; see the known issues in
  `docs/findings.md`.
