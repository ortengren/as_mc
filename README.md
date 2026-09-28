# asmcmc

**AniSOAP Markov-chain Monte Carlo.** Monte Carlo simulation of benzene, with each
molecule coarse-grained to a rigid oblate ellipsoid. It is also an effort to
improve the potential those ellipsoids interact through, using a machine-learned
AniSOAP correction.

## Where things stand

- **The sampler is done and validated.** NPT/NVT Metropolis MC of rigid ellipsoids.
  With Cacelli et al.'s (2004) Gay-Berne + quadrupole potential (GBQIII) it
  reproduces their 100 K crystal quantitatively.
- **That potential is physically wrong in known ways.** It melts at least 130 K too
  low, its liquid is ~28% too dense, and it prefers the wrong crystal polymorph.
- **Refitting GB+Q to DFT crystal energies fails the physics test.** The refit fits
  the energies well but makes the stacked dimer repulsive. So the current work is a
  **Δ-learning correction**: ridge regression on AniSOAP descriptors, trained on
  dimers labelled by the UMA machine-learned potential. The best model so far
  improves the dimer-well error by 23% (0.465 → 0.359 kcal/mol). It is not yet used
  in MC.

The reasoning and the numbers behind all of this are in
[`docs/findings.md`](docs/findings.md). Read it before changing the physics.

## Install

The code runs in a conda/mamba environment with Python ≥ 3.11. It is tested with
Python 3.12, NumPy 2.2, ASE 3.29 and SciPy 1.16. Install the package in editable
mode:

```bash
pip install -e . --no-deps    # into an environment that already has the dependencies
pip install -e ".[dev]"       # or let pip install the core dependencies and pytest
```

`asmcmc.mc` needs only the core dependencies. Two optional pieces:

- **AniSOAP** (`delta_learning` descriptors and model): install AniSOAP from a
  clone of the lab's AniSOAP repository (it builds a Rust extension), then the
  `anisoap` extra (`metatensor`, `scikit-learn`).
- **UMA** (dataset labels): the `uma` extra installs `fairchem-core` (tested with
  2.12). The UMA checkpoint is gated on Hugging Face, so run
  `huggingface-cli login` once.

## Test

```bash
pytest
```

About 330 tests, ~1.5 minutes. They need neither a GPU nor fairchem (UMA is
replaced by a stub), and everything they read is tracked. The AniSOAP tests are
skipped if AniSOAP is not installed.

## Quick start

Units are eV, Å and K; pressure is in eV/Å³ (`asmcmc.units.ATM_TO_EV_PER_A3` is 1 atm).

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

Plot what a run did (structure, phase, acceptance, energy):

```bash
python scripts/plot_run.py results/simulations/demo --db simulation.db
```

A run directory holds `equilibration.db` and/or `simulation.db` (ASE databases,
one row per recorded block) plus a write-once `run_config.json`.
`MetropolisSampler.from_equilibration(run_dir)` rebuilds a sampler from them, and
`continue_equilibration(run_dir, extra_steps)` extends a run in place.

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
    dimer_benchmark.py     the physics gate every potential must pass
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
```

`delta_learning` and `fitting_gbq` build on `mc`; `mc` imports neither of them.

## Workflows

**A validation state point** (herringbone start, 1 atm, the protocol from
`docs/findings.md` §2). There are five temperatures, each with a fixed seed and run
directory under `results/validation/`:

```bash
python scripts/run_herringbone.py --temp 150 equilibrate   # 1e6 steps from the crystal
python scripts/run_herringbone.py --temp 150 resume        # 9.2e6 more
python scripts/run_herringbone.py --temp 150               # 1.5e7 production steps, then measure
```

These are long runs (hours); check `plot_run.py` output before moving on.

**The Δ-learning pipeline:**

```bash
python -m asmcmc.delta_learning.dataset --n-configs 5000 --out-dir results/cluster_new   # UMA labels
python -m asmcmc.delta_learning.sweep --campaign results/cluster_new --out-dir results/anisoap_fit_new
```

`dataset` needs fairchem and runs for hours; keep `--max-workers` at 4 on a 14 GB
machine. Then use `dataset_analysis.qa_report` on the new campaign. The sweep writes
`comparison.csv`: rank points by `improves_on_baseline` and `well_rmse_kcal`, and
use `--regate` to re-score a finished sweep against a new reference without
refitting.

**The GB+Q refit:** `python -m asmcmc.fitting_gbq.run --help`. The campaign in
`results/fitting/` was made with `scripts/run_fits.sh` and
`scripts/run_fit_seeds.sh`; `python -m asmcmc.fitting_gbq.summary` redraws its
figures.

**Regenerating the UMA dimer reference:** `python scripts/uma_cacelli_dimers.py`,
which writes `data/uma_dimers/`. It takes about two minutes on CPU. How the
supplement's Euler angles are read is explained in `docs/findings.md` §3.

Other scripts: `export_xyz.py` (a run's db as extended XYZ, for OVITO) and
`build_herringbone_motif.py` (coarse-grain a Pbca cif into the starting motif).

## Branches

- `main`: this layout.
- `archive/pre-cleanup`: the tree before the September 2026 cleanup, including
  the (T, P) grid scan and replica-statistics tools, historical notebooks and
  archived scripts.
- `pre-revert-snapshot`: the July–August 2026 work reverted on 2026-08-11 (a
  unified driver, the N = 500 fixed-width protocol, the `"ZXY"` Euler analysis).
- `signac-flow-htc-impl`: the signac/CHTC harness for the planned (T, P) ×
  potential sweep. It predates this layout; see the known issues in
  `docs/findings.md`.
