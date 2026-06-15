# asmcmc

**AniSOAP Markov Chain Monte Carlo** — Monte Carlo simulation of a system of
benzenes, coarse-grained as ellipsoidal particles.

## Overview

This project serves as a real-world benchmark of AniSOAP, designed to compare
its effectiveness with other approaches (especially analytic potentials, i.e.
the Gay-Berne potential).

The bulk of the simulation logic lives in `asmcmc/metropolis.py`.  The
`MetropolisCalculator` class contains methods for performing full simulation
runs, including equilibration.  Simulations can be read and evaluated using
tools from `asmcmc/measurements.py`.  In particular, the `TrajectoryAnalyzer`
class offers an efficient way to determine quantities of interest, such as heat
capacity and orientational correlation.

The codebase is designed with modularity in mind so that new measurements can
be made (by defining subclasses of `Measurement`) or new energy calculation
methods can be used.

## Installation

Editable install into an environment that already has the scientific stack
(conda or otherwise):

```bash
pip install -e . --no-deps   # --no-deps if conda manages your dependencies
```

Core dependencies: ASE, NumPy (≥2), SciPy, pandas, `tqdm`, `matplotlib`.
The AniSOAP feature-generation modules additionally need `anisoap`,
`metatensor`, `scikit-matter`, `scikit-learn` (`pip install -e ".[anisoap]"`).
Tests need `pytest`.

## Usage

Run the commands and snippets below from the repository root (output paths are
relative to the working directory).

### Map the model's phase behaviour (NVT scan)

```bash
python -m asmcmc.nvt_scan                # writes results/scan_results/nvt_scan.csv + figures
python -m asmcmc.nvt_scan --plot-only    # re-render figures from the existing CSV
```

Sweeps a grid of reduced temperatures and densities (T\*, ρ\*), running
constant-volume MC **on the GB + quadrupole potential** at each point and
recording the reduced energy, heat capacity, and nematic order parameter.
These observables should indicate the occurrence of a phase transformation.
This maps the analytic baseline's own phase behaviour; AniSOAP is not yet
involved.

### Fit the GB + quadrupole potential

The Gay-Berne + quadrupole parameters that the simulations use are fit to a
dataset of ellipsoid configurations with reference DFT energies. The fit has a
command-line front-end:

```bash
python -m asmcmc.fitting.run                       # full fit, both weightings
python -m asmcmc.fitting.run --index :200          # fast smoke run on 200 frames
python -m asmcmc.fitting.run --workers -1 --maxiter 100 --tol 1e-3
```

It builds the dataset once, makes a single train/test split, fits the
parameter vector `[sigma0, eps0, kappa, kappa_prime, mu, nu, Q, E_intra]` with
`scipy.optimize.differential_evolution` (a global optimiser — no starting guess
needed), then writes parameters, metrics, a report, and diagnostic plots.

#### What gets fit

Each run fits one or more **weighting variants** on the *same* split, so they
are directly comparable:

- `uniform` — equal weight on every frame. This is the production fit the
  simulations use: the Boltzmann weighting concentrates the objective on too few
  frames to earn its keep on held-out error (see `results/fitting/summary/`).
- `boltzmann` — weights each frame by `exp(-alpha * E)`, emphasising
  near-equilibrium (low-energy) configurations. Kept as a reference.

By default both are fit. Restrict with `--weighting` (repeatable):

```bash
python -m asmcmc.fitting.run --weighting boltzmann          # just one
python -m asmcmc.fitting.run --weighting boltzmann --weighting uniform   # both, explicit
```

#### Options

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--dataset PATH` | `data/xyz_files/ellipsoids_with_axes_and_energies.xyz` | Training `.xyz` (per-molecule reference energies) |
| `--cutoff Å` | `15.0` | Lattice-sum cutoff; matches the MC neighbour radius |
| `--out-dir DIR` | `results/fit_gb` | Output root (use a fresh dir to keep previous runs) |
| `--cache-dir DIR` | `<out-dir>/cache` | Shared built-dataset cache; point several runs at one dir to skip the per-run neighbour-list rebuild |
| `--weighting {boltzmann,uniform}` | both | Weighting variant; repeatable |
| `--index SLICE` | `:` (all) | `ase.io.read` frame slice, e.g. `:200` for a smoke run |
| `--test-frac F` | `0.2` | Held-out fraction for the test metrics |
| `--split-seed N` | `0` | Seed for the train/test split |
| `--fit-seed N` | `0` | Seed for `differential_evolution` |
| `--alpha A` | `2.90` (~4000 K) | Boltzmann weight scale (1/eV); `alpha = 1/(k_B·T)`. Ignored by `uniform` |
| `--no-progress` | (bar on) | Silence the tqdm progress bar (e.g. in logs) |

`differential_evolution` knobs (omit one to use SciPy's own default):

| Flag | SciPy default | Meaning |
| ---- | ------------- | ------- |
| `--workers N` | `1` | Parallel workers; `-1` uses every core |
| `--maxiter N` | `1000` | Max DE generations |
| `--popsize N` | `15` | Population-size multiplier |
| `--tol F` | `0.01` | Relative convergence tolerance |

> **Parallelism note:** the per-evaluation energy sum is memory-bandwidth-bound,
> so `--workers -1` plateaus at roughly a **3×** speedup rather than scaling with
> core count. It still pays off — a `--maxiter 100` Boltzmann fit drops from
> ~27 min serial to ~9 min — but don't expect linear scaling.

#### Outputs

Everything lands under `--out-dir` (default `results/fit_gb`, gitignored):

```text
results/fit_gb/
├── cache/                       # cached built dataset (reused on re-runs)
├── comparison.csv               # one row per weighting: test RMSE/MAE/R² + objective
├── boltzmann/
│   ├── params.json              # fitted parameters (with units)
│   ├── metrics.json             # train/test RMSE, MAE, max error, R² + sanity checks
│   ├── fit_report.md            # human-readable summary
│   ├── parity.png               # predicted vs reference energy
│   ├── residuals_vs_energy.png  # error vs reference energy
│   └── residuals_vs_nn_distance.png   # error vs nearest-neighbour distance
└── uniform/
    └── …                        # same set of files for the uniform variant
```

`comparison.csv` is the quickest read-off; `params.json` holds the numbers you
plug back into the potential. The plots diagnose *where* the fit misses (e.g.
high error concentrated at close contacts, which carry near-zero Boltzmann
weight and so are correctly de-emphasised).

#### Typical workflow

```bash
# 1. Sanity-check the pipeline quickly on a slice (seconds, not minutes):
python -m asmcmc.fitting.run --index :200 --out-dir results/fit_gb_smoke

# 2. Real fit, parallel, to a named output dir so prior runs are preserved:
python -m asmcmc.fitting.run --workers -1 --maxiter 100 --tol 1e-3 \
    --out-dir results/fit_gb_full

# 3. Read results/fit_gb_full/comparison.csv and inspect the plots.
```

To explore the Boltzmann weighting sharpness, sweep `--alpha`, each fit to its
own dir under a shared cache. The `scripts/` wrappers do this for you:

```bash
# Alpha sweep + uniform reference -> results/fitting/alpha_scan/{alpha_<a>,uniform}/
./scripts/run_fits.sh                 # default sweep; or pass alphas: ./scripts/run_fits.sh 1.5 2.0

# Higher-budget multi-seed finalisation -> results/fitting/multiseed/<campaign>/seed_{0,1,2}/
./scripts/run_fit_seeds.sh            # uniform (default)
WEIGHTING=boltzmann ALPHA=2.90 ./scripts/run_fit_seeds.sh   # boltzmann campaign
```

Equivalently, a single run with an explicit shared cache:

```bash
python -m asmcmc.fitting.run --weighting boltzmann --alpha 2.90 \
    --workers -1 --popsize 22 --maxiter 250 --tol 1e-3 \
    --cache-dir results/fitting/cache \
    --out-dir results/fitting/alpha_scan/alpha_2.90
```

Cross-run comparison figures are built from this tree by
`python -m asmcmc.fitting.summary` (writes `results/fitting/summary/`).

### Temperature × pressure grid runs (with reporting)

```python
from asmcmc.simulation.run import run_grid

run_grid(temps=[200, 300, 400], pressures=[1e-6, 1e-5])
# runs every (T, P) combination →
# results/simulations/npt_<datetime>/T{temp}_P{pressure}/
#   {equilibration,simulation}.db
#   run_config.json, observables.json, observables.npz, report.md
#   energy_trace.png, acceptance_trace.png, nematic_order.png, rdf.png,
#   orientational_correlation.png
```

Or from the command line. The CLI has three subcommands — `equilibrate`,
`continue-eq`, and `produce` — one per action:

```bash
python -m asmcmc.simulation.run produce --temps 200 300 400 --pressures 1e-6 1e-5
```

Pass `--repeat x` (or `repeat=x`) to run `x` replicas with **different initial
configurations** at each (T, P), to quantify run-to-run uncertainty. Each
replica lands in its own `rep{i}/` subdir and a cross-replica
`summary.json`/`summary.md` (mean/std/sem of every scalar observable) is written
to the point dir:

```bash
python -m asmcmc.simulation.run produce --temps 300 --repeat 5 --seed 0
# results/simulations/npt_<datetime>/T300_P1e-06/
#   rep00/ rep01/ … rep04/   (full artifact set each)
#   summary.json, summary.md  (mean ± std over replicas)
```

#### Equilibrate first, then check convergence before production

To verify each grid point has equilibrated (e.g. the cell volume has plateaued)
*before* committing to expensive production, run the `equilibrate` subcommand,
then `produce --from` to restart production from those equilibrated configs:

```bash
# 1. equilibrate every (T, P); writes equilibration.db + trace plots per point
python -m asmcmc.simulation.run equilibrate --temps 200 300 400 \
    --out-dir results/simulations/eq_run
# inspect results/simulations/eq_run/T*/volume_trace.png for a plateau

# 2. start production from each point's final equilibrated frame (production-only)
python -m asmcmc.simulation.run produce --from results/simulations/eq_run \
    --temps 200 300 400
```

`equilibrate` writes `equilibration.db`, `equilibration_config.json`, and the
trajectory trace PNGs (`volume_trace.png`/`energy_trace.png`/…) per point.
`produce --from` reuses each point's final frame *and* its tuned trial-move
widths and runs production-only (no re-equilibration). To render the trace plots
for any existing db on their own: `python -m asmcmc.simulation.plots <path/to/db>`.

If a point hasn't equilibrated for long enough, **continue the equilibration in
place** for more steps with `continue-eq`:

```bash
# add 20k more equilibration steps, appended to the same equilibration.db
python -m asmcmc.simulation.run continue-eq --from results/simulations/eq_run \
    --temps 200 300 400 --num-eq-steps 20000
# the regenerated volume_trace.png now covers the whole combined run
```

Here `--num-eq-steps` is the number of *additional* steps; the step counter
continues so the trace plots stay monotonic across the resumed run.

To act on **only some points** (e.g. just the ones that haven't converged), pass
an explicit `--points` list of `T,P` pairs, which overrides `--temps`/`--pressures`:

```bash
# continue equilibration only at (200, 1e-6) and (400, 1e-5); other points untouched
python -m asmcmc.simulation.run continue-eq --from results/simulations/eq_run \
    --points 200,1e-6 400,1e-5 --num-eq-steps 20000
```

`--points` works for `produce` too (`points=[(T, P), …]` in `run_grid`/
`equilibrate_grid`), to run an arbitrary subset of the grid.

`--repeat` works through the whole equilibrate → check → produce workflow. With
`equilibrate --repeat n` each point gets `n` replicas with different initial
configs (seeds `seed`, `seed+1`, …) in their own `rep{i}/` subdirs, each with its
own `equilibration.db` and trace plots to check (and continue) independently.
Then `produce --from … --repeat n` starts each replica's production from its
matching `rep{i}/equilibration.db`:

```bash
# equilibrate 3 replicas per point, check each rep{i}/volume_trace.png
python -m asmcmc.simulation.run equilibrate --repeat 3 --seed 0 \
    --temps 200 300 400 --out-dir results/simulations/eq_run
# run production for all 3 replicas, each resuming its own equilibrated config
python -m asmcmc.simulation.run produce --from results/simulations/eq_run --repeat 3 --seed 0 \
    --temps 200 300 400
```

(Pass the same `--repeat`/`--seed` to the `produce`/`continue-eq` step as to the
`equilibrate` step.)

### Drive the sampler directly

```python
from asmcmc.metropolis import MetropolisCalculator

# NPT by default; pass npt_ensemble=False for fixed-volume NVT.
# init_frame=None auto-generates a starting configuration.
metro = MetropolisCalculator(temp=300, pressure=0.0, output_dir="results/simulations/demo")
metro.calculate_trajectory(num_steps=200_000, num_eq_steps=100_000)
# → results/simulations/demo/{equilibration,simulation}.db
```

### Analyse a trajectory

```python
from asmcmc.measurements import TrajectoryAnalyzer, HeatCapacity, RadialDistributionFunction

analyzer = TrajectoryAnalyzer("results/simulations/demo/simulation.db")
analyzer.add_measurement("Cv", HeatCapacity(temperature=300, num_particles=125))
analyzer.add_measurement("rdf", RadialDistributionFunction(r_max=20.0, num_bins=50))
results = analyzer.run_analysis()
```

Add your own observable by subclassing `Measurement` (implement `compute` and
`finalize`).

## Project structure

| Path | Role |
| ---- | ---- |
| `asmcmc/` | The installable package — everything importable lives here |
| `asmcmc/metropolis.py` | `MetropolisCalculator` — the Metropolis-Hastings sampler (NPT default, NVT optional) and full simulation runs |
| `asmcmc/potentials.py` | Gay-Berne + quadrupole pair potential (Walsh benzene parameters) |
| `asmcmc/trial_moves.py` | Trial moves: translation, quaternion rotation, isotropic volume scaling |
| `asmcmc/initialize.py` | `generate_random_config` — randomized starting configuration at a target density |
| `asmcmc/measurements.py` | Observable framework: `Measurement` base, `TrajectoryAnalyzer`, and ready-made measurements (energy, heat capacity, RDF, orientational correlation, nematic order) |
| `asmcmc/simulation/` | MC run orchestration + reporting: (T, P) grid runs writing per-point config/observables/report/plots (`python -m asmcmc.simulation.run`) |
| `asmcmc/nvt_scan.py` | Reduced-units (T\*, ρ\*) phase scan (`python -m asmcmc.nvt_scan`) |
| `asmcmc/fitting/` | Fit the GB + quadrupole potential to reference energies (`python -m asmcmc.fitting.run`) |
| `asmcmc/generate_cg_reps.py` | AniSOAP coarse-grained representation generation |
| `tests/` | pytest suite |
| `data/` | Input datasets (`data/xyz_files/` crystal structures; bulk files kept on disk, not in VCS) |
| `notebooks/` | Exploratory analysis notebooks (not load-bearing) |
| `results/` | Regenerable outputs (gitignored): `results/simulations/` NPT runs, `results/scan_results/` NVT scans |

Trajectories are stored as ASE `.db` files.

## Testing

```bash
pytest tests/
```

Covers the potentials, trial moves, initialization, the measurement framework,
an end-to-end integration run, and the NVT scan.

## Status

This project is very much still a work-in progress.  The core logic
seems to work well, and the test suite currently passes (at least on my
machine!).  At present, only the GB + quadrupole potential is implemented, and
 _not_ the AniSOAP ML potential.

Currently, I am working on ensuring that simulations using the GB potential
give reasonable results.  Next, I will implement energy calculation via AniSOAP,
after which the two methods can finally be compared.
