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

### Run a simulation

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
| `asmcmc/driver.py` | Multi-temperature NPT runs plus AniSOAP feature generation (`python -m asmcmc.driver`) |
| `asmcmc/fitting_gbq/` | Fit the GB + quadrupole potential to reference energies (`python -m asmcmc.fitting_gbq.run`) |
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
