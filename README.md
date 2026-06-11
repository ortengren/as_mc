# asmcmc

**AniSOAP Markov Chain Monte Carlo** — Monte Carlo simulation of a system of
benzenes, coarse-grained as ellipsoidal particles.

## Overview

This project serves as a real-world benchmark of AniSOAP, designed to compare
its effectiveness with other approaches (especially analytic potentials, i.e.
the Gay-Berne potential).

The bulk of the simulation logic lives in `metropolis.py`.  The
`MetropolisCalculator` class contains methods for performing full simulation
runs, including equilibration.  Simulations can be read and evaluated using
tools from `measurements.py`.  In particular, the `TrajectoryAnalyzer` class
offers an efficient way to determine quantities of interest, such as heat
capacity and orientational correlation.  

The codebase is designed with modularity in mind so that new measurements can
be made (by defining subclasses of `Measurement`) or new energy calculation
methods can be used.

## Usage

There is no packaged install yet. You'll need a Python environment with the
dependencies below available:
ASE, NumPy, SciPy, `anisoap`, `metatensor`, `scikit-matter`, `tqdm`,
`matplotlib`, and `pytest`. Run the commands and snippets below from the
repository root.

### Map the model's phase behaviour (NVT scan)

```bash
python nvt_scan.py      # writes scan_results/nvt_scan.csv and nvt_scan.png
```

Sweeps a grid of reduced temperatures and densities (T\*, ρ\*), running
constant-volume MC **on the GB + quadrupole potential** at each point and
recording the reduced energy, heat capacity, and nematic order parameter.
These observables should indicate the occurrence of a phase transformation.
This maps the analytic baseline's own phase behaviour; AniSOAP is not yet
involved.

### Multi-temperature NPT runs

```python
from driver import run_multi_temp_trial

run_multi_temp_trial(temps=[200, 300, 400], press=0.0)
# → simulations/npt_test/{temp}/{equilibration,simulation}.db
```

### Drive the sampler directly

```python
from metropolis import MetropolisCalculator

# NPT by default; pass npt_ensemble=False for fixed-volume NVT.
# init_frame=None auto-generates a starting configuration.
metro = MetropolisCalculator(temp=300, pressure=0.0, output_dir="simulations/demo")
metro.calculate_trajectory(num_steps=200_000, num_eq_steps=100_000)
# → simulations/demo/{equilibration,simulation}.db
```

### Analyse a trajectory

```python
from measurements import TrajectoryAnalyzer, HeatCapacity, RadialDistributionFunction

analyzer = TrajectoryAnalyzer("simulations/demo/simulation.db")
analyzer.add_measurement("Cv", HeatCapacity(temperature=300, num_particles=125))
analyzer.add_measurement("rdf", RadialDistributionFunction(r_max=20.0, num_bins=50))
results = analyzer.run_analysis()
```

Add your own observable by subclassing `Measurement` (implement `compute` and
`finalize`).

## Project structure

| File | Role |
| ---- | ---- |
| `metropolis.py` | `MetropolisCalculator` — the Metropolis-Hastings sampler (NPT default, NVT optional) and full simulation runs |
| `potentials.py` | Gay-Berne + quadrupole pair potential (Walsh benzene parameters) |
| `trial_moves.py` | Trial moves: translation, quaternion rotation, isotropic volume scaling |
| `initialize.py` | `generate_random_config` — randomized starting configuration at a target density |
| `measurements.py` | Observable framework: `Measurement` base, `TrajectoryAnalyzer`, and ready-made measurements (energy, heat capacity, RDF, orientational correlation, nematic order) |
| `driver.py` | Multi-temperature NPT runs plus AniSOAP feature generation |
| `nvt_scan.py` | Reduced-units (T\*, ρ\*) phase scan |

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
