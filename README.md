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
 *not* the AniSOAP ML potential.

Currently, I am working on ensuring that simulations using the GB potential
give reasonable results.  Next, I will implement energy calculation via AniSOAP,
after which the two methods can finally be compared.

## TODO: path to an AniSOAP potential (as of 2026-07-16)

**Motivating finding:** a potential can fit condensed-phase per-configuration
energies almost perfectly and still get the *pair interaction* badly wrong —
per-molecule energies are sums over many pairs, so pairwise errors cancel in
the fit target, and MC then samples exactly the geometries the fit never
constrained. The GB+Q refit to the crystal dataset is the cautionary example:
~3 kcal/mol test RMSE on the crystals, yet *repulsive* at the 3.9 Å cofacial
stacking distance and anti-correlated with the ab initio dimer wells (which is
why the literature Cacelli parameters — dimer-PES-fit, ~10 kcal/mol RMSE on
the same crystals — nonetheless give far more realistic simulations).
Consequence: every candidate potential must pass the **dimer-well benchmark**
and an **in-MC physics test**, not just held-out energy parity.

- [x] **Phase 0 — validation harness.** `asmcmc/validation.py`: scores any
  `Potential` against the Cacelli et al. (2004) ab initio benzene dimer set
  (`data/new_data/3648_1_supplements/`) — well correlation/RMSE, per-family
  well depths (cofacial / parallel-displaced / T-shaped), and the fatal
  `stacking_bound` check. Tests in `tests/test_validation.py` pin the two
  reference points: Cacelli passes (0.21 kcal/mol well RMSE), the
  condensed-phase refit fails (frozen inline as the known-bad fixture).
- [ ] **Phase 1 — diagnoses.** (a) Coverage: compare pair-geometry
  distributions (r, orientation invariants) of the 6,826 training crystals
  (`data/anisoap_data/benzenes/`) against MC trajectories, to quantify how the
  dataset under-samples MC-visited stacking geometries. (b) Baseline residual:
  characterise `E_DFT − E_Cacelli` per molecule — the function the ML
  correction must learn.
- [ ] **Phase 2 — model: Cacelli-baselined Δ-learning.**
  `E = E_GBQ(Cacelli) + AniSOAP·w` (ridge on AniSOAP descriptors,
  mean-referenced target). The physical baseline hard-codes the repulsive core
  and bound π-stacking so the ML correction cannot invert them; an
  unconstrained linear model on raw energies is ruled out by the finding above.
  Blocker: the GFRE-tuned hyperparameters (`optimized_gfres.npz`) are not in
  the `data/anisoap_data` drop — obtain from the authors/SI or re-run
  `hyperparameter_tuning/gfre.py`. **Gate A:** beat Cacelli on condensed-phase
  test RMSE *and* pass the dimer benchmark.
- [ ] **Phase 3 — MC integration.** Generalise the `Potential` seam for a
  local-but-not-pairwise energy (incremental single-particle updates within
  the descriptor cutoff; full re-evaluation on volume moves). Measure the
  per-move AniSOAP descriptor cost early — it is the main feasibility risk
  (MC needs energies only, no forces). **Gate B:** rerun the herringbone
  validation protocol (100 K / 1 atm, N = 400); success = slipped-parallel
  crystal with V/molecule pulled from Cacelli's ~96 Å³ toward experiment's
  ~116 Å³.
- [ ] **Phase 4 — only if Gate B stalls: close the data gap.** Active
  learning: new PBE-D3 reference calculations (QuantumEspresso settings from
  `data/anisoap_data/benzenes/README.md`) on MC-visited and close-contact
  configurations, then retrain.
