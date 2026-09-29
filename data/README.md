# data/

This directory holds the input data the code reads. The code finds it through
`asmcmc.paths.data_path`, and you can point that somewhere else by setting
`ASMCMC_DATA_DIR`. Everything listed here is tracked in git except the files
marked *local only*, which are too large and are gitignored.

## Crystal structures and the MC starting motif

| File | What it is |
|---|---|
| `benzene_pbca_cod_7238223.cif` | Experimental Pbca benzene crystal, COD 7238223 (grown in situ at cryogenic temperature and reported at 150 K). Its cell is about 4% larger than benzene's at that temperature: 123.58 Å³ per molecule, or ρ = 1.050 g/cm³ against the accepted 1.09–1.10. |
| `benzene_Pbca_csd_1108750.cif` | The 138 K neutron structure (Bacon, Curry & Wilson 1964; CSD BENZEN01, CCDC 1108750). Cacelli et al. started their simulations from this one. |
| `benzene_herringbone_cg.xyz` | The COD crystal coarse-grained to four ellipsoids (`c_q`, `or_vec`, `axes`). `HerringboneLatticeInitializer` tiles this motif by default, so its cell sets the starting density of every herringbone run. `scripts/build_herringbone_motif.py --cif data/benzene_pbca_cod_7238223.cif --out data/benzene_herringbone_cg.xyz` regenerates it, apart from the sign of each disc normal, which the potential doesn't care about. |
| `benzene_herringbone_cg_138K.xyz` | The same thing built from the 138 K cif, which is that script's default. |

## Potentials

| File | What it is |
|---|---|
| `lit_gbq_params.json` | Cacelli et al.'s GBQIII Gay-Berne + quadrupole parameters, loaded as `CACELLI_POTENTIAL`, the sampler's default. The file name (without `.json`) is the potential name recorded in every run, so please don't rename it. |

## Dimer reference data (used by the dimer benchmark)

| Path | What it is |
|---|---|
| `cacelli_2004_dimers/` | The supplementary material of Cacelli, Cinacchi, Prampolini & Tani, J. Chem. Phys. 120, 3648 (2004): 197 benzene dimer geometries with MP2/6-31G* interaction energies. `README.TXT` describes how the geometries are specified. |
| `uma_dimers/dimer_energies.csv` | UMA (`uma-s-1p1`) interaction energies at those 197 geometries, alongside the MP2 and GBQIII values. `dimer_benchmark` scores potentials against the UMA column. |
| `uma_dimers/family_curves.csv` | Finely spaced UMA and GBQIII scans through the cofacial, parallel-displaced and T-shaped wells, plotted in `notebooks/uma_vs_cacelli.ipynb`. |

Both `uma_dimers` files are written by `scripts/uma_cacelli_dimers.py`, which
reads the supplement's Euler angles as proper z-y-z angles (see
`docs/findings.md` §3). `tests/test_dimer_benchmark.py` checks that the tracked
files match the geometries the code builds.

## DFT benzene crystals (`benzene_crystals/`)

These come from the AniSOAP paper's Materials Cloud archive. There are 6,826
benzene crystal configurations, most with one or two molecules per cell, and
their energies were computed with Quantum ESPRESSO using PBE with Prandini et al.
cutoffs, Grimme D3 dispersion and a 3 × 3 Monkhorst–Pack k-point grid.

| File | What it is |
|---|---|
| `benzenes.xyz` | The atomistic configurations. Each frame has `energy` (eV), `energy_pa` (eV/atom) and a `signac_id` in `info`. |
| `ellipsoids.xyz` | The same configurations as ellipsoids, matched to `benzenes.xyz` by `signac_id`. `tests/test_coarse_graining.py` uses it as a reference. |
| `ellipsoids_with_axes_and_energies.xyz` | `ellipsoids_with_axes.xyz` with `energy` and `energy_pa` copied over frame by frame from `benzenes.xyz`. This is the training set for the GB+Q refit (`asmcmc.fitting_gbq`). |
| `ellipsoids_with_axes.xyz` | *Local only.* The ellipsoids, with their principal axes stored as arrays. |
| `duped_benzenes.xyz`, `duped_ellipsoids_with_axes.xyz` | *Local only* (3.6 GB). Every configuration replicated 7 × 7 × 7. `notebooks/cacelli_vs_dft.ipynb` and `notebooks/training_coverage_residual.ipynb` read these. |
