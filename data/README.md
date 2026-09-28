# data/

Inputs the code reads, resolved through `asmcmc.paths.data_path` (override the
location with `ASMCMC_DATA_DIR`). Everything listed here is tracked except the
files marked *local only*, which are large and gitignored.

## Crystal structures and the MC starting motif

| File | What it is |
|---|---|
| `benzene_pbca_cod_7238223.cif` | Experimental Pbca benzene crystal, COD 7238223 (in-situ cryo-grown, reported at 150 K). Its cell is ~4% larger than benzene at that temperature: 123.58 Å³/molecule, ρ = 1.050 g/cm³ against ~1.09–1.10. |
| `benzene_Pbca_csd_1108750.cif` | The 138 K neutron structure (Bacon, Curry & Wilson 1964; CSD BENZEN01, CCDC 1108750), the starting structure Cacelli et al. used. |
| `benzene_herringbone_cg.xyz` | The COD crystal coarse-grained to four ellipsoids (`c_q`, `or_vec`, `axes`). This is the default motif `HerringboneLatticeInitializer` tiles, so its cell sets the starting density of every herringbone run. `scripts/build_herringbone_motif.py --cif data/benzene_pbca_cod_7238223.cif --out data/benzene_herringbone_cg.xyz` regenerates it up to the sign of each disc normal, which the potential ignores. |
| `benzene_herringbone_cg_138K.xyz` | The same, built from the 138 K cif (that script's default). |

## Potentials

| File | What it is |
|---|---|
| `lit_gbq_params.json` | Cacelli et al.'s GBQIII Gay-Berne + quadrupole parameters: `CACELLI_POTENTIAL`, the sampler's default. Its file stem is the potential name recorded in every run, so do not rename it. |

## Dimer reference data (the physics gate)

| Path | What it is |
|---|---|
| `cacelli_2004_dimers/` | Supplementary material of Cacelli, Cinacchi, Prampolini & Tani, J. Chem. Phys. 120, 3648 (2004): 197 benzene dimer geometries with MP2/6-31G* interaction energies. `README.TXT` gives the geometry convention. |
| `uma_dimers/dimer_energies.csv` | UMA (`uma-s-1p1`) interaction energies at those 197 geometries, plus the MP2 and GBQIII values: the reference `dimer_benchmark` scores against. |
| `uma_dimers/family_curves.csv` | Dense UMA and GBQIII scans through the cofacial, parallel-displaced and T-shaped wells, plotted by `notebooks/uma_vs_cacelli.ipynb`. |

Both `uma_dimers` files are written by `scripts/uma_cacelli_dimers.py`, which
reads the supplement's Euler angles as proper z-y-z angles (`docs/findings.md`
§3). `tests/test_dimer_benchmark.py` checks that the tracked files match the
geometry the code builds.

## DFT benzene crystals (`benzene_crystals/`)

From the Materials Cloud archive of the AniSOAP paper: 6,826 benzene crystal
configurations (mostly one or two molecules per cell) with energies from Quantum
ESPRESSO, using PBE with Prandini et al. cutoffs, Grimme D3 dispersion and a
3 × 3 Monkhorst–Pack k-point grid.

| File | What it is |
|---|---|
| `benzenes.xyz` | The atomistic configurations; `energy` (eV) and `energy_pa` (eV/atom) in `info`, and a `signac_id` per frame. |
| `ellipsoids.xyz` | The same configurations as ellipsoids, matched to `benzenes.xyz` by `signac_id`. Used as the reference in `tests/test_coarse_graining.py`. |
| `ellipsoids_with_axes_and_energies.xyz` | `ellipsoids_with_axes.xyz` with `energy`/`energy_pa` copied frame by frame from `benzenes.xyz`. The training set of the GB+Q refit (`asmcmc.fitting_gbq`). |
| `ellipsoids_with_axes.xyz` | *Local only.* The ellipsoids with principal-axis arrays. |
| `duped_benzenes.xyz`, `duped_ellipsoids_with_axes.xyz` | *Local only* (3.6 GB). Every configuration replicated 7 × 7 × 7, read by `notebooks/cacelli_vs_dft.ipynb` and `notebooks/training_coverage_residual.ipynb`. |
