# Findings and design decisions

This file explains why the code is the way it is: what has been measured, what
was decided because of it, and what is still open. The README covers how to use
the code. Numbers are in kcal/mol per dimer or per molecule unless stated.

## 1. The model and the reference potential

Each benzene molecule is a rigid oblate ellipsoid, a position plus a disc normal
(`or_vec`). Molecules interact through a Gay-Berne + point-quadrupole pair
potential (`asmcmc.mc.potentials`). The parameters are Cacelli et al.'s GBQIII
(J. Chem. Phys. 120, 3648 (2004)), fitted by them to MP2 benzene-dimer energies.
This is `CACELLI_POTENTIAL`, the sampler's default.

One trap: `data/lit_gbq_params.json` stores `kappa_prime = 1/4.39`, not Table II's
4.39. The paper's well-depth convention is inverted relative to the standard
Gay-Berne form used here, and 1/4.39 reproduces their fitted dimer curves; the
file carries a note saying so.

## 2. Validating the sampler against Cacelli et al.

`scripts/run_herringbone.py` runs the validation protocol: N = 400, 1 atm, starting
from the experimental herringbone crystal (`data/benzene_herringbone_cg.xyz`).
`docs/cacelli_protocol_diff.md` compares it with the paper's protocol line by line.

- **100 K reproduces their crystal.** The herringbone start converts to Cacelli's
  slipped-parallel crystal, as in their own runs: V/molecule 96.5 Å³ (theirs 96.8;
  experiment 116.0), density 1.344 g/cm³ (1.34), cell 8.75 / 6.15 / 7.17 Å
  (8.75 / 6.19 / 7.15). ΔH_vap is 10.86 against their ~10.2; 0.16 of that gap comes
  from their truncate-and-shift cutoff, and the rest is within their fit's error.
- **300 K reproduces their liquid.** Density is 1.13 g/cm³, within 1.8% of their
  GBQIII value, and both are 27–29% above experiment (0.877). The over-density is the
  potential's, not the sampler's.
- **GBQIII melts far too low.** Our 150 K and 273.15 K runs are liquid, which puts
  our melting point between 100 and 150 K. Real benzene melts at 278.7 K. The paper
  contradicts itself here: its text says GBQIII melts between 100 and 150 K, while
  its Table IV and Fig. 5 say 150–200 K.

**Why the protocol caps rotations (`max_or_delt = 0.25` rad).** Under GBQIII the
herringbone's orientational landscape is shallow, about 1 kT per site. Left
uncapped, the adaptive tuner grows the rotation width to ~1.2 rad. The crystal
then melts orientationally while the box is still collapsing, and ends as a dense
glass. The working recipe is to collapse the box fast (`vol_delt = 0.025`) while
capping rotations, so the lattice densifies before the orientations disorder.

A later attempt to match Cacelli's protocol exactly (N = 500, fixed and uncapped
widths, `or_delt = 0.6`) melted the 150 K crystal, even though orientation
acceptance was on target (34%). On-target acceptance does not mean a width
preserves the lattice. That attempt was abandoned (its code is on
`pre-revert-snapshot`). One experiment is still untested: 150 K at
`or_delt = 0.25` with fixed widths.

## 3. The physics gate: fit quality is not enough

A potential can fit condensed-phase energies well and still get the pair
interaction wrong. A per-molecule energy is a sum over many pairs, so pair errors
cancel in the fit target. The training crystals also have one or two molecules
per cell, so every periodic self-image pair is exactly parallel, and orientations
are barely constrained.

`asmcmc.fitting_gbq` refits GB+Q to the 6,826 DFT (PBE-D3) crystal energies in
`data/benzene_crystals/`. The refit (`results/fitting/multiseed/uniform/seed_0`)
fits those energies well, yet its cofacial stack is **repulsive** (+2.6 kcal/mol
where UMA gives -2.0). Its dimer-well errors are also anti-correlated with the
reference (r = -0.40 against MP2).

So every candidate potential must pass `asmcmc.delta_learning.dimer_benchmark`
before it is used in MC. The benchmark scores the 197 Cacelli dimer geometries
through all three wells (cofacial, parallel-displaced, T-shaped). Its verdict is
`improves_on_baseline`: does the candidate beat GBQIII's well RMSE? The older
`stacking_bound` check only asks whether the cofacial stack is bound at all, and
it passes models that make the wells worse.

**Reading the dimer geometries.** The supplement gives molecule B's orientation as
three Euler angles (α, β, γ) but does not name the convention. The code reads them
as proper z-y-z angles (`dimer_benchmark.EULER_SEQ = "ZYZ"`, scipy's intrinsic
form). This is the only standard reading consistent with all 53 rows that carry
angles; the other 144 rows are pure translations. The supplement's own energies
decide it:

- The four `(90, 90, 0)` rows along y repeat the energies of the `(0, 90, 90)`
  T-shaped rows along z (−2.27956 at 5 Å in both), so each pair is one dimer seen
  from two frames. Only proper Euler sequences (z-y-z, x-y-x, …) make them
  congruent.
- The `(0, 0, 90)` rows along z lie on a repulsive wall (+3.6 at 6.5 Å), like the
  in-plane rows at the same distance (+3.7). The rotation by γ must therefore turn
  B about the C–H bond it points at A, which keeps the head-on H···H contact.
- The `(90, 90, 90)` rows must not clash. Readings that put their atoms 1.3 Å
  apart, such as extrinsic z-y-x, give +75 in UMA where MP2 gives +25.

Rebuilding the 24-atom dimers under every scipy convention and scoring them with
UMA against MP2 gives the same answer. For scale, UMA's error on the 144
angle-free rows, where no convention is involved, is 0.37:

| UMA − MP2 RMSE (kcal/mol) | `"ZYZ"` | `"ZXY"` | `"zyx"` |
|---|---|---|---|
| `(0, 0, 90)`, 13 rows | 0.20 | 1.22 | 1.32 |
| `(90, 90, 90)`, 5 rows | 1.51 | 1.51 | 23.0 |
| `(90, 90, 0)`, 4 rows | 0.57 | 0.19 | 0.19 |
| all 53 angle-carrying rows | 0.64 | 0.86 | 7.12 |

The `(90, 90, 0)` line is not a counter-example. Under `"ZYZ"` those rows are the
T-shaped dimer, and they inherit UMA's T-shape error (0.55 on the T-shaped rows
themselves); the other readings build a different dimer that happens to sit closer
to MP2. GBQIII cannot settle the question: a uniaxial model misses the
`(0, 0, 90)` wall under every reading. `tests/test_dimer_benchmark.py` pins the
three geometric checks. Benchmark numbers computed before 28 September 2026 mixed
`"zyx"` geometry with UMA labels built under `"ZXY"`. Settling the reading moved
the baseline well RMSE from 0.436 to 0.465 and left the ranking of the 64 sweep
points essentially unchanged (rank correlation 0.98).

## 4. The wrong crystal: polymorph ordering

MC under GBQIII is not failing to equilibrate. It finds GBQIII's global minimum,
which is the wrong crystal. Scanning the lattice energy against volume puts the
slipped-parallel minimum 1.28 kcal/mol below the herringbone one (at 95.8 and
109.1 Å³/molecule; about 1.1 after relaxing both structures), and the MC
production density, 96.5 Å³, sits on the slipped-parallel minimum. The ordering is
volume-dependent and changes sign near 120 Å³:

| V (Å³/molecule) | E_herringbone − E_slipped-parallel |
|---|---|
| 96.5 (MC) | +2.26 |
| 116.0 (experiment) | +0.18 |
| 123.6 (the COD cif's cell) | −0.14 |
| 193.5 (DFT training cells) | −0.46 |

GBQIII therefore gets the ordering right at the density of the DFT data and wrong
everywhere MC goes, and fitting at that density cannot detect it.

This gives a static test that needs no MC: a better potential must put herringbone
below slipped-parallel at their respective minima. That is a directional
requirement, not an RMSE target. See `notebooks/polymorph_ordering.ipynb`.

## 5. UMA as the reference

Meta's UMA MLIP (`uma-s-1p1`, `asmcmc.delta_learning.uma`) labels the training dimers
and is the benchmark's reference. It was never trained on the Cacelli dimers, yet
it reproduces their MP2 energies well:

| Model vs MP2 | r (all 197) | RMSE (all) | r (157 attractive) | RMSE (attractive) |
|---|---|---|---|---|
| UMA | 0.996 | 0.46 | 0.996 | 0.27 |
| GBQIII | 0.915 | 1.67 | 0.971 | 0.21 |

MP2 cannot be the target, because GBQIII was fitted to those rows: on the
attractive rows GBQIII beats UMA against MP2 (0.21 vs 0.27), so a correction that
moved GBQIII towards UMA would score as worse. The MP2 energies stay loadable as a
diagnostic (`reference="mp2"`).

Against UMA, every GBQIII well is too shallow, so a correct Δ deepens all three
(`tests/test_dimer_benchmark.py` pins this):

| Well | UMA | GBQIII |
|---|---|---|
| cofacial | −2.02 at 3.90 Å | −1.81 at 3.93 Å |
| parallel-displaced | −3.07 at 3.94 Å | −2.49 at 3.99 Å |
| T-shaped | −2.81 at 5.00 Å | −1.94 at 5.08 Å |

The baseline to beat is GBQIII's well RMSE against UMA: **0.465**.

**UMA's 6 Å horizon.** UMA returns exactly zero interaction when no atom pair is
within 6 Å (its graph cutoff), because the two molecules are then disconnected
graph components. Raising the cutoff at inference gives nonsense. Past that
distance, `Δ = −E_GBQ` is an artefact of the labeller, so datasets stay inside a
5.5 Å closest-atom distance and leave the tail to GB+Q, whose r⁻⁶/r⁻⁵ asymptotics
have the right form. The horizon is on the closest *atom–atom* distance, not the
centre–centre distance.

## 6. The Δ-learning correction

`asmcmc.delta_learning` fits `Δ = E_UMA − E_GBQ` by ridge regression on AniSOAP
power-spectrum descriptors of the coarse-grained dimers. The choices that are
physics, not statistics (see `model.py`'s docstring):

- **no intercept and no feature centring**: a pair with nothing inside the cutoff
  must get exactly zero Δ;
- **descriptors summed over centres**: the energy is extensive;
- **the ridge penalty chosen per hyperparameter point**: feature counts run from
  64 to 490.

**Results so far** (the 64-point sweep over `max_angular` × `max_radial` ×
`cutoff_radius`, on `results/cluster_train`): 64/64 points improve on GBQIII. The
best, `l9-n6-rc12`, reaches a well RMSE of **0.359** (−23%). All three grid axes
are still improving at the grid's edge, so the optimum has not been found. The
correction is not yet used in MC.

**What made campaigns fail.** Two later campaigns engineered the mix of contact
motifs and failed on every point (0/64). The cause was not the composition. It was
how much of the squared-error objective sat on hard-core clashes (closest atoms at
2.4–2.7 Å, strongly repulsive, never visited by MC, which stays beyond ~3.6 Å):
41% in the successful campaign against ~75% in the failed ones. The dataset
generator therefore now samples uniformly and filters on a 3.0–5.5 Å closest-atom
window.

**Only dimers.** Across 10,545 trimers, even the compact ones carry an rms three-body
energy of 0.048, about 10× below the fit error, so trimers are no longer generated.

**Scoring caveats.**

- `test_skill` is normalised by each campaign's own null RMSE, so compare absolute
  RMSEs across campaigns instead.
- The sweeps all used `split_seed = 0`, which is the worst of six seeds on the
  close-contact subset. Score on repeated splits.
- Inner-CV `RidgeCV` under-regularises at close contact: it picks α = 0.1 where the
  test optimum is ≥ 1.
- Close contact is limited by data, not by model size: adding features barely helps
  there.

**Open question.** The model is deployed pairwise, but the AniSOAP power spectrum
is not additive over pairs: a trimer's descriptor is not the sum of its three pair
descriptors. This is worth ~0.09 rms in predicted Δ. It is zero on the dimer
benchmark, but it matters for MC.

## 7. Next steps

1. Generate a new campaign with the current generator (3.0–5.5 Å window), widen the
   hyperparameter grid, and score on repeated splits.
2. MC integration: the sampler only supports pair potentials (see `Potential`). A
   many-body AniSOAP energy needs a per-particle energy hook. Measure the per-move
   descriptor cost early, then apply the static polymorph test (§4) and the 100 K
   herringbone protocol.
3. The (T, P) × potential sweep, using the signac/CHTC harness on
   `signac-flow-htc-impl`.

## Known issues

- **Resuming is only reproducible for numeric run-directory names.**
  `continue_equilibration` reseeds from the directory name; a non-numeric name
  falls back to `hash()`, which varies between Python processes. The herringbone
  runs do not seed the sampler at all: their seed only sets the initial jitter.
- **The 300 K run on disk used seed 12321**, per its `run_config.json`, while
  `run_herringbone.py` (like the script it replaced) says 123219.
- **Provenance names.** Four of the five herringbone run configs record the
  potential as `"data"`, an older naming rule. The parameters are Cacelli's.
- **`signac-flow-htc-impl`** passes pressure in atm to a sampler that expects eV/Å³
  (use `asmcmc.units.ATM_TO_EV_PER_A3`), and its imports predate the current
  package layout.
- **`AverageEnergy.finalize` returns (mean, variance)**, while `AverageEnthalpy`
  returns (mean, std).
- **Old campaigns can't be regenerated.** The `results/cluster_*` campaigns,
  including the winning `cluster_train`, were made by earlier versions of the
  generator and can't be reproduced by the current code.
- **Pre-`xi` fits.** The tracked GB+Q fits in `results/fitting/` predate the `xi`
  parameter; `GBQPotential.from_json` loads them with `xi = 1`, the form they were
  fitted with.
- **Sweep results are bitwise reproducible only single-threaded.** The parallel
  modules set single-threaded BLAS/OpenMP before numpy loads, which only takes
  effect in a fresh process. Fits run inside a notebook or test process that
  already loaded numpy can differ in the last digits.
- **Most notebooks need local results.** Only `uma_vs_cacelli.ipynb` runs from a
  fresh clone; the others read `results/` or the large `data/benzene_crystals`
  files, which are local only.
