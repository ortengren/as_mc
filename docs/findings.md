# Findings and design decisions

This file is the "why" behind the code: what we've measured, what we decided
because of it, and what's still open. The README covers how to use the code.
Energies are in kcal/mol (per dimer or per molecule) unless stated otherwise.

## 1. The model and the reference potential

Each benzene molecule is modelled as a rigid oblate ellipsoid, described by a
position and a disc normal (`or_vec`). Molecules interact through a Gay-Berne plus
point-quadrupole pair potential (`asmcmc.mc.potentials`). We use the parameters
that Cacelli et al. fitted to MP2 benzene-dimer energies, which they call GBQIII
(J. Chem. Phys. 120, 3648 (2004)). In the code this is `CACELLI_POTENTIAL`, and
it's the sampler's default.

One thing to watch out for: `data/lit_gbq_params.json` stores
`kappa_prime = 1/4.39` rather than the 4.39 listed in the paper's Table II. The
paper defines the well-depth anisotropy the other way round from the standard
Gay-Berne form we use, and 1/4.39 is the value that reproduces their fitted dimer
curves. The file has a note explaining this.

## 2. Validating the sampler against Cacelli et al. (2004)

`scripts/run_herringbone.py` runs our validation protocol: N = 400 molecules at
1 atm, starting from the experimental herringbone crystal
(`data/benzene_herringbone_cg.xyz`). `docs/cacelli_protocol_diff.md` compares it
with the paper's protocol point by point. The main results are:

- At 100 K we reproduce their crystal. The herringbone start converts to the
  slipped-parallel crystal that Cacelli et al. also found. Our volume is 96.5 Å³
  per molecule (theirs 96.8, experiment 116.0), our density 1.344 g/cm³ (theirs
  1.34), and our cell edges 8.75, 6.15 and 7.17 Å (theirs 8.75, 6.19 and 7.15).
  Our ΔH_vap is 10.86 against their ~10.2. Their truncate-and-shift cutoff
  accounts for 0.16 of that gap, and the rest is within the error of their fit.
- At 300 K we reproduce their liquid. Our density of 1.13 g/cm³ is within 1.8%
  of their GBQIII value. Both are 27–29% above experiment (0.877), so the excess
  density comes from the potential rather than the sampler.
- GBQIII melts far too low. Our 150 K and 273.15 K runs are both liquid, which
  puts its melting point somewhere between 100 and 150 K. Real benzene melts at
  278.7 K. The paper is inconsistent on this point: the text says GBQIII melts
  between 100 and 150 K, but Table IV and Fig. 5 say 150–200 K.

### Why rotations are capped at 0.25 rad

Under GBQIII the herringbone crystal's orientational energy landscape is shallow,
about 1 kT per site. If rotations aren't capped, the adaptive tuner grows the
rotation width to about 1.2 rad. The crystal then melts orientationally while the
box is still shrinking, and the run ends up as a dense glass. What works is to let
the box collapse quickly (`vol_delt = 0.025`) while keeping rotations small
(`max_or_delt = 0.25`), so the lattice densifies before the orientations have a
chance to disorder.

We later tried to copy Cacelli's protocol exactly (N = 500, fixed uncapped
widths, `or_delt = 0.6`). That melted the 150 K crystal even though the
orientation acceptance was right on target (34%), so a width can give the target
acceptance and still destroy the lattice. We abandoned that attempt. One
combination hasn't been tried yet: 150 K with fixed widths and `or_delt = 0.25`.

## 3. The dimer benchmark

A potential can fit condensed-phase energies well and still get the pair
interaction wrong. Each molecule's energy is a sum over many pairs, so errors in
individual pairs can cancel out. On top of that, the DFT training crystals have
only one or two molecules per cell. Every pair of a molecule with its own periodic
image is therefore exactly parallel, and the data says very little about other
orientations.

We've seen this happen. `asmcmc.fitting_gbq` refits GB+Q to the 6,826 DFT
(PBE-D3) crystal energies in `data/benzene_crystals/`. The refit
(`results/fitting/multiseed/uniform/seed_0`) matches those energies well, but its
cofacial (face-to-face) stack is repulsive: +2.6 kcal/mol where UMA gives −2.0.
Its errors in the dimer wells are even anti-correlated with the reference
(r = −0.40 against MP2).

So before any candidate potential is used in MC, it has to pass
`asmcmc.delta_learning.dimer_benchmark`. The benchmark scores the candidate on
the 197 dimer geometries published by Cacelli et al., which run through all three
wells (cofacial, parallel-displaced and T-shaped). The number that matters is the
RMSE over the attractive rows, which we call the well RMSE. The verdict,
`improves_on_baseline`, is simply whether the candidate's well RMSE is lower than
GBQIII's. There's also an older check, `stacking_bound`, which only asks whether
the cofacial stack is bound at all. It passes models that make the wells worse,
so don't rank candidates on it.

### How the dimer geometries are read

The supplement gives molecule B's orientation as three Euler angles (α, β, γ) but
doesn't say which convention it uses. We read them as proper z-y-z angles
(`dimer_benchmark.EULER_SEQ = "ZYZ"`, scipy's intrinsic form). Of the 197 rows,
144 are pure translations with no angles, and z-y-z is the only standard reading
that fits all 53 rows that do have angles. The supplement's own energies settle
the question:

- The four `(90, 90, 0)` rows along y have the same energies as the `(0, 90, 90)`
  T-shaped rows along z (−2.27956 at 5 Å in both). Each pair must be the same
  dimer seen from two different frames, and only proper Euler sequences (z-y-z,
  x-y-x, …) make the two geometries congruent.
- The `(0, 0, 90)` rows along z sit on a repulsive wall (+3.6 at 6.5 Å), just
  like the in-plane rows at the same distance (+3.7). So the γ rotation has to
  spin B about the C–H bond that points at A, which keeps the head-on H···H
  contact.
- The `(90, 90, 90)` rows can't have atoms on top of each other. Readings such as
  extrinsic z-y-x put atoms 1.3 Å apart in those rows, and UMA then gives +75
  where MP2 gives +25.

Rebuilding the 24-atom dimers under every scipy convention and scoring them with
UMA against MP2 points the same way. For comparison, UMA's error on the 144
angle-free rows, where the convention doesn't matter, is 0.37:

| UMA − MP2 RMSE (kcal/mol) | `"ZYZ"` | `"ZXY"` | `"zyx"` |
|---|---|---|---|
| `(0, 0, 90)`, 13 rows | 0.20 | 1.22 | 1.32 |
| `(90, 90, 90)`, 5 rows | 1.51 | 1.51 | 23.0 |
| `(90, 90, 0)`, 4 rows | 0.57 | 0.19 | 0.19 |
| all 53 rows with angles | 0.64 | 0.86 | 7.12 |

The `(90, 90, 0)` line doesn't contradict this. Under `"ZYZ"` those rows are the
T-shaped dimer, so they inherit UMA's error on T-shapes (0.55 on the T-shaped rows
themselves). The other readings build a different dimer that happens to land
closer to MP2. GBQIII can't help decide, because as a uniaxial model it misses the
`(0, 0, 90)` wall under every reading. `tests/test_dimer_benchmark.py` checks all
three of the geometric arguments above.

Benchmark numbers from before 28 September 2026 were computed on `"zyx"`
geometries but scored against UMA energies calculated under `"ZXY"`. Fixing this
raised the baseline well RMSE from 0.436 to 0.465 and hardly changed the ranking
of the 64 sweep points (rank correlation 0.98).

## 4. GBQIII prefers the wrong crystal

MC under GBQIII isn't failing to equilibrate. It finds GBQIII's global minimum,
and that minimum is the wrong crystal. If you scan the lattice energy against
volume, the slipped-parallel minimum (at 95.8 Å³ per molecule) lies 1.28 kcal/mol
below the herringbone minimum (at 109.1 Å³), or about 1.1 below it after both
structures are relaxed. The MC production volume, 96.5 Å³, sits right on the
slipped-parallel minimum. Which polymorph is lower depends on the volume, and the
order flips near 120 Å³:

| V (Å³/molecule) | E_herringbone − E_slipped-parallel |
|---|---|
| 96.5 (MC) | +2.26 |
| 116.0 (experiment) | +0.18 |
| 123.6 (the COD cif's cell) | −0.14 |
| 193.5 (DFT training cells) | −0.46 |

So GBQIII gets the order right at the density of the DFT training data and wrong
everywhere the MC actually goes. Fitting at that density can't catch the problem.

This gives us a static test that doesn't need any MC: a better potential has to
put herringbone below slipped-parallel, each at its own minimum. What matters is
which structure comes out lower, rather than hitting an RMSE target. See
`notebooks/polymorph_ordering.ipynb`.

## 5. UMA as the reference

UMA is Meta's machine-learned interatomic potential. We use `uma-s-1p1`, wrapped
in `asmcmc.delta_learning.uma`, both to label the training dimers and as the
benchmark's reference. It was never trained on the Cacelli dimers, but it
reproduces their MP2 energies well:

| Model vs MP2 | r (all 197) | RMSE (all) | r (157 attractive) | RMSE (attractive) |
|---|---|---|---|---|
| UMA | 0.996 | 0.46 | 0.996 | 0.27 |
| GBQIII | 0.915 | 1.67 | 0.971 | 0.21 |

We can't use MP2 as the target, because GBQIII was fitted to those same rows. On
the attractive rows GBQIII actually matches MP2 better than UMA does (0.21 vs
0.27), so a correction that moved GBQIII towards UMA would look as if it were
making things worse. The MP2 energies can still be loaded as a diagnostic
(`reference="mp2"`).

Compared with UMA, all three GBQIII wells are too shallow, so a correct Δ should
make all three deeper. `tests/test_dimer_benchmark.py` checks this.

| Well | UMA | GBQIII |
|---|---|---|
| cofacial | −2.02 at 3.90 Å | −1.81 at 3.93 Å |
| parallel-displaced | −3.07 at 3.94 Å | −2.49 at 3.99 Å |
| T-shaped | −2.81 at 5.00 Å | −1.94 at 5.08 Å |

The baseline to beat is GBQIII's well RMSE against UMA: **0.465**.

### UMA's 6 Å cutoff

UMA returns exactly zero interaction energy when no pair of atoms is within 6 Å.
That's its graph cutoff: beyond it the two molecules are disconnected in UMA's
graph, and raising the cutoff at inference time gives nonsense. Past that
distance the label `Δ = −E_GBQ` is an artefact of UMA rather than real physics.
Our datasets therefore keep the closest atom–atom distance under 5.5 Å and leave
the long-range tail to GB+Q, whose r⁻⁶ and r⁻⁵ tails have the right form. Note
that the cutoff applies to the closest *atom–atom* distance, not to the distance
between the molecular centres.

## 6. The Δ-learning correction

`asmcmc.delta_learning` fits `Δ = E_UMA − E_GBQ` by ridge regression on AniSOAP
power-spectrum descriptors of the coarse-grained dimers. A few of the design
choices come from the physics rather than the statistics (`model.py`'s docstring
has the details):

- There's no intercept and no feature centring, because a pair with nothing
  inside the cutoff must get exactly zero Δ.
- Descriptors are summed over centres rather than averaged, because energy is
  extensive.
- The ridge penalty is chosen separately at each hyperparameter point, because
  the number of features ranges from 64 to 490 across the grid.

### Results so far

The current sweep covers 64 points (`max_angular` × `max_radial` ×
`cutoff_radius`), trained on `results/cluster_train`. All 64 beat GBQIII on the
benchmark, though the weakest only by 0.004. The best point, `l9-n6-rc12`, reaches
a well RMSE of **0.359**, 23% below GBQIII's 0.465. All three grid axes are still
improving at the edge of the grid, so we haven't found the optimum yet. The
correction hasn't been used in MC.

### Why two campaigns failed

Two later campaigns tried to engineer the mix of contact motifs, and every point
in both failed the benchmark (0/64). The motif mix wasn't the problem. What
mattered was how much of the squared-error objective came from hard-core clashes:
geometries whose closest atoms are 2.4–2.7 Å apart, which are strongly repulsive
and never visited by MC (it stays beyond about 3.6 Å). Clashes made up 41% of the
objective in the successful campaign and about 75% in the failed ones. The dataset
generator now samples geometries uniformly and keeps only those whose closest
atoms are between 3.0 and 5.5 Å apart.

### Dimers only

Across 10,545 trimers, even the compact ones carry an rms three-body energy of
only 0.048, about ten times smaller than the fit error. So we no longer generate
trimers.

### Things to keep in mind when comparing scores

- `test_skill` is normalised by each campaign's own null RMSE, so it can't be
  compared between campaigns. Compare absolute RMSEs instead.
- All sweeps so far used `split_seed = 0`, which turned out to be the worst of six
  seeds on the close-contact subset. Future scoring should use repeated splits.
- `RidgeCV`'s inner cross-validation under-regularises at close contact. It picks
  α = 0.1, while the best α on the test set is 1 or more.
- At close contact the model is limited by the data rather than by its size.
  Adding features barely helps there.

### Open question

The model is deployed as a pair potential, but the AniSOAP power spectrum isn't
additive over pairs: a trimer's descriptor isn't the sum of its three pair
descriptors. The difference is worth about 0.09 rms in predicted Δ. It has no
effect on the dimer benchmark, but it will matter in MC.

## 7. Next steps

1. Generate a new campaign with the current generator (the 3.0–5.5 Å window),
   widen the hyperparameter grid, and score on repeated splits.
2. Use the correction in MC. The sampler only supports pair potentials (see
   `Potential`), so a many-body AniSOAP energy needs a per-particle energy hook.
   It's worth measuring the per-move cost of computing descriptors early on. After
   that, run the static polymorph test (§4) and the 100 K herringbone protocol.
3. Run the (T, P) × potential sweep with the signac/CHTC setup on the
   `signac-flow-htc-impl` branch.

## Known issues

- Resuming a run is only reproducible if the run directory has a numeric name.
  `continue_equilibration` reseeds from the directory name, and a non-numeric name
  falls back to `hash()`, which changes between Python processes. The herringbone
  runs don't seed the sampler at all; their seed only sets the initial jitter.
- According to its `run_config.json`, the 300 K run on disk used seed 12321, but
  `run_herringbone.py` (like the script it replaced) uses 123219.
- Four of the five herringbone run configs record the potential's name as
  `"data"`, from an older naming rule. The parameters are Cacelli's.
- The `signac-flow-htc-impl` branch passes pressure in atm, but the sampler
  expects eV/Å³ (convert with `asmcmc.units.ATM_TO_EV_PER_A3`). Its imports also
  predate the current package layout.
- `AverageEnergy.finalize` returns the mean and the variance, whereas
  `AverageEnthalpy` returns the mean and the standard deviation.
- The `results/cluster_*` campaigns, including `cluster_train`, which gave the
  best results, were made by earlier versions of the generator and can't be
  reproduced with the current code.
- The GB+Q fits tracked in `results/fitting/` predate the `xi` parameter.
  `GBQPotential.from_json` loads them with `xi = 1`, which is the form they were
  fitted with.
- Sweep results are only bit-for-bit reproducible when run single-threaded. The
  parallel modules switch BLAS and OpenMP to a single thread before numpy loads,
  and that only works in a fresh process. A fit run inside a notebook, or in a
  test process that has already imported numpy, can differ in the last few digits.
- Most notebooks need local results. Only `uma_vs_cacelli.ipynb` runs from a fresh
  clone. The others read `results/` or the large `data/benzene_crystals` files,
  which aren't tracked.
