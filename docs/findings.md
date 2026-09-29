# Findings and design decisions

This file explains some things I've found while working on this project and
what conclusions I've drawn from them. The README covers how to use the code.
Energies are in kcal/mol (per dimer or per molecule) unless stated otherwise.

## 1. The model and the reference potential

Each benzene molecule is modelled as a rigid oblate ellipsoid, described by a
position and a disc normal (`or_vec`). Molecules interact through a Gay-Berne plus
point-quadrupole pair potential (`asmcmc.mc.potentials`). We use the parameters
that Cacelli et al. fitted to MP2 benzene-dimer energies, which they call GBQIII
(J. Chem. Phys. 120, 3648 (2004)). In the code this is `CACELLI_POTENTIAL`, and
it's the sampler's default.

Note that `data/lit_gbq_params.json` stores `kappa_prime = 1/4.39` rather than
the 4.39 listed in the paper's Table II. The paper defines the well-depth
anisotropy the other way round from the standard Gay-Berne form we use, and
1/4.39 is the value that reproduces their fitted dimer curves.

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
  278.7 K. I believe the paper is inconsistent with regard to this behavior: the text says GBQIII melts between 100 and 150 K, but Table IV and Fig. 5 say
  150–200 K.

### Why rotations are capped at 0.25 rad

Under GBQIII the herringbone crystal's orientational energy landscape is
shallow. If rotations aren't capped, the adaptive tuner grows the rotation
width to about 1.2 rad. The crystal then melts orientationally while the box is
still shrinking, and the run ends up as a dense glass. What works is to let
the box collapse quickly (`vol_delt = 0.025`) while keeping rotations small
(`max_or_delt = 0.25`), so the lattice densifies before the orientations have a
chance to disorder.

We later tried to copy Cacelli's protocol exactly (N = 500, fixed uncapped
widths, `or_delt = 0.6`). That melted the 150 K crystal even though the
orientation acceptance was right on target (34%), so a width can give the target
acceptance and still destroy the lattice. We abandoned that attempt. One
combination hasn't been tried yet: 150 K with fixed widths and `or_delt = 0.25`.

The somewhat ad-hoc equilibration procedure bothers me a little bit; I would
much rather not need to deliberately compress the simulation box to prevent
orientational melting, but I haven't been able to figure anything else out.

## 3. The dimer benchmark

A potential can fit condensed-phase energies well and still get the pair
interaction wrong. Each molecule's energy is a sum over many pairs, so errors
in individual pairs can cancel out. On top of that, the DFT training crystals
have only one or two molecules per cell. Every pair of a molecule with its own
periodic image is therefore exactly parallel, and the data says very little
about other orientations.

A previous attempt, `asmcmc.fitting_gbq`, refits GB+Q to the 6,826 DFT (PBE-D3)
crystal energies in `data/benzene_crystals/`. The refit
(`results/fitting/multiseed/uniform/seed_0`) matches those energies well, but
its cofacial (face-to-face) stack is repulsive: +2.6 kcal/mol where UMA gives
−2.0. The pairwise interaction inaccuracy is very consequential for MC because
the sampler moves only one particle at a time and must rely on that particl's
interactions with its neighbors to inform the decision of whether or not to
discard the trial move.

So before any candidate potential is used in MC, it probably should pass
`asmcmc.delta_learning.dimer_benchmark`. The benchmark scores the candidate on
the 197 dimer geometries published by Cacelli et al., which run through all
three wells (cofacial, parallel-displaced and T-shaped). The number that
matters is the RMSE over the attractive rows, which we call the well RMSE. T
This is because these geometries are much more representative of the ones
that the MC sampler actually visits. `improves_on_baseline`, is simply whether
the candidate's well RMSE is lower than GBQIII's. There's also an older check,
`stacking_bound`, which only asks whether the cofacial stack is bound at all.
It passes models that make the wells worse, so don't rank candidates on it.

### How the dimer geometries are read

The supplement gives molecule B's orientation as three Euler angles (α, β, γ) but
doesn't say which convention it uses. We read them as proper z-y-z angles
(`dimer_benchmark.EULER_SEQ = "ZYZ"`, scipy's intrinsic form). Of the 197 rows,
144 are pure translations with no angles, and z-y-z is the only standard reading
that fits all 53 rows that do have angles.

## 4. GBQIII prefers the wrong crystal

MC with Cacelli et al.'s potential finds the potential's minimum, but that
configuration is not the experimentally observed polymorph. If you scan the
lattice energy against volume, the slipped-parallel minimum (at 95.8 Å³ per
molecule) lies 1.28 kcal/mol below the herringbone minimum (at 109.1 Å³), or
about 1.1 below it after both structures are relaxed. The MC production volume,
96.5 Å³, sits right on the slipped-parallel minimum. Which polymorph is lower
depends on the volume, and the order flips near 120 Å³:

| V (Å³/molecule) | E_herringbone − E_slipped-parallel |
| --- | --- |
| 96.5 (MC) | +2.26 |
| 116.0 (experiment) | +0.18 |
| 123.6 (the COD cif's cell) | −0.14 |
| 193.5 (DFT training cells) | −0.46 |

So GBQIII gets the order right at the density of the ab initio training data
and wrong everywhere the MC actually goes. Fitting at that density can't catch
the problem.

This gives us a static test that doesn't need any MC: a better potential has to
put herringbone below slipped-parallel, each at its own minimum. See
`notebooks/polymorph_ordering.ipynb`.

## 5. UMA as the reference

UMA is Meta's machine-learned interatomic potential. We use `uma-s-1p1`, wrapped
in `asmcmc.delta_learning.uma`, both to label the training dimers and as the
benchmark's reference. It reproduces the dimers' MP2 energies well:

| Model vs MP2 | r (all 197) | RMSE (all) | r (157 attractive) | RMSE (attractive) |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- |
| cofacial | −2.02 at 3.90 Å | −1.81 at 3.93 Å |
| parallel-displaced | −3.07 at 3.94 Å | −2.49 at 3.99 Å |
| T-shaped | −2.81 at 5.00 Å | −1.94 at 5.08 Å |

The baseline to beat is GBQIII's well RMSE against UMA: **0.465**.

One worry I have about UMA is that it may not be a good enough ground truth,
which would mean that we would need to do ab initio calculations on a new
dataset more representative of realistic MC samples.

### UMA's 6 Å cutoff

UMA returns exactly zero interaction energy when no pair of atoms is within 6 Å.
Past that distance the label `Δ = −E_GBQ` is an artifact of UMA rather than
real physics. Our datasets therefore keep the closest atom–atom distance under
5.5 Å and leave the long-range tail to GB+Q, whose r⁻⁶ and r⁻⁵ tails have the
right form. Note that the cutoff applies to the closest *atom–atom* distance,
not to the distance between the molecular centres.

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
benchmark, though the weakest only by 0.004. The best point, `l9-n6-rc12`,
reaches a well RMSE of 0.359, 23% below GBQIII's 0.465. All three grid axes are
still improving at the edge of the grid, so we haven't found the optimum yet. Computational cost is another consideration and should be considered when we
decide which hyperparameters to go with. The correction hasn't yet been used in MC.

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
effect on the dimer benchmark, but it will matter in MC. So, a key question to
answer is how to implement AniSOAP into the MC simulation. Recalculating the
AniSOAP descriptor for the entire system at each time step would likely be far
too slow.

## 7. Next steps

1. Generate a new campaign with the current generator (the 3.0–5.5 Å window),
   widen the hyperparameter grid, and score on repeated splits.
2. Use the correction in MC. The sampler only supports pair potentials (see
   `Potential`), so a many-body AniSOAP energy will require some engineering.
   It's worth measuring the per-move cost of computing descriptors. After    that, run the static polymorph test (§4) and the 100 K herringbone protocol.
3. Run the (T, P) × potential sweep with the signac/CHTC setup on the
   `signac-flow-htc-impl` branch.

## Known issues

- Resuming a run is only reproducible if the run directory has a numeric name.
  `continue_equilibration` reseeds from the directory name, and a non-numeric
  name falls back to `hash()`, which changes between Python processes. The
  herringbone runs don't seed the sampler at all; their seed only sets the
  initial jitter.
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
- Sweep results are only bit-for-bit reproducible when run single-threaded. The
  parallel modules switch BLAS and OpenMP to a single thread before numpy loads,
  and that only works in a fresh process. A fit run inside a notebook, or in a
  test process that has already imported numpy, can differ in the last few digits.
- Most notebooks need local results. Only `uma_vs_cacelli.ipynb` runs from a
  fresh clone. The others read `results/` or the large `data/benzene_crystals`
  files, which aren't tracked.  Let me know if you'd like me to share these with you via Google Drive or email.
