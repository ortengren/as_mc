# Protocol differences: asmcmc vs Cacelli et al. (2004)

Reference: I. Cacelli, G. Cinacchi, G. Prampolini, A. Tani, *Modeling benzene with
single-site potentials from ab initio calculations: a step toward hybrid models of
complex molecules*, J. Chem. Phys. **120**, 3648 (2004). PDF in `literature/`
(gitignored). Page numbers below are the journal's (3648–3657).

Runs described here (all `CACELLI_POTENTIAL`, all 1 atm, driver
`scripts/single_state_run_hb_1atm.py` / `..._300K.py`):

| Run dir | T | N | steps (eq + prod) |
|---|---|---|---|
| `results/validation/100.0_6.324209e-07/herringbone` | 100 K | 400 | 10.2M + 15M |
| `results/validation/100.0_6.324209e-07/herringbone_jittered` | 100 K | 400 | 10.2M + 15M |
| `results/validation/100.0_6.324209e-07/herringbone_jittered_2` | 100 K | 400 | 10.2M + 15M |
| `results/validation/300.0_6.324209e-07/herringbone_jittered_0` | 300 K | 400 | 10.2M + 15M |

**Bottom line.** Structural observables agree with their Table V to <1% (density
0.3%, lattice constants <1%). The energetics differ by 6% (our ΔH_vap 10.86 vs
their 10.2 kcal/mol), of which 0.16 kcal/mol is the cutoff convention (§1) and
the remainder is within the GBQIII fit's own 0.5 kcal/mol dimer rms. No difference
in this document is known to invalidate the comparison; the ones that are not
quantified are listed in §5.

**Status flags.** `✓` matched · `⚠` differs, effect quantified · `✗` differs,
effect **not** quantified · `?` not stated in the paper.

---

## 1. Pair potential and its evaluation

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ ε₀, σ₀, κ, μ, ν, ξ | 0.800 kcal/mol, 5.720 Å, 0.542, −10.0, 0.41, 0.73 | identical (ε₀ = 0.0347 eV = 0.800 kcal/mol) | none | Table II; `data/lit_gbq_params.json` |
| ✓ Quadrupole Q | −4.130·10⁻²⁶ esu·cm² | −3.263 (eV·Å⁵)^½ — the same value converted (Q²[eV·Å⁵] = Q²[esu²cm⁴] × 6.2415·10⁵¹) | none | Table II; `potentials.py:59` |
| ⚠ κ′ | 4.39 (as listed) | 0.2278 = 1/4.39 | Using the listed 4.39 under the standard GB convention χ′=(κ′^(1/μ)−1)/(κ′^(1/μ)+1) makes face-involved wells 4.4× **too shallow** and does not reproduce their Fig 3; the inverted value does (χ′(1/κ′) = −χ′(κ′) identically). Resulting wells: FF −1.81 @3.93 Å, SP −1.99, TS −1.94 @5.08 Å, SS −0.80, CR −0.78 kcal/mol — on or slightly *shallower* than their Fig 3 GBQIII curves, i.e. not a source of excess binding | Table II, Fig 3; `potentials.py:33` |
| ⚠ Cutoff scheme | truncate **and shift** at 15 Å | plain truncation at an effective 13.6 Å (`nl_radius=6.8` is a *per-atom* radius and ASE sums the pair's two radii) | **+0.16 kcal/mol on ΔH_vap, ours high.** On the final 100 K frame: 147 neighbours/molecule inside 15 Å, ⟨u(15 Å)⟩ = −0.139 meV/pair, so their shift removes 0.236 kcal/mol of binding while our missing 13.6–15 Å tail costs 0.077. U/N: ours −10.665, theirs' convention −10.506 kcal/mol | p. 3653; `potentials.py:99`, `metropolis.py:196` |
| ✓ Electrostatics | quadrupole–quadrupole truncated at the same cutoff, no Ewald | same | none | p. 3653; `potentials.py:59` |
| ✓ Pair enumeration | N² all-pairs | ASE `neighbor_list` (1.0 Å skin) | same physics, different cost | p. 3653; `potentials.py:116` |
| — Working units | kcal/mol, Å | eV, Å (×23.0605 to compare) | bookkeeping only | — |

## 2. Ensemble and trial moves

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ Ensemble | MC NPT, P = 1 atm | MC NPT, P = 6.324209·10⁻⁷ eV/Å³ = 1 atm | none | p. 3653; `metropolis.py` |
| ✓ Volume-move geometry | one randomly chosen box edge is stretched (orthorhombic shape relaxes) | `aniso_vol=True`: one randomly chosen axis rescaled per move | none — all four runs above use `aniso_vol=True`. Earlier isotropic-only runs (in `results/archive/`) could not relax the a:b:c ratio and produced a box-templated texture | p. 3653; `trial_moves.py` |
| ✓ Box shape freedom | axis lengths only; box stays orthorhombic | same — no shear/tilt degree of freedom | neither can find a monoclinic/triclinic cell | p. 3653; `trial_moves.py` |
| ✓ Delta tuning during production | maximum displacements adjusted to ~30% acceptance, then production | tuned during equilibration only; `calculate_trajectory` calls `block_update(dynamic_delta=False)` | none — both sample with fixed proposal widths, so detailed balance holds in production | p. 3653; `metropolis.py:752` |
| ⚠ Acceptance target | ~30% for all move kinds | `TARGET_ACC_RATE = 0.275` | efficiency only, not the equilibrium distribution | p. 3653; `metropolis.py:24` |
| ? Move mix | "all kinds of moves"; mix not stated (a "cycle" is presumably N attempts) | per step: P(translation) = P(rotation) = (N−1)/2N, P(volume) = 1/N — i.e. one volume attempt per sweep | assumed equivalent; unverifiable from the paper | p. 3653; `metropolis.py:299` |
| ⚠ Close-contact / overlap rejection | moves sampling r < r_asy (≤ σ − σ₀ξ) are **refused**, to avoid nonphysical regions of the GB surface | **no overlap test and no r_asy guard** — rejection is left entirely to the Boltzmann factor | Harmless at these densities, verified numerically: with the GBQIII parameters the core stays strongly repulsive on *both* sides of the r = σ(r̂) − ξσ₀ pole (edge-edge pole at 1.54 Å, U ≥ 10⁶ kcal/mol at every r < 2 Å; face-to-face and T-shaped have no pole at r > 0), so overlaps are rejected with probability ~1. Closest contact in our crystal is 4.4 Å. Costs acceptance efficiency at low density / high T, where proposals can land in the core; a proposal landing exactly on the pole yields `inf` and is rejected | p. 3653; `metropolis.py:293` |

## 3. System and run protocol

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ⚠ Particle count N | **500** — the 4-molecule experimental cell replicated 5×5×5 | **400** — same motif, reps (5, 5, 4) | Finite-size difference of 20%. Consequence for the box: their converged cell (Table V) gives a 43.75 × 30.95 × 35.75 Å box, shortest edge 30.95 Å ≥ 2 r_c = 30 Å — they satisfy the conventional L > 2 r_c criterion (just). Our mean box is 35.88 × 43.75 × **24.60** Å, shortest edge < 2 r_c = 27.2 Å, so along that axis a molecule can interact with two periodic images of the same neighbour. The energy is still the correct truncated lattice sum (ASE sums all images within the cutoff; the shortest lattice vector 24.6 Å exceeds the 13.6 Å cutoff, so there are no self-image terms), but the configuration is more correlated with its own images than theirs | p. 3653, Table V; `initialize.py` HerringboneLatticeInitializer |
| ⚠ Starting configuration | experimental crystal structure at 138 K, four molecules per unit cell; **every** simulation restarts from it at each T | same experimental Pbca motif via `HerringboneLatticeInitializer`, with positional/orientational jitter (0.0 / 0.1 / 0.15 across the three replicas) | Same starting polymorph. The jitter exists to make replicas independent; it is not in their protocol | p. 3653; `scripts/single_state_run_hb_1atm.py` |
| ✗ Equilibration protocol | not described beyond 50 000 cycles at the target T | **compress-before-melt**: `vol_delt = 0.025` (fast box collapse) + `max_or_delt = 0.25 rad` (capped rotations). Needed because the uncapped adaptive tuner walks or_delt to ~1.2 rad in the shallow herringbone orientational landscape and melts the crystal before it densifies, giving a glass | Not quantifiable against the paper — they do not report a protocol at this level of detail. Their reported 30%-acceptance volume amplitude is ~10× ours, i.e. their collapse also beat their orientational melt | p. 3653; `scripts/single_state_run_hb_1atm.py` |
| ⚠ Run length | 50 000 cycles equilibration + 100 000 cycles production at N = 500 | 10.2M + 15M single-particle steps at N = 400 = 25 500 + 37 500 sweeps | They have ~2.7× more production sweeps per particle. Our production is converged by the half-split test: volume drifts −0.01% (100 K) and −0.08% (300 K) between halves | p. 3653; `notebooks/herringbone_runs.ipynb` (convergence cell) |
| ⚠ State points | 100–400 K in 50 K steps, all at 1 atm | 100 K and 300 K at 1 atm | We have no melting curve; their GBQIII melts between 100–150 K (p. 3654 text) or 150–200 K (Table IV — the paper is internally inconsistent here). Our 300 K run is liquid, consistent with either | Fig 5, Table IV |
| ⚠ Replicas | 1 (not stated otherwise) | 3 at 100 K, 1 at 300 K | Our replica spread is <0.1% on every scalar, so it is not the limiting uncertainty | — |
| ? Error bars | block method | replica spread across 3 runs | comparable in spirit; theirs not tabulated | p. 3653 |

## 4. Observables and their definitions

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ ΔH_vap definition | ΔH_vap(T) = H_gas(T) − H(T) ≈ RT − H(T), gas assumed ideal | identical: `kT − H/N`, with H from `AverageEnthalpy` (U + P·V) | none — the P·V term is 0.006 kJ/mol at 1 atm | p. 3654; `measurements.py` |
| ✓ Orientational order | η = largest eigenvalue of the Q tensor | S from `NematicOrderParameter` — same definition | none, **but** global S is texture-degenerate: ~0–0.25 for a good 4-domain crystal, ≥0.25 for their 2-spot texture, ~0 for a glass. Judge runs by local face-contact P₂ and the orientation map, never by global S alone | p. 3655; `measurements.py` |
| ⚠ Lattice constants | Table V, experimental (a, b, c) lineage | box edges ÷ supercell reps, then reindexed `[1, 2, 0]` | The initializer tiles the motif in experimental (c, a, b) order; without the reindex our `a` is silently compared against their `c` | Table V; `notebooks/herringbone_runs.ipynb` (`run_scalars`) |
| ✗ Coordination number | 12.5, from the integral of the **liquid** g(r) to its first minimum (7.1 Å) | ≈14 at 100 K, first minimum 6.75 Å, shells overlap | **Not comparable** — different phase and an ambiguous first minimum. Do not quote this pair | p. 3656 |
| ⚠ g₂(r) | Fig 8 is the **liquid at 300 K** | our OCF is reported for both phases | A crystal OCF must not be compared to their Fig 8; the 300 K run is the valid comparison | Fig 8 |
| — C_p, H/N | no published value | reported | nothing to compare against | — |

## 5. Open items (differences whose effect is not quantified)

1. **Equilibration protocol** (§3) — ours is tuned to avoid a glassy trap that their
   paper never mentions. Whether their 100 K crystal is as well converted as ours is
   unknowable from the paper, but it has the right sign to explain the residual
   ΔH_vap gap: a less perfectly converted crystal is less bound, and their Fig 5 η at
   100 K (~0.25) sits slightly below our S = 0.288.
2. **N = 400 vs 500 and the L > 2 r_c criterion** (§3) — untested. The cheap check is
   a single 100 K run at reps (5,5,5) (N = 500), which both matches their N and lifts
   the shortest box edge above 2 r_c.
3. **Coordination number** (§4) — needs a like-for-like definition before it can be
   quoted at all.
4. **Cutoff convention** (§1) — quantified once, by hand, on one frame. It should be
   a flag (`shift=True` in `calc_total_energy`) so both conventions can be reported
   from the same trajectory instead of corrected in prose.

## Reproducing the quantified numbers

The §1 cutoff/shift figures and the §2 core-repulsion probe were computed directly
from `results/validation/100.0_6.324209e-07/herringbone_jittered_2/simulation.db`
and `CACELLI_POTENTIAL`; the §3 convergence and box numbers come from the same db.
They are **not** yet regenerated by a checked-in script — see open item 4. Until they
are, treat the numbers in this document as pinned to the runs listed at the top.
