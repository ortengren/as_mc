# How our protocol differs from Cacelli et al. (2004)

This document compares our validation runs with the simulations in I. Cacelli,
G. Cinacchi, G. Prampolini and A. Tani, *Modeling benzene with single-site
potentials from ab initio calculations: a step toward hybrid models of complex
molecules*, J. Chem. Phys. **120**, 3648 (2004). The PDF is in `literature/`
(gitignored), and page numbers below are the journal's (3648–3657).

The runs compared here all use `CACELLI_POTENTIAL` at 1 atm. They were made with
the per-temperature driver scripts that `scripts/run_herringbone.py` has since
replaced:

| Run dir | T | N | steps (eq + prod) |
|---|---|---|---|
| `results/validation/100.0_6.324209e-07/herringbone` | 100 K | 400 | 10.2M + 15M |
| `results/validation/100.0_6.324209e-07/herringbone_jittered` | 100 K | 400 | 10.2M + 15M |
| `results/validation/100.0_6.324209e-07/herringbone_jittered_2` | 100 K | 400 | 10.2M + 15M |
| `results/validation/300.0_6.324209e-07/herringbone_jittered_0` | 300 K | 400 | 10.2M + 15M |

## Summary

Our structural results agree with their Table V to within 1% (density 0.3%,
lattice constants under 1%). The energetics differ by 6%: our ΔH_vap is 10.86
kcal/mol against their 10.2. The cutoff convention (§1) accounts for 0.16
kcal/mol of that, and the rest is within the 0.5 kcal/mol rms error of their
GBQIII dimer fit. None of the differences listed here is known to invalidate the
comparison. The ones we haven't been able to quantify are collected in §5.

Each row is marked with one of these symbols: `✓` the two protocols match, `⚠`
they differ and we've quantified the effect, `✗` they differ and the effect is
**not** quantified, `?` the paper doesn't say.

---

## 1. The pair potential and how it's evaluated

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ ε₀, σ₀, κ, μ, ν, ξ | 0.800 kcal/mol, 5.720 Å, 0.542, −10.0, 0.41, 0.73 | identical (ε₀ = 0.0347 eV = 0.800 kcal/mol) | none | Table II; `data/lit_gbq_params.json` |
| ✓ Quadrupole Q | −4.130·10⁻²⁶ esu·cm² | −3.263 (eV·Å⁵)^½, which is the same value in our units (Q²[eV·Å⁵] = Q²[esu²cm⁴] × 6.2415·10⁵¹) | none | Table II; `potentials.quadrupole` |
| ⚠ κ′ | 4.39 (as listed) | 0.2278 = 1/4.39 | With the listed 4.39 and the standard GB convention χ′=(κ′^(1/μ)−1)/(κ′^(1/μ)+1), the wells involving a face are 4.4× **too shallow** and their Fig 3 isn't reproduced. The inverted value does reproduce it (χ′(1/κ′) = −χ′(κ′) identically). The resulting wells are FF −1.81 at 3.93 Å, SP −1.99, TS −1.94 at 5.08 Å, SS −0.80 and CR −0.78 kcal/mol. These are on or slightly *shallower* than their Fig 3 GBQIII curves, so κ′ is not a source of excess binding | Table II, Fig 3; `potentials.gb_directional_energy` |
| ⚠ Cutoff scheme | truncated **and shifted** at 15 Å | plain truncation at an effective 13.6 Å (`nl_radius=6.8` is a *per-atom* radius, and ASE adds the two radii of a pair) | **+0.16 kcal/mol on ΔH_vap, ours high.** On the final 100 K frame there are 147 neighbours per molecule within 15 Å and ⟨u(15 Å)⟩ = −0.139 meV per pair. Their shift therefore removes 0.236 kcal/mol of binding, while the 13.6–15 Å tail we leave out costs 0.077. U/N is −10.665 kcal/mol for us and −10.506 in their convention | p. 3653; `potentials.calc_total_energy`, `MetropolisSampler(nl_radius=...)` |
| ✓ Electrostatics | quadrupole–quadrupole truncated at the same cutoff, no Ewald sum | same | none | p. 3653; `potentials.quadrupole` |
| ✓ Pair enumeration | all N² pairs | ASE `neighbor_list` with a 1.0 Å skin | same physics, different cost | p. 3653; `potentials.calc_total_energy` |
| Working units | kcal/mol, Å | eV, Å (multiply by 23.0605 to compare) | bookkeeping only | |

## 2. Ensemble and trial moves

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ Ensemble | MC NPT, P = 1 atm | MC NPT, P = 6.324209·10⁻⁷ eV/Å³ = 1 atm | none | p. 3653; `metropolis.py` |
| ✓ Volume-move geometry | one randomly chosen box edge is stretched, so the orthorhombic shape can relax | `aniso_vol=True`: one randomly chosen axis is rescaled per move | None. All four runs above use `aniso_vol=True`. Earlier isotropic-only runs (in `results/archive/`) couldn't relax the a:b:c ratio and ended up with a texture imposed by the box | p. 3653; `trial_moves.py` |
| ✓ Box shape freedom | axis lengths only, so the box stays orthorhombic | same, with no shear or tilt | neither can find a monoclinic or triclinic cell | p. 3653; `trial_moves.py` |
| ✓ Tuning the move widths | maximum displacements adjusted to ~30% acceptance, then production | tuned during equilibration only; `calculate_trajectory` calls `block_update(dynamic_delta=False)` | None. Both sample production with fixed proposal widths, so detailed balance holds | p. 3653; `MetropolisSampler.calculate_trajectory` |
| ⚠ Acceptance target | ~30% for all move types | `TARGET_ACC_RATE = 0.275` | affects efficiency only, not the equilibrium distribution | p. 3653; `metropolis.TARGET_ACC_RATE` |
| ? Move mix | "all kinds of moves", with the mix not stated (a "cycle" is presumably N attempts) | per step, P(translation) = P(rotation) = (N−1)/2N and P(volume) = 1/N, so one volume move per sweep on average | we assume they're equivalent, but the paper doesn't give enough detail to check | p. 3653; `MetropolisSampler.step` |
| ⚠ Rejecting close contacts | moves that sample r < r_asy (≤ σ − σ₀ξ) are **refused**, to avoid unphysical regions of the GB surface | **no overlap test and no r_asy guard**: rejection is left entirely to the Boltzmann factor | We checked numerically that this is harmless at these densities. With the GBQIII parameters the core stays strongly repulsive on *both* sides of the r = σ(r̂) − ξσ₀ pole: the edge-to-edge pole is at 1.54 Å with U ≥ 10⁶ kcal/mol at every r < 2 Å, and face-to-face and T-shaped have no pole at r > 0. Overlaps are therefore rejected with probability ~1, and the closest contact in our crystal is 4.4 Å. The cost is lower acceptance at low density or high T, where proposals can land in the core. A proposal that lands exactly on the pole gives `inf` and is rejected | p. 3653; `MetropolisSampler.step` |

## 3. System and run protocol

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ⚠ Particle count N | **500**, the 4-molecule experimental cell replicated 5×5×5 | **400**, the same motif replicated (5, 5, 4) | A 20% finite-size difference, which also changes the box. Their converged cell (Table V) gives a 43.75 × 30.95 × 35.75 Å box whose shortest edge, 30.95 Å, just meets the usual L > 2 r_c = 30 Å criterion. Our mean box is 35.88 × 43.75 × **24.60** Å, whose shortest edge is below 2 r_c = 27.2 Å, so along that axis a molecule can interact with two periodic images of the same neighbour. The energy is still the correct truncated lattice sum: ASE sums every image within the cutoff, and the shortest lattice vector (24.6 Å) is longer than the 13.6 Å cutoff, so there are no self-image terms. But our configuration is more correlated with its own images than theirs | p. 3653, Table V; `HerringboneLatticeInitializer` |
| ⚠ Starting configuration | the experimental crystal structure at 138 K, with four molecules per unit cell; **every** simulation starts from it, at every T | the same experimental Pbca motif via `HerringboneLatticeInitializer`, with position and orientation jitter (0.0, 0.1 and 0.15 across the three replicas) | Same starting polymorph. The jitter is there to make the replicas independent, and isn't part of their protocol | p. 3653; `scripts/run_herringbone.py` |
| ✗ Equilibration | not described beyond 50 000 cycles at the target T | **compress before melting**: `vol_delt = 0.025` (fast box collapse) and `max_or_delt = 0.25 rad` (capped rotations). Without the cap, the adaptive tuner pushes or_delt to ~1.2 rad in the shallow orientational landscape of the herringbone crystal, which melts the crystal before it densifies and leaves a glass | Can't be compared with the paper, which doesn't describe its protocol in this much detail. Their volume step at 30% acceptance is about 10× ours, so their box would also have collapsed before the orientations melted | p. 3653; `scripts/run_herringbone.py` |
| ⚠ Run length | 50 000 cycles of equilibration and 100 000 of production at N = 500 | 10.2M + 15M single-particle steps at N = 400, which is 25 500 + 37 500 sweeps | They have about 2.7× more production sweeps per particle. Our production passes the half-split convergence test: the volume drifts by −0.01% (100 K) and −0.08% (300 K) between the two halves | p. 3653; convergence cell of `notebooks/herringbone_runs.ipynb` on `archive/pre-cleanup` |
| ⚠ State points | 100–400 K in 50 K steps, all at 1 atm | 100 K and 300 K at 1 atm | We don't have a melting curve. Their GBQIII melts between 100 and 150 K according to the text on p. 3654, or between 150 and 200 K according to Table IV; the paper contradicts itself here. Our 300 K run is liquid, which is consistent with either | Fig 5, Table IV |
| ⚠ Replicas | 1 (as far as the paper says) | 3 at 100 K, 1 at 300 K | Our replicas agree to within 0.1% on every scalar, so replica spread isn't the limiting uncertainty | |
| ? Error bars | block averaging | spread across the 3 replicas | similar in spirit; they don't tabulate theirs | p. 3653 |

## 4. Observables and how they're defined

| Aspect | Cacelli et al. 2004 | asmcmc | Effect | Source |
|---|---|---|---|---|
| ✓ ΔH_vap | ΔH_vap(T) = H_gas(T) − H(T) ≈ RT − H(T), treating the gas as ideal | the same: `kT − H/N`, with H = U + P·V from `AverageEnthalpy` | none; the P·V term is 0.006 kJ/mol at 1 atm | p. 3654; `measurements.AverageEnthalpy` |
| ✓ Orientational order | η, the largest eigenvalue of the Q tensor | S from `NematicOrderParameter`, with the same definition | None, **but** global S can't distinguish textures: it's ~0–0.25 for a good 4-domain crystal, 0.25 or more for their 2-spot texture, and ~0 for a glass. Judge a run by the local face-contact P₂ and the orientation map, not by global S alone | p. 3655; `measurements.NematicOrderParameter` |
| ⚠ Lattice constants | Table V, with the experimental (a, b, c) labelling | box edges divided by the supercell repeats, then reordered `[1, 2, 0]` | The initializer tiles the motif in the experimental (c, a, b) order. Without the reordering, our `a` would be compared against their `c` without any warning | Table V; `run_scalars` in `notebooks/herringbone_runs.ipynb` on `archive/pre-cleanup` |
| ✗ Coordination number | 12.5, from integrating the **liquid** g(r) up to its first minimum (7.1 Å) | about 14 at 100 K, with the first minimum at 6.75 Å and overlapping shells | **Not comparable.** The phases differ and our first minimum is ambiguous, so don't quote these two numbers side by side | p. 3656 |
| ⚠ g₂(r) | Fig 8 is the **liquid at 300 K** | we report the orientational correlation function for both phases | Don't compare a crystal's OCF with their Fig 8. Our 300 K run is the right comparison | Fig 8 |
| C_p, H/N | no published values | reported | nothing to compare against | |

## 5. Open items (differences whose effect we haven't quantified)

1. **Equilibration protocol** (§3). Ours is tuned to avoid a glassy state that
   their paper never mentions. We can't tell from the paper whether their 100 K
   crystal converted as fully as ours. If it didn't, that would help explain the
   remaining ΔH_vap gap, since a less fully converted crystal is less bound, and
   their Fig 5 η at 100 K (~0.25) is slightly below our S = 0.288.
2. **N = 400 vs 500, and the L > 2 r_c criterion** (§3). Not tested yet. The
   cheap check is a single 100 K run with (5, 5, 5) repeats (N = 500), which both
   matches their N and makes the shortest box edge longer than 2 r_c.
3. **Coordination number** (§4). We need a like-for-like definition before this
   can be quoted at all.
4. **Cutoff convention** (§1). This was quantified once, by hand, on one frame.
   It should become an option (say `shift=True` in `calc_total_energy`) so that
   both conventions can be computed from the same trajectory instead of being
   corrected for in the text.

## Reproducing the numbers

The cutoff and shift figures in §1 and the core-repulsion check in §2 were
computed directly from
`results/validation/100.0_6.324209e-07/herringbone_jittered_2/simulation.db` and
`CACELLI_POTENTIAL`. The convergence and box numbers in §3 come from the same db.
No checked-in script regenerates them yet (see open item 4), so for now treat
them as specific to the runs listed at the top.
