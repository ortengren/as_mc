"""What a :mod:`asmcmc.data_preparation.cluster_dataset` campaign says about itself.

Two questions, which is why this exists as one module rather than notebook cells:

**QA.** Is the campaign trustworthy? Checks for disjoint shards, no hard-core
violations, no duplicates, and that the recomputed GB+Q baseline reproduces the
stored one. :func:`qa_report`.

**Target.** How large the Delta-learning target ``E_UMA - E_GBQ`` is and where
in geometry it lives, which is what sets a fit's weighting. :func:`pair_records`
is the unpacking a Delta-learning fit wants; it is model-agnostic (geometry plus
both energies), with the GB invariants applied on top (:func:`gb_invariants`).
:func:`radial_profile`, :func:`motif_masks` (now a post-hoc census -- see
CLAUDE.md's MOTIF GENERATION note; there is no generator-side taxonomy any more).

Every campaign is dimers only (2026-09; trimers dropped, see CLAUDE.md's TRIMER
note), so a frame's ``gbq_interaction_energy`` is one pair's baseline directly --
no cluster-sum attribution is needed. :func:`qa_report` still cross-checks it
against a recomputation via ``CACELLI_POTENTIAL.pair_energy``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.neighborlist import neighbor_list

from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.data_preparation.cluster_dataset import CONFIG_NAME, dataset_frames

EV_TO_KCAL = 23.060541945329334

# Motif cuts in (|b|, slip). Reporting labels, not a taxonomy: these quantities
# are rotation- and inversion-invariant, so they name a contact and cannot
# identify a lattice -- two crystals sharing a motif but sitting on different
# lattices score the same (measured on the 100 K validation run vs the Cacelli
# minimum: same motif, same contact fractions, differing lattices and RDFs).
COFACIAL = "cofacial"
T_SHAPED = "T-shaped"
PARALLEL_DISPLACED = "parallel-displaced"
FAR_SLIPPED = "far-slipped"

WELL_RANGE = (3.4, 6.0)

# Boundaries in **slip** (lateral offset, A), not in a_hi. An earlier version cut
# parallel-displaced as ``a_hi < 0.6``, which was wrong: the Cacelli PD minimum
# sits at a_hi = 0.909 (r = 3.85, stacking height 3.5, slip 1.6), so the actual
# PD motif was being counted as cofacial and the "PD" bucket collected
# far-slipped pairs instead. Cofacial and PD differ only by slip -- 0.0 vs
# 1.6 A at essentially the same height -- so slip is the coordinate that
# separates them and a_hi is not.
COFACIAL_MAX_SLIP = 1.0
DISPLACED_MAX_SLIP = 3.0
PARALLEL_MIN_B = 0.8
T_MAX_B = 0.35
DEFAULT_RADIAL_EDGES = (3.4, 5.0, 6.0, 7.0, 9.0, 15.0, np.inf)


def load_campaign(out_dir):
    """``(frames, config)`` for a campaign directory."""
    out_dir = Path(out_dir)
    config_path = out_dir / CONFIG_NAME
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return dataset_frames(out_dir), config


def _normals(frame):
    u = np.asarray(frame.info["or_vec"], dtype=float)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def pair_records(frames, potential=CACELLI_POTENTIAL):
    """Every dimer in a campaign, as a dict of flat arrays (one entry per frame).

    Keys: ``r``, ``a_i``, ``a_j``, ``b`` (the pair geometry), ``e_uma`` and
    ``e_gbq`` (interaction energies, eV), ``delta`` (``e_uma - e_gbq``, the
    Delta-learning target), plus provenance ``shard``, ``config_index``.

    Geometry comes from the stored ``molecular_com``/``or_vec``, so no frame is
    re-coarse-grained and the numbers are exactly the ones the generator used.
    """
    r, a_i, a_j, b = [], [], [], []
    e_uma, e_gbq = [], []
    shard, config_index = [], []

    for frame in frames:
        com = np.asarray(frame.info["molecular_com"], dtype=float)
        u = _normals(frame)
        disp = com[1] - com[0]
        dist = float(np.linalg.norm(disp))
        r_hat = disp / dist

        r.append(dist)
        a_i.append(float(np.dot(r_hat, u[0])))
        a_j.append(float(np.dot(r_hat, u[1])))
        b.append(float(np.dot(u[0], u[1])))
        e_uma.append(float(frame.info["interaction_energy"]))
        e_gbq.append(float(potential.pair_energy(u[:1], u[1:], disp[None, :])[0]))
        shard.append(int(frame.info.get("shard", -1)))
        config_index.append(int(frame.info.get("config_index", -1)))

    def arr(values, dtype=float):
        return np.asarray(values, dtype=dtype)

    records = {
        "r": arr(r),
        "a_i": arr(a_i),
        "a_j": arr(a_j),
        "b": arr(b),
        "e_uma": arr(e_uma),
        "e_gbq": arr(e_gbq),
        "shard": arr(shard, int),
        "config_index": arr(config_index, int),
    }
    records["delta"] = records["e_uma"] - records["e_gbq"]
    return records


def reweighting_ess(records, target_r, edges=None):
    """ESS fraction of reweighting a campaign's pairs to a target radial census.

    **This is the criterion that means something**: it asks whether the campaign
    can be reweighted to a distribution you would actually fit against -- the
    separations an MC run really visits -- and how many effective samples
    survive. Unlike :func:`weight_diagnostics` it needs no recorded density, so
    it is comparable across campaigns sampled by different proposals.

    Support, not shape, is what can make this collapse: a campaign that never
    sampled a shell the target occupies gets zero weight there and no
    reweighting can recover it.
    """
    edges = np.asarray(
        edges if edges is not None else np.linspace(3.4, 15.0, 24), dtype=float
    )
    campaign_counts, _ = np.histogram(records["r"], bins=edges)
    target_counts, _ = np.histogram(np.asarray(target_r, dtype=float), bins=edges)

    inside = (records["r"] >= edges[0]) & (records["r"] <= edges[-1])
    if not inside.any() or campaign_counts.sum() == 0 or target_counts.sum() == 0:
        return float("nan")

    p_campaign = campaign_counts / campaign_counts.sum()
    p_target = target_counts / target_counts.sum()
    index = np.clip(np.digitize(records["r"][inside], edges) - 1, 0, len(edges) - 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(
            campaign_counts[index] > 0, p_target[index] / p_campaign[index], 0.0
        )
    if weights.sum() == 0:
        return 0.0
    return float(weights.sum() ** 2 / np.sum(weights**2) / len(weights))


def gb_invariants(records):
    """``(|b|, a_hi, a_lo)`` -- the symmetry-folded pair coordinates.

    Uniaxial discs satisfy ``u = -u`` and pairs are unordered, so only absolute
    values and the sorted pair carry information, in cosines rather than
    degrees because the coverage maps are plotted against ``|b|``.
    """
    abs_i, abs_j = np.abs(records["a_i"]), np.abs(records["a_j"])
    return (
        np.abs(records["b"]),
        np.maximum(abs_i, abs_j),
        np.minimum(abs_i, abs_j),
    )


def stack_coordinates(records):
    """``(height, slip)`` are the natural coordinates of a near-parallel pair.

    ``height = r·a_hi`` is the separation along the more-aligned normal and
    ``slip = r·sqrt(1 - a_hi²)`` the lateral offset. Cofacial, parallel-displaced
    and far-slipped are one continuous family in ``slip`` at roughly fixed
    ``height``, which is why these -- not ``a_hi`` -- are what the motif cuts use.
    """
    _, a_hi, _ = gb_invariants(records)
    r = records["r"]
    return r * a_hi, r * np.sqrt(np.clip(1.0 - a_hi**2, 0.0, None))


def motif_masks(records, well_range=WELL_RANGE):
    """Boolean masks naming the canonical benzene dimer contacts.

    Restricted to the well region: outside it the same angles describe a pair
    that is barely interacting, and pooling those in would dilute every
    per-motif statistic toward zero.

    Split on **slip**, per the module constants -- see ``COFACIAL_MAX_SLIP`` for
    why an ``a_hi`` cut misclassifies the parallel-displaced motif. ``FAR_SLIPPED``
    is reported separately rather than folded into PD because it is where the
    GB+Q baseline is worst (Δ rms 2.0 kcal/mol against PD's 0.45), so merging
    them would hide the one region that most needs sampling.
    """
    abs_b, _, a_lo = gb_invariants(records)
    _, slip = stack_coordinates(records)
    lo, hi = well_range
    in_well = (records["r"] >= lo) & (records["r"] < hi)
    parallel = in_well & (abs_b > PARALLEL_MIN_B)

    return {
        COFACIAL: parallel & (slip < COFACIAL_MAX_SLIP),
        PARALLEL_DISPLACED: parallel
        & (slip >= COFACIAL_MAX_SLIP)
        & (slip <= DISPLACED_MAX_SLIP),
        FAR_SLIPPED: parallel & (slip > DISPLACED_MAX_SLIP),
        T_SHAPED: in_well & (abs_b < T_MAX_B) & (a_lo < T_MAX_B),
    }


def radial_profile(records, edges=DEFAULT_RADIAL_EDGES, units=EV_TO_KCAL):
    """Per-shell counts and rms energies -- the table that localises Delta.

    ``delta_sq_share`` is each shell's fraction of the total ``sum(Delta^2)``:
    the quantity that decides where a fit's error budget actually goes, which
    a count or a mean cannot show.
    """
    edges = np.asarray(edges, dtype=float)
    delta = records["delta"] * units
    total_sq = float(np.sum(delta**2))

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (records["r"] >= lo) & (records["r"] < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "r_lo": lo,
                    "r_hi": hi,
                    "count": 0,
                    "fraction": 0.0,
                    "e_uma_rms": 0.0,
                    "delta_rms": 0.0,
                    "delta_absmax": 0.0,
                    "delta_sq_share": 0.0,
                }
            )
            continue
        d = delta[mask]
        rows.append(
            {
                "r_lo": lo,
                "r_hi": hi,
                "count": count,
                "fraction": count / len(records["r"]),
                "e_uma_rms": float(
                    np.sqrt(np.mean((records["e_uma"][mask] * units) ** 2))
                ),
                "delta_rms": float(np.sqrt(np.mean(d**2))),
                "delta_absmax": float(np.max(np.abs(d))),
                "delta_sq_share": float(np.sum(d**2) / total_sq) if total_sq else 0.0,
            }
        )
    return rows


# --- QA ----------------------------------------------------------------------


@dataclass(frozen=True)
class QAReport:
    """Campaign integrity. ``ok`` is the single gate; the fields say why."""

    n_frames: int
    n_dimers: int
    n_pairs: int
    unique_ids: bool
    shard_seeds: dict
    monomer_energy_spread: float
    min_atom_distance: float
    hard_core_violations: int
    baseline_max_error: float
    duplicate_signatures: int
    non_finite_frames: int
    problems: tuple = field(default_factory=tuple)

    @property
    def ok(self):
        return not self.problems


def _min_intermolecular_distance(frame):
    positions = frame.get_positions()
    ids = np.asarray(frame.arrays["molecule_id"], dtype=int)
    groups = [positions[ids == m] for m in range(int(ids.max()) + 1)]
    best = np.inf
    for a in range(len(groups)):
        for b_ in range(a + 1, len(groups)):
            delta = groups[a][:, None, :] - groups[b_][None, :, :]
            best = min(best, float(np.sqrt(np.min(np.sum(delta**2, axis=-1)))))
    return best


def _configuration_signature(frame, decimals=4):
    """A fingerprint identifying a cluster up to rigid motion and relabelling.

    Per pair: separation plus the ``(|a_i|, |a_j|, |b|)`` invariants, sorted
    over pairs. Rotation-, inversion- and relabelling-invariant, which is what
    "the same configuration" has to mean here.

    **Orientation is included deliberately.** An earlier version fingerprinted
    inter-centre distances alone, which for a dimer is a *single* number: with
    141 dimers concentrated by motif sampling into a narrow radial band, two
    unrelated configurations collide at 1e-4 A resolution with ~30% probability,
    and the check duly reported a phantom duplicate on the first motif campaign.
    Four invariants per pair make coincidence negligible while a genuine
    duplicate still matches exactly.
    """
    com = np.asarray(frame.info["molecular_com"], dtype=float)
    u = _normals(frame)
    i, j = np.triu_indices(len(com), k=1)

    disp = com[j] - com[i]
    r = np.linalg.norm(disp, axis=1)
    r_hat = disp / np.where(r > 0, r, 1.0)[:, None]
    a_i = np.abs(np.einsum("pk,pk->p", r_hat, u[i]))
    a_j = np.abs(np.einsum("pk,pk->p", r_hat, u[j]))
    b = np.abs(np.einsum("pk,pk->p", u[i], u[j]))

    per_pair = np.round(
        np.stack([r, np.minimum(a_i, a_j), np.maximum(a_i, a_j), b], axis=1), decimals
    )
    return (len(com),) + tuple(sorted(map(tuple, per_pair.tolist())))


def qa_report(frames, config=None, potential=CACELLI_POTENTIAL):
    """Check a campaign end to end."""
    config = config or {}
    settings = config.get("settings", {})
    min_atom_distance = float(settings.get("min_atom_distance", 0.0))

    problems = []
    n_dimers = 0
    ids, signatures = [], []
    monomer_energies = []
    baseline_error = 0.0
    min_distance = np.inf
    violations = non_finite = 0

    for frame in frames:
        n_dimers += int(frame.info["n_molecules"]) == 2

        ids.append(
            (int(frame.info.get("shard", -1)), int(frame.info.get("config_index", -1)))
        )
        signatures.append(_configuration_signature(frame))
        monomer_energies.extend(np.atleast_1d(frame.info["monomer_energies"]).tolist())

        distance = _min_intermolecular_distance(frame)
        min_distance = min(min_distance, distance)
        violations += distance < min_atom_distance - 1e-9

        energy = float(frame.get_potential_energy())
        if not np.isfinite(energy):
            non_finite += 1

        # The stored baseline against this module's per-pair rebuild: the check
        # that licenses pair_records to trust gbq_interaction_energy directly.
        com = np.asarray(frame.info["molecular_com"], dtype=float)
        u = _normals(frame)
        i, j = np.triu_indices(len(com), k=1)
        rebuilt = float(np.sum(potential.pair_energy(u[i], u[j], com[j] - com[i])))
        baseline_error = max(
            baseline_error, abs(rebuilt - float(frame.info["gbq_interaction_energy"]))
        )

    monomer_spread = (
        float(np.max(monomer_energies) - np.min(monomer_energies))
        if monomer_energies
        else 0.0
    )
    unique_ids = len(set(ids)) == len(ids)
    duplicates = len(signatures) - len(set(signatures))

    shard_seeds = {}
    for frame in frames:
        shard_seeds.setdefault(int(frame.info.get("shard", -1)), set()).add(
            int(frame.info.get("generator_seed", -1))
        )
    shard_seeds = {k: sorted(v) for k, v in sorted(shard_seeds.items())}

    if not unique_ids:
        problems.append("duplicate (shard, config_index)")
    if violations:
        problems.append(
            f"{violations} hard-core violations below {min_atom_distance} A"
        )
    if baseline_error > 1e-6:
        problems.append(f"GBQ baseline mismatch {baseline_error:.3e} eV")
    if duplicates:
        problems.append(f"{duplicates} duplicate configurations")
    if non_finite:
        problems.append(f"{non_finite} frames with non-finite energy")
    if settings.get("rigid", True) and monomer_spread > 1e-9:
        problems.append(
            f"rigid run has a varying monomer reference ({monomer_spread:.3e} eV)"
        )
    if any(len(seeds) > 1 for seeds in shard_seeds.values()):
        problems.append("a shard mixes generator seeds")

    return QAReport(
        n_frames=len(frames),
        n_dimers=n_dimers,
        n_pairs=n_dimers,
        unique_ids=unique_ids,
        shard_seeds=shard_seeds,
        monomer_energy_spread=monomer_spread,
        min_atom_distance=float(min_distance),
        hard_core_violations=violations,
        baseline_max_error=baseline_error,
        duplicate_signatures=duplicates,
        non_finite_frames=non_finite,
        problems=tuple(problems),
    )


# --- why short-range coverage is thin ----------------------------------------


def _random_rotations(rng, n):
    """``n`` uniform SO(3) rotations, from unit quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        axis=1,
    )


def _small_rotations(rng, n, sigma):
    """``n`` rotations of angle ``N(0, sigma)`` about uniformly random axes."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.normal(0.0, sigma, size=n)

    zero = np.zeros(n)
    K = np.stack(
        [
            np.stack([zero, -axis[:, 2], axis[:, 1]], -1),
            np.stack([axis[:, 2], zero, -axis[:, 0]], -1),
            np.stack([-axis[:, 1], axis[:, 0], zero], -1),
        ],
        axis=1,
    )
    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    s, c = np.sin(angle)[:, None, None], (1 - np.cos(angle))[:, None, None]
    return eye + s * K + c * (K @ K)


def hard_core_pass_rate(
    radii, orientations="random", n_trials=2000, threshold=None, tilt_sigma=0.35, seed=0
):
    """Fraction of trial dimer placements at each separation clearing the hard core.

    This is the measurement that explains why a campaign's short-range coverage
    is thin, and it is a property of the *proposal*, not of the data: at close
    approach two benzenes in random relative orientation almost always clash, so
    the ``min_atom_distance`` filter in ``make_cluster`` rejects most of what is
    proposed there. Raising ``compact_probability`` pushes against that filter
    rather than past it -- the rejected draws are simply redrawn -- which is why
    the radial distribution alone cannot fix short-range coverage.

    ``orientations``:
      * ``"random"`` -- uniform SO(3), what ``make_cluster`` currently does.
      * ``"cofacial"`` -- both discs near a common normal with ``tilt_sigma``
        jitter and the separation along it, i.e. proposing a geometry that close
        approach actually permits.

    Only these two are offered on purpose. Cofacial is the clean demonstration
    that the proposal, not the filter, is the binding constraint; seeding
    T-shaped or parallel-displaced needs their real geometry (PD sits near 3.8 A
    with ~1.6 A slip) and a crude seed would understate them rather than make
    the point.

    ``threshold`` defaults to ``SamplingSettings.min_atom_distance``.
    """
    from asmcmc.data_preparation.cluster_dataset import (
        SamplingSettings,
        build_reference_benzene,
    )

    if threshold is None:
        threshold = SamplingSettings().min_atom_distance
    if orientations not in ("random", "cofacial"):
        raise ValueError(f"unknown orientations {orientations!r}")

    rng = np.random.default_rng(seed)
    positions = build_reference_benzene().get_positions()
    radii = np.atleast_1d(np.asarray(radii, dtype=float))

    rates = []
    for radius in radii:
        if orientations == "random":
            rot_a = _random_rotations(rng, n_trials)
            rot_b = _random_rotations(rng, n_trials)
            direction = rng.normal(size=(n_trials, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        else:
            rot_a = _small_rotations(rng, n_trials, tilt_sigma)
            rot_b = _small_rotations(rng, n_trials, tilt_sigma)
            direction = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n_trials, 3))

        pos_a = np.einsum("nij,kj->nki", rot_a, positions)
        pos_b = (
            np.einsum("nij,kj->nki", rot_b, positions) + radius * direction[:, None, :]
        )

        delta = pos_a[:, :, None, :] - pos_b[:, None, :, :]
        closest = np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=(1, 2)))
        rates.append(float(np.mean(closest >= threshold)))

    return np.asarray(rates)


# --- the MC reference --------------------------------------------------------


def _mc_frame_orientations(frame, or_vecs=None):
    """Unit disc normals for an MC ``ase.Atoms`` frame, from ``or_vecs`` if given.

    Distinct from :func:`_normals`: a campaign frame carries orientation in
    ``frame.info["or_vec"]``, an MC db frame in ``frame.arrays["or_vec"]``
    (read from ``row.data`` in :func:`mc_pair_reference`, since
    ``row.toatoms()`` does not carry it).
    """
    u = frame.arrays["or_vec"] if or_vecs is None else or_vecs
    u = np.asarray(u, dtype=float)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def pair_geometry(frame, cutoff, or_vecs=None):
    """All directed neighbour pairs within ``cutoff`` (PBC, incl. self-images).

    The same quantities and conventions as
    ``fitting_gbq.data.extract_periodic_pairs`` -- so pair statistics here stay
    comparable with the fitting code -- but the centre index ``i`` is kept,
    which that function discards.

    Returns a dict of equal-length arrays: ``i``, ``j``, ``r`` (separation),
    ``a_i``/``a_j`` (``r_hat . u``), ``b`` (``u_i . u_j``).
    """
    u = _mc_frame_orientations(frame, or_vecs)
    i, j, offsets = neighbor_list("ijS", frame, cutoff)
    disp = frame.positions[j] + offsets @ np.asarray(frame.cell) - frame.positions[i]
    r = np.linalg.norm(disp, axis=1)
    r_hat = disp / r[:, None]
    return {
        "i": i,
        "j": j,
        "r": r,
        "a_i": np.einsum("pk,pk->p", r_hat, u[i]),
        "a_j": np.einsum("pk,pk->p", r_hat, u[j]),
        "b": np.einsum("pk,pk->p", u[i], u[j]),
    }


def mc_pair_reference(db_path, n_frames=20, cutoff=15.0, tail_fraction=0.5):
    """Pair geometry the sampler actually visits, as a ``pair_records``-shaped dict.

    This is the comparison that says whether a cluster campaign covers the
    geometries the MC will ask the potential about. Frames are drawn evenly
    across the run's converged tail, so the sample is decorrelated rather than
    being one snapshot.

    ``or_vec`` is read from ``row.data`` because ``row.toatoms()`` does not
    carry it -- the same reconstruction ``notebooks/training_coverage_residual.ipynb``
    uses. Energies are not defined for these pairs (an MC frame carries no
    per-pair label), so ``e_uma``/``delta`` are absent by construction.
    """
    with connect(str(db_path)) as db:
        row_ids = [row.id for row in db.select()]
        if not row_ids:
            raise ValueError(f"{db_path} holds no rows")

        tail = row_ids[int(len(row_ids) * (1.0 - tail_fraction)) :]
        step = max(1, len(tail) // n_frames)
        picked = tail[::step][:n_frames]

        chunks = []
        for row_id in picked:
            row = db.get(id=row_id)
            frame = row.toatoms()
            frame.arrays["or_vec"] = np.asarray(row.data["or_vec"], dtype=float)
            chunks.append(pair_geometry(frame, cutoff))

    return {
        "r": np.concatenate([c["r"] for c in chunks]),
        "a_i": np.concatenate([c["a_i"] for c in chunks]),
        "a_j": np.concatenate([c["a_j"] for c in chunks]),
        "b": np.concatenate([c["b"] for c in chunks]),
        "n_frames": len(picked),
    }
