"""Unpacking, QA, and reduction over a cluster campaign.

No MLIP: campaigns are driven through the generator with the same stub
calculator ``test_cluster_dataset.py`` uses, so the suite stays runnable from a
fresh clone without fairchem or a GPU.
"""

import numpy as np
import pytest

from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.data_preparation.cluster_analysis import (
    COFACIAL,
    _configuration_signature,
    frame_records,
    weight_diagnostics,
    EV_TO_KCAL,
    FAR_SLIPPED,
    PARALLEL_DISPLACED,
    T_SHAPED,
    gb_invariants,
    hard_core_pass_rate,
    load_campaign,
    motif_masks,
    pair_records,
    qa_report,
    radial_profile,
    reweighting_ess,
    stack_coordinates,
    three_body_records,
)
from asmcmc.data_preparation.cluster_dataset import build_reference_benzene, main

from test_cluster_dataset import StubCalculator  # noqa: F401  (shared stub)


@pytest.fixture
def stub_uma(monkeypatch):
    import asmcmc.data_preparation.cluster_dataset as cd

    monkeypatch.setattr(cd, "load_uma_calculator", lambda *a, **k: StubCalculator())
    return cd


@pytest.fixture
def campaign(tmp_path, stub_uma):
    """A small full-decomposition campaign: dimers and trimers both present."""
    main(
        n_configs=12,
        out_dir=tmp_path,
        n_shards=1,
        decomposition="full",
        trimer_fraction=0.5,
        max_workers=1,
    )
    return load_campaign(tmp_path)


def _dimer(offset, u_a=None, u_b=None):
    """A two-molecule frame with the metadata pair_records reads.

    Built directly rather than through the generator so the geometry under test
    is the one written here.
    """
    reference = build_reference_benzene()
    a, b = reference.copy(), reference.copy()
    b.translate(offset)
    frame = a + b
    frame.arrays["molecule_id"] = np.array([0] * 12 + [1] * 12, dtype=np.int32)

    normals = np.array(
        [
            [0.0, 0.0, 1.0] if u_a is None else u_a,
            [0.0, 0.0, 1.0] if u_b is None else u_b,
        ],
        dtype=float,
    )
    com = np.array([[0.0, 0.0, 0.0], np.asarray(offset, dtype=float)])
    frame.info.update(
        {
            "n_molecules": 2,
            "molecular_com": com,
            "or_vec": normals,
            "interaction_energy": -0.05,
            "gbq_interaction_energy": float(
                CACELLI_POTENTIAL.pair_energy(normals[:1], normals[1:], com[1:] - com[:1])[0]
            ),
            "monomer_energies": np.array([-1.0, -1.0]),
            "shard": 0,
            "config_index": 0,
        }
    )
    return frame


# --- the unpacking -----------------------------------------------------------


def test_a_trimer_is_worth_three_pair_labels(campaign):
    """The headline: pair count is dimers + 3*trimers, not the frame count."""
    frames, _ = campaign
    n_dimers = sum(f.info["n_molecules"] == 2 for f in frames)
    n_trimers = sum(f.info["n_molecules"] == 3 for f in frames)
    assert n_dimers and n_trimers, "fixture must exercise both cluster sizes"

    records = pair_records(frames)
    assert len(records["r"]) == n_dimers + 3 * n_trimers
    assert (records["n_molecules"] == 3).sum() == 3 * n_trimers


def test_trimer_pair_energies_are_the_stored_mlip_labels(campaign):
    """Pair labels must be the frame's own values -- these came from real MLIP
    calls on the 2-molecule subsets and must not be re-derived or approximated."""
    frames, _ = campaign
    records = pair_records(frames)

    stored = np.concatenate(
        [
            np.atleast_1d(
                f.info["pair_interaction_energies"]
                if f.info["n_molecules"] == 3
                else f.info["interaction_energy"]
            )
            for f in frames
        ]
    )
    assert np.allclose(np.sort(records["e_uma"]), np.sort(stored))


def test_per_pair_baseline_sums_to_the_stored_cluster_baseline(campaign):
    """A trimer's stored gbq_interaction_energy is cluster-summed; attributing
    it to individual pairs is only legitimate if the rebuild reproduces it."""
    frames, _ = campaign
    records = pair_records(frames)

    start = 0
    for frame in frames:
        count = 1 if frame.info["n_molecules"] == 2 else 3
        rebuilt = records["e_gbq"][start : start + count].sum()
        assert rebuilt == pytest.approx(float(frame.info["gbq_interaction_energy"]), abs=1e-9)
        start += count


def test_pair_geometry_matches_a_hand_built_dimer():
    """Cofacial stacking along z: r along the normals, so a_i = a_j = b = 1."""
    records = pair_records([_dimer([0.0, 0.0, 4.0])])
    assert records["r"][0] == pytest.approx(4.0)
    assert records["a_i"][0] == pytest.approx(1.0)
    assert records["a_j"][0] == pytest.approx(1.0)
    assert records["b"][0] == pytest.approx(1.0)

    # T-shaped: the second normal perpendicular to the separation and the first.
    t_records = pair_records([_dimer([0.0, 0.0, 5.0], u_b=[1.0, 0.0, 0.0])])
    assert t_records["a_i"][0] == pytest.approx(1.0)
    assert t_records["a_j"][0] == pytest.approx(0.0)
    assert t_records["b"][0] == pytest.approx(0.0)


def test_delta_is_uma_minus_baseline(campaign):
    frames, _ = campaign
    records = pair_records(frames)
    assert np.allclose(records["delta"], records["e_uma"] - records["e_gbq"])


def test_invariants_are_blind_to_normal_sign():
    """u = -u for a uniaxial disc, so flipping a normal must move nothing that
    the coverage maps or motif cuts are built on."""
    up = pair_records([_dimer([1.0, 0.0, 4.0])])
    down = pair_records([_dimer([1.0, 0.0, 4.0], u_a=[0.0, 0.0, -1.0])])

    for a, b in zip(gb_invariants(up), gb_invariants(down)):
        assert np.allclose(a, b)
    # ... and the raw signed quantity genuinely did change, so the test is not
    # passing because both sides are identical.
    assert not np.allclose(up["b"], down["b"])


def test_pair_records_on_no_frames_is_empty_not_an_error():
    records = pair_records([])
    assert len(records["r"]) == 0
    assert len(records["delta"]) == 0


# --- reductions --------------------------------------------------------------


def test_radial_profile_partitions_the_records():
    """Every pair lands in exactly one shell and the shares sum to one."""
    frames = [_dimer([0.0, 0.0, d]) for d in (4.0, 5.5, 8.0, 12.0)]
    rows = radial_profile(pair_records(frames), edges=(3.4, 5.0, 6.0, 9.0, 15.0))

    assert sum(row["count"] for row in rows) == 4
    assert all(row["count"] == 1 for row in rows)
    assert sum(row["delta_sq_share"] for row in rows) == pytest.approx(1.0)


def test_radial_profile_reports_kcal_not_ev():
    """Unit slips here would silently rescale every conclusion about Delta."""
    frames = [_dimer([0.0, 0.0, 4.0])]
    records = pair_records(frames)
    rows = radial_profile(records, edges=(3.4, 5.0))
    assert rows[0]["delta_rms"] == pytest.approx(abs(records["delta"][0]) * EV_TO_KCAL)


def test_motifs_separate_the_canonical_contacts():
    cofacial = pair_records([_dimer([0.0, 0.0, 3.9])])
    t_shaped = pair_records([_dimer([0.0, 0.0, 5.0], u_b=[1.0, 0.0, 0.0])])
    # Normals along z, separation offset laterally: the slipped-parallel contact.
    displaced = pair_records([_dimer([1.6, 0.0, 3.5])])
    far = pair_records([_dimer([4.5, 0.0, 3.5])])

    assert motif_masks(cofacial)[COFACIAL][0]
    assert motif_masks(t_shaped)[T_SHAPED][0]
    assert motif_masks(displaced)[PARALLEL_DISPLACED][0]
    assert motif_masks(far)[FAR_SLIPPED][0]


def test_the_cacelli_minima_classify_as_their_own_motifs():
    """The regression this cut exists for.

    An earlier version cut parallel-displaced as ``a_hi < 0.6``. The real PD
    minimum has ``a_hi = 0.909``, so it was silently classified as *cofacial* and
    the PD bucket collected far-slipped pairs instead -- which inverted the
    conclusion about which motif was undersampled.
    """
    from asmcmc.data_preparation.proposals import motif_reference

    reference = motif_reference()

    # PD: stacking height 3.5, slip 1.6 -> must be parallel-displaced, not cofacial.
    pd = pair_records([_dimer([1.6, 0.0, 3.5])])
    masks = motif_masks(pd)
    assert masks[PARALLEL_DISPLACED][0]
    assert not masks[COFACIAL][0]

    r, _, a_i, _ = reference["parallel_displaced"]
    assert r * np.sqrt(1 - a_i**2) == pytest.approx(1.6, abs=0.02)
    assert a_i > 0.6, "the old a_hi<0.6 cut could never have selected this"

    cofacial_r = reference["cofacial"][0]
    assert motif_masks(pair_records([_dimer([0.0, 0.0, cofacial_r])]))[COFACIAL][0]


def test_stack_coordinates_decompose_the_separation():
    """height along the normal, slip perpendicular to it, Pythagoras in between."""
    records = pair_records([_dimer([1.6, 0.0, 3.5])])
    height, slip = stack_coordinates(records)

    assert height[0] == pytest.approx(3.5, abs=1e-6)
    assert slip[0] == pytest.approx(1.6, abs=1e-6)
    assert np.hypot(height[0], slip[0]) == pytest.approx(records["r"][0])


def test_motif_buckets_are_disjoint():
    """A pair must not be counted under two motifs, or the per-motif shares of a
    fixed budget stop adding up."""
    frames = [
        _dimer(offset)
        for offset in ([0.0, 0.0, 3.9], [1.6, 0.0, 3.5], [4.5, 0.0, 3.5], [0.0, 0.0, 5.5])
    ]
    frames.append(_dimer([0.0, 0.0, 5.0], u_b=[1.0, 0.0, 0.0]))
    masks = np.stack(list(motif_masks(pair_records(frames)).values()))
    assert masks.sum(axis=0).max() <= 1


def test_motifs_exclude_pairs_outside_the_well():
    """A cofacial geometry at 12 A is not a cofacial *contact* -- counting it
    would dilute the per-motif energies toward zero."""
    far = pair_records([_dimer([0.0, 0.0, 12.0])])
    assert not any(mask.any() for mask in motif_masks(far).values())


def test_three_body_compactness_uses_the_widest_pair(campaign):
    """One distant molecule makes a trimer a dimer plus a spectator however
    tight the other two are, so compactness is max_pair_r, not min."""
    frames, _ = campaign
    records = three_body_records(frames)

    assert len(records) == sum(f.info["n_molecules"] == 3 for f in frames)
    assert np.all(records.max_pair_r >= records.min_pair_r)
    assert np.array_equal(records.compact(cutoff=7.0), records.max_pair_r < 7.0)

    # A spread trimer is excluded even when its closest pair is in contact.
    tight_but_spread = (records.min_pair_r < 7.0) & (records.max_pair_r >= 7.0)
    if tight_but_spread.any():
        assert not records.compact(cutoff=7.0)[tight_but_spread].any()


def test_three_body_records_are_kcal(campaign):
    frames, _ = campaign
    records = three_body_records(frames)
    trimers = [f for f in frames if f.info["n_molecules"] == 3]
    expected = float(trimers[0].info["three_body_energy"]) * EV_TO_KCAL
    assert records.three_body[0] == pytest.approx(expected)


# --- QA ----------------------------------------------------------------------


def test_qa_passes_on_a_clean_campaign(campaign):
    frames, config = campaign
    report = qa_report(frames, config)

    assert report.ok, report.problems
    assert report.n_frames == len(frames)
    assert report.n_pairs == len(pair_records(frames)["r"])
    assert report.unique_ids
    assert report.three_body_max_error == 0.0
    assert report.pair_algebra_max_error == 0.0


def test_qa_flags_a_hard_core_violation(campaign):
    """The clash filter is the one guarantee a generated frame cannot self-check
    after the fact, so QA has to be able to see a breach."""
    frames, config = campaign
    config = {**config, "settings": {**config["settings"], "min_atom_distance": 99.0}}

    report = qa_report(frames, config)
    assert not report.ok
    assert report.hard_core_violations == len(frames)
    assert any("hard-core" in p for p in report.problems)


def test_qa_flags_duplicate_config_ids(campaign):
    frames, config = campaign
    frames[1].info["config_index"] = frames[0].info["config_index"]
    frames[1].info["shard"] = frames[0].info["shard"]

    report = qa_report(frames, config)
    assert not report.ok
    assert not report.unique_ids
    assert any("duplicate (shard, config_index)" in p for p in report.problems)


def test_qa_flags_broken_three_body_algebra(campaign):
    frames, config = campaign
    trimer = next(f for f in frames if f.info["n_molecules"] == 3)
    trimer.info["three_body_energy"] = float(trimer.info["three_body_energy"]) + 0.5

    report = qa_report(frames, config)
    assert not report.ok
    assert report.three_body_max_error == pytest.approx(0.5)
    assert any("three-body algebra" in p for p in report.problems)


def test_qa_flags_a_baseline_that_does_not_rebuild(campaign):
    """If the stored cluster baseline disagrees with a per-pair rebuild, then
    pair_records cannot attribute it and the Delta column is wrong."""
    frames, config = campaign
    frames[0].info["gbq_interaction_energy"] = float(frames[0].info["gbq_interaction_energy"]) + 1.0

    report = qa_report(frames, config)
    assert not report.ok
    assert any("GBQ baseline" in p for p in report.problems)


def test_qa_flags_a_varying_monomer_reference_in_a_rigid_run(campaign):
    """Rigid monomers are all rotated copies of one geometry, so the reference
    is a single constant; a spread means the run was not what it claims."""
    frames, config = campaign
    frames[0].info["monomer_energies"] = np.asarray(frames[0].info["monomer_energies"]) + 0.25

    report = qa_report(frames, config)
    assert not report.ok
    assert any("monomer reference" in p for p in report.problems)


def test_qa_records_disjoint_shard_seeds(campaign):
    frames, config = campaign
    report = qa_report(frames, config)
    assert all(len(seeds) == 1 for seeds in report.shard_seeds.values())


# --- the hard-core proposal filter -------------------------------------------


def test_pass_rate_rises_with_separation_and_saturates():
    """Far apart, nothing can clash; close in, most random orientations do."""
    radii = [3.4, 4.0, 5.0, 6.0, 8.0]
    rates = hard_core_pass_rate(radii, n_trials=400, seed=1)

    assert np.all(np.diff(rates) >= -0.02), rates  # monotone up to sampling noise
    assert rates[0] < 0.15
    assert rates[-1] == pytest.approx(1.0)


def test_cofacial_seeding_beats_random_where_it_matters():
    """The point of the whole measurement: at close approach the *proposal*, not
    the filter, is what limits short-range coverage."""
    random_rate = hard_core_pass_rate([3.6], n_trials=600, seed=2)[0]
    cofacial_rate = hard_core_pass_rate([3.6], orientations="cofacial", n_trials=600, seed=2)[0]

    assert random_rate < 0.10
    assert cofacial_rate > 0.60
    assert cofacial_rate > 10 * random_rate


def test_a_vanishing_threshold_admits_everything():
    """Isolates the filter from the geometry: with no hard core nothing is
    rejected, so a low pass rate can only come from the threshold."""
    rates = hard_core_pass_rate([3.4, 4.0], n_trials=200, threshold=0.0, seed=3)
    assert np.all(rates == 1.0)


def test_pass_rate_is_reproducible_and_shaped_like_its_input():
    first = hard_core_pass_rate([4.0, 5.0], n_trials=200, seed=7)
    second = hard_core_pass_rate([4.0, 5.0], n_trials=200, seed=7)
    assert np.array_equal(first, second)
    assert first.shape == (2,)
    assert hard_core_pass_rate(4.0, n_trials=100, seed=7).shape == (1,)


def test_unknown_orientation_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown orientations"):
        hard_core_pass_rate([4.0], orientations="herringbone")


# --- proposal density provenance ---------------------------------------------


def test_pair_records_marks_only_the_seeded_pair(campaign):
    """log_q is the density of the whole cluster, and the mixture proposes only
    the (0,1) pair -- so a trimer's other two pairs must not be marked usable."""
    frames, _ = campaign
    for frame in frames:
        frame.info["log_q"] = -7.5

    records = pair_records(frames)
    n_dimers = sum(f.info["n_molecules"] == 2 for f in frames)
    n_trimers = sum(f.info["n_molecules"] == 3 for f in frames)

    # one seeded pair per cluster, whatever its size
    assert records["seeded"].sum() == n_dimers + n_trimers
    assert np.all(records["log_q"] == -7.5)
    # ... and the incidental trimer pairs are exactly the rest
    assert (~records["seeded"]).sum() == 2 * n_trimers


def test_log_q_is_nan_for_a_campaign_that_recorded_none(campaign):
    """A uniform campaign has no proposal density; it must read as missing
    rather than as some default that would silently weight wrong."""
    frames, _ = campaign
    records = pair_records(frames)
    assert np.all(np.isnan(records["log_q"]))
    assert not records["seeded"].any()


def test_frame_records_are_per_frame_not_per_pair(campaign):
    frames, _ = campaign
    for i, frame in enumerate(frames):
        frame.info["log_q"] = -float(i)
        frame.info["proposal_component"] = "stacked_tight"

    records = frame_records(frames)
    assert len(records["log_q"]) == len(frames)
    assert np.array_equal(records["log_q"], -np.arange(len(frames), dtype=float))
    assert set(records["proposal_component"]) == {"stacked_tight"}


def test_weight_diagnostics_are_perfect_for_a_flat_proposal():
    """Identical densities mean identical weights: nothing is lost reweighting."""
    stats = weight_diagnostics(np.full(500, -3.2))
    assert stats["ess_fraction"] == pytest.approx(1.0)
    assert stats["p99_over_p50"] == pytest.approx(1.0)
    assert stats["log_q_range"] == pytest.approx(0.0)


def test_ess_catches_a_lone_dominating_weight_that_the_robust_ratio_misses():
    """Both statistics are reported because neither alone is sufficient.

    One configuration proposed far more rarely than the rest dominates the
    reweighted sample. ESS sees it; ``p99_over_p50`` is a robust quantile ratio
    and deliberately does not, which is exactly why it is not reported alone.
    """
    log_q = np.concatenate([np.full(999, 0.0), [-25.0]])
    stats = weight_diagnostics(log_q)

    assert stats["ess_fraction"] < 0.01
    assert stats["log_q_range"] == pytest.approx(25.0)
    assert stats["p99_over_p50"] == pytest.approx(1.0)


def test_weight_diagnostics_survive_large_log_q():
    """exp(-log_q) overflows without the shift; the ESS must stay finite."""
    stats = weight_diagnostics(np.linspace(-800, -600, 400))
    assert np.isfinite(stats["ess_fraction"])
    assert 0.0 < stats["ess_fraction"] <= 1.0


def test_weight_diagnostics_on_no_density_is_nan():
    stats = weight_diagnostics(np.full(10, np.nan))
    assert stats["n"] == 0
    assert np.isnan(stats["ess_fraction"])


# --- duplicate detection -----------------------------------------------------


def test_a_genuine_duplicate_is_caught(campaign):
    frames, config = campaign
    # The frame itself, not a .copy() -- copying drops the SinglePointCalculator
    # and qa_report's energy check would raise before reaching the duplicate.
    frames.append(frames[0])
    report = qa_report(frames, config)
    assert report.duplicate_signatures == 1
    assert any("duplicate configurations" in p for p in report.problems)


def test_matching_centre_distance_alone_is_not_a_duplicate():
    """The regression this fingerprint exists for.

    A distance-only signature is a *single number* for a dimer, so two unrelated
    configurations collide at 1e-4 A with high probability once motif sampling
    concentrates separations into a narrow band -- and the first motif campaign
    duly reported a phantom duplicate. Same separation, different orientations,
    is a different configuration.
    """
    same_distance = 4.0
    a = _dimer([0.0, 0.0, same_distance])
    b = _dimer([0.0, 0.0, same_distance], u_b=[1.0, 0.0, 0.0])
    assert np.isclose(
        np.linalg.norm(np.asarray(a.info["molecular_com"])[1]),
        np.linalg.norm(np.asarray(b.info["molecular_com"])[1]),
    )
    assert _configuration_signature(a) != _configuration_signature(b)


def test_the_signature_ignores_rigid_motion_and_relabelling():
    """It must still catch a true duplicate that was merely rotated."""
    frame = _dimer([0.0, 0.0, 4.0], u_a=[0.0, 0.0, 1.0], u_b=[1.0, 0.0, 0.0])

    rotated = frame.copy()
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated.info["molecular_com"] = np.asarray(frame.info["molecular_com"]) @ turn.T
    rotated.info["or_vec"] = np.asarray(frame.info["or_vec"]) @ turn.T

    assert _configuration_signature(frame) == _configuration_signature(rotated)


def test_reweighting_ess_is_perfect_when_campaign_matches_target():
    """Nothing to correct means nothing is lost."""
    frames = [_dimer([0.0, 0.0, d]) for d in np.linspace(4.0, 14.0, 200)]
    records = pair_records(frames)
    assert reweighting_ess(records, records["r"]) == pytest.approx(1.0, abs=0.02)


def test_reweighting_ess_falls_when_the_target_sits_where_sampling_is_thin():
    """A campaign concentrated away from its target pays for the correction."""
    frames = [_dimer([0.0, 0.0, d]) for d in np.concatenate(
        [np.linspace(4.0, 5.0, 190), np.linspace(12.0, 14.0, 10)]
    )]
    records = pair_records(frames)
    target = np.linspace(12.0, 14.0, 500)  # all target mass where only 10 pairs sit
    assert reweighting_ess(records, target) < 0.2


def test_reweighting_ess_is_zero_without_overlapping_support():
    """The failure reweighting cannot fix: a region that was never sampled."""
    frames = [_dimer([0.0, 0.0, d]) for d in np.linspace(4.0, 5.0, 60)]
    records = pair_records(frames)
    assert reweighting_ess(records, np.linspace(12.0, 14.0, 200)) == 0.0


def test_weight_diagnostics_spread_survives_a_bottom_heavy_proposal():
    """Most weights at the maximum is exactly where max-over-median reads 1.0 and
    hides a real spread; the p99/p50 form must not."""
    log_q = np.concatenate([np.full(900, 0.0), np.full(100, -9.0)])
    stats = weight_diagnostics(log_q)
    assert stats["p99_over_p50"] > 1e3
    assert stats["log_q_range"] == pytest.approx(9.0)
    assert stats["ess_fraction"] < 0.2
