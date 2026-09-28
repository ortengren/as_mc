"""Unpacking, QA, and reduction over a cluster campaign.

No MLIP: campaigns are driven through the generator with the same stub
calculator the dataset tests use (see ``conftest.py``), so the suite stays runnable from a
fresh clone without fairchem or a GPU.
"""

import numpy as np
import pytest

from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.units import EV_TO_KCAL
from asmcmc.delta_learning.dataset_analysis import (
    COFACIAL,
    _configuration_signature,
    FAR_SLIPPED,
    PARALLEL_DISPLACED,
    T_SHAPED,
    gb_invariants,
    load_campaign,
    motif_masks,
    pair_records,
    qa_report,
    radial_profile,
    stack_coordinates,
)
from asmcmc.delta_learning.dataset import build_reference_benzene, generate_dataset



@pytest.fixture
def campaign(tmp_path, stub_uma):
    """A small campaign of dimers, for exercising qa_report/pair_records."""
    generate_dataset(
        n_configs=12,
        out_dir=tmp_path,
        n_shards=1,
        decomposition="monomers",
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


def test_pair_records_unpacks_one_row_per_frame(campaign):
    frames, _ = campaign
    records = pair_records(frames)
    assert len(records["r"]) == len(frames)


def test_pair_records_uses_the_stored_mlip_label(campaign):
    """Pair labels must be the frame's own value, not re-derived."""
    frames, _ = campaign
    records = pair_records(frames)
    stored = np.array([float(f.info["interaction_energy"]) for f in frames])
    assert np.allclose(np.sort(records["e_uma"]), np.sort(stored))


def test_pair_records_on_a_hand_built_dimer():
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
    minimum has stacking height 3.5, slip 1.6 A -- a_hi = 0.909 -- so it was
    silently classified as *cofacial* and the PD bucket collected far-slipped
    pairs instead, which inverted the conclusion about which motif was
    undersampled.
    """
    pd = pair_records([_dimer([1.6, 0.0, 3.5])])
    masks = motif_masks(pd)
    assert masks[PARALLEL_DISPLACED][0]
    assert not masks[COFACIAL][0]

    a_hi = 3.5 / np.hypot(3.5, 1.6)
    assert a_hi > 0.6, "the old a_hi<0.6 cut could never have selected this"

    # The real cofacial minimum (r=3.90, a_hi=1.0) still classifies as cofacial.
    assert motif_masks(pair_records([_dimer([0.0, 0.0, 3.90])]))[COFACIAL][0]


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


# --- QA ----------------------------------------------------------------------


def test_qa_passes_on_a_clean_campaign(campaign):
    frames, config = campaign
    report = qa_report(frames, config)

    assert report.ok, report.problems
    assert report.n_frames == len(frames)
    assert report.n_pairs == len(pair_records(frames)["r"])
    assert report.unique_ids


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


def test_qa_flags_a_baseline_that_does_not_rebuild(campaign):
    """If the stored baseline disagrees with a per-pair rebuild, then
    pair_records's e_gbq column is wrong."""
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
    configurations collide at 1e-4 A with high probability once sampling
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


