"""Geometry and bookkeeping for the UMA-labelled cluster dataset.

No MLIP anywhere in here: every test either builds geometry or drives the
generator with a stub calculator, so the suite stays runnable from a fresh
clone without fairchem, a model download, or a GPU.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
from ase.io import read

from asmcmc.delta_learning.dataset import (
    CONFIG_NAME,
    orthonormal_basis,
    rotation_to_normal,
    SamplingSettings,
    build_reference_benzene,
    config_rng,
    dataset_frames,
    energy_decomposition,
    gbq_baseline,
    _shard_sizes,
    generate_shard,
    generate_dataset,
    make_dimer,
    molecule_indices,
    shard_count,
    shard_path,
    subset_atoms,
)
from asmcmc.mc.potentials import CACELLI_POTENTIAL


@pytest.fixture(scope="module")
def reference():
    return build_reference_benzene()


# --- radial sampling ---------------------------------------------------------

def test_volume_uniform_is_flat_in_r_cubed():
    """The chosen sampling: uniform in r^3, so configuration density per unit
    volume is flat and shell occupancy matches a bulk pair census."""
    s = SamplingSettings(radial_sampling="volume-uniform")
    rng = np.random.default_rng(0)
    r = s.radial().sample(rng, 40_000)

    assert r.min() >= s.min_com_distance
    assert r.max() <= s.max_com_distance

    # Analytic shell fractions: (hi^3 - lo^3) / (max^3 - min^3)
    span = s.max_com_distance**3 - s.min_com_distance**3
    edges = [s.min_com_distance, 6.0, 9.0, s.max_com_distance]
    for lo, hi in zip(edges[:-1], edges[1:]):
        expected = (hi**3 - lo**3) / span
        got = np.mean((r >= lo) & (r < hi))
        assert got == pytest.approx(expected, abs=0.01), f"shell {lo}-{hi}"


def test_default_range_covers_the_uma_horizon_and_the_hard_core():
    """min/max_atom_distance -- not max_com_distance -- are what shape a
    campaign's label distribution (see CLAUDE.md's UMA HORIZON and TRIMER
    notes): below ~3 A the pair is a hard-core clash UMA scores strongly
    repulsive, and past its ~6 A minimum-atom-atom horizon the label is a
    truncation artifact (Delta = -E_GBQ exactly). The window sits inside both.
    """
    s = SamplingSettings()
    assert 2.4 < s.min_atom_distance < s.max_atom_distance < 6.0


def test_mixture_sampling_concentrates_on_the_wells():
    """The alternative shape stays reachable, and actually differs."""
    s = SamplingSettings(radial_sampling="mixture", compact_probability=0.7)
    rng = np.random.default_rng(0)
    r = s.radial().sample(rng, 20_000)
    assert np.mean(r < 6.0) > 0.65


# --- cluster construction ----------------------------------------------------

def test_cluster_shape_and_labelling(reference):
    cluster = make_dimer(reference, config_rng(3, 0), SamplingSettings())

    assert len(cluster) == 24
    assert int(cluster.info["n_molecules"]) == 2
    assert not cluster.pbc.any()
    assert np.allclose(cluster.cell, 0.0)
    assert cluster.info["charge"] == 0 and cluster.info["spin"] == 1

    idx = molecule_indices(cluster)
    assert len(idx) == 2
    for block in idx:
        assert len(block) == 12
        # contiguous, so subset_atoms and molecule_id agree with slicing
        assert np.array_equal(block, np.arange(block[0], block[0] + 12))


def test_no_cluster_violates_the_hard_core(reference):
    """min_atom_distance/max_atom_distance bound the realised separation; they
    are rejection criteria, not suggestions."""
    s = SamplingSettings(min_atom_distance=2.5, max_atom_distance=4.0)
    for k in range(25):
        cluster = make_dimer(reference, config_rng(5, k), s)
        pos = cluster.get_positions()
        ids = cluster.arrays["molecule_id"]
        d = np.linalg.norm(
            pos[ids == 0][:, None, :] - pos[ids == 1][None, :, :], axis=-1
        )
        assert s.min_atom_distance - 1e-9 <= d.min() <= s.max_atom_distance + 1e-9


def test_rigid_monomers_are_the_reference_up_to_rotation(reference):
    """Why monomers are rigid: one monomer energy is valid for every molecule.

    That is only true if each molecule is a *rigid rotation* of the reference,
    which this pins via the (rotation-invariant) sorted internal distances.
    """
    cluster = make_dimer(reference, config_rng(7, 0), SamplingSettings())
    ref_d = np.sort(reference.get_all_distances().ravel())
    for block in molecule_indices(cluster):
        mol = cluster[block]
        np.testing.assert_allclose(
            np.sort(mol.get_all_distances().ravel()), ref_d, atol=1e-9
        )


def test_config_rng_is_reproducible_and_index_dependent(reference):
    """Per-configuration seeding is what makes resume exact."""
    a = make_dimer(reference, config_rng(11, 4), SamplingSettings())
    b = make_dimer(reference, config_rng(11, 4), SamplingSettings())
    c = make_dimer(reference, config_rng(11, 5), SamplingSettings())
    np.testing.assert_allclose(a.get_positions(), b.get_positions())
    assert not np.allclose(a.get_positions(), c.get_positions())


def test_retry_salt_changes_the_stream(reference):
    """A failed placement must not be retried from an identical stream."""
    a = make_dimer(reference, config_rng(11, 4, 0), SamplingSettings())
    b = make_dimer(reference, config_rng(11, 4, 1), SamplingSettings())
    assert not np.allclose(a.get_positions(), b.get_positions())


# --- energy decomposition ----------------------------------------------------

def test_decomposition_algebra_is_exact(reference, make_stub_calculator):
    """interaction = E_cluster - sum E_mono."""
    calc = make_stub_calculator()
    cluster = make_dimer(reference, config_rng(13, 0), SamplingSettings())
    e_cluster = -0.1 * len(cluster)

    out = energy_decomposition(cluster, calc, e_cluster, mode="monomers")

    e_mono = out["monomer_energies"]
    assert e_mono.shape == (2,)
    assert out["interaction_energy"] == pytest.approx(e_cluster - e_mono.sum())


def test_rigid_monomer_energy_skips_the_per_molecule_calls(reference, make_stub_calculator):
    """The cost saving is real: passing the constant makes zero monomer calls."""
    cluster = make_dimer(reference, config_rng(13, 1), SamplingSettings())

    calc = make_stub_calculator()
    energy_decomposition(cluster, calc, -1.0, mode="monomers")
    assert calc.n_calls == 2

    calc = make_stub_calculator()
    out = energy_decomposition(
        cluster, calc, -1.0, mode="monomers", rigid_monomer_energy=-1.2
    )
    assert calc.n_calls == 0
    np.testing.assert_allclose(out["monomer_energies"], [-1.2, -1.2])


def test_decomposition_none_is_empty(reference, make_stub_calculator):
    cluster = make_dimer(reference, config_rng(13, 2), SamplingSettings())
    assert energy_decomposition(cluster, make_stub_calculator(), -1.0, mode="none") == {}


def test_subset_atoms_selects_whole_molecules(reference):
    cluster = make_dimer(reference, config_rng(13, 3), SamplingSettings())
    sub = subset_atoms(cluster, [0])
    assert len(sub) == 12
    assert sub.info["charge"] == 0 and sub.info["spin"] == 1
    assert not sub.pbc.any()


# --- the Delta-learning baseline --------------------------------------------

def test_gbq_baseline_matches_a_direct_pair_energy(reference):
    """A hand-built cofacial dimer at 3.9 A: the stored baseline must equal a
    direct CACELLI_POTENTIAL call on the same two discs."""
    a = reference.copy()
    b = reference.copy()
    b.translate([0.0, 0.0, 3.9])
    dimer = a + b
    dimer.set_pbc(False)
    dimer.set_cell(np.zeros((3, 3)))
    dimer.arrays["molecule_id"] = np.array([0] * 12 + [1] * 12, dtype=np.int32)

    out = gbq_baseline(dimer)
    normals = out["or_vec"]
    assert normals.shape == (2, 3)

    direct = CACELLI_POTENTIAL.pair_energy(
        normals[:1], normals[1:], np.array([[0.0, 0.0, 3.9]])
    )
    assert out["gbq_interaction_energy"] == pytest.approx(float(np.sum(direct)))
    # g2 benzene lies in the xy-plane, so the stacked pair is face-to-face and bound
    assert out["gbq_interaction_energy"] < 0.0


def test_baseline_names_the_potential_it_used():
    """Provenance has to distinguish Cacelli from the known-broken refit."""
    ref = build_reference_benzene()
    a, b = ref.copy(), ref.copy()
    b.translate([0.0, 0.0, 5.0])
    dimer = a + b
    dimer.arrays["molecule_id"] = np.array([0] * 12 + [1] * 12, dtype=np.int32)
    name = gbq_baseline(dimer)["gbq_potential"]
    assert name and name != "data"
    assert name == CACELLI_POTENTIAL.name


# --- sharding, incremental writes, resume ------------------------------------

def _run(tmp_path, n_configs, **kw):
    """Drive a campaign through the in-process path.

    ``n_shards=1`` is deliberate, not a simplification: ``generate_dataset`` dispatches
    multi-shard runs through a **spawned** ProcessPoolExecutor, and a
    monkeypatched ``load_uma_calculator`` does not survive that boundary -- the
    child re-imports the real module and would quietly load real UMA, turning
    these into slow MLIP tests. The shard plan the pool would execute is
    tested directly in ``test_shard_plan_*`` below.
    """
    return generate_dataset(
        n_configs=n_configs,
        out_dir=tmp_path,
        n_shards=1,
        decomposition=kw.pop("decomposition", "monomers"),
        max_workers=1,
        **kw,
    )


def test_shard_plan_covers_the_request_exactly():
    """Every configuration is assigned to exactly one shard, sizes balanced."""
    for n_configs, n_shards in [(7, 3), (500, 8), (10, 10), (1, 4)]:
        sizes = _shard_sizes(n_configs, min(n_shards, n_configs))
        assert sum(sizes) == n_configs
        assert max(sizes) - min(sizes) <= 1


def test_shard_plan_gives_shards_disjoint_seeds():
    """seed0 + k, so no two shards draw the same configurations."""
    seed0, n_shards = 20260731, 8
    seeds = [seed0 + k for k in range(n_shards)]
    assert len(set(seeds)) == n_shards


def test_frames_are_written_incrementally(stub_uma, tmp_path):
    """The old script buffered everything and wrote once at the end, so a crash
    at the last configuration lost the whole run. Frames must land on disk as
    they are produced."""
    generate_shard(
        out_dir=tmp_path, shard=0, n_configs=6, seed=1,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=2,
    )
    path = shard_path(tmp_path, 0)
    assert path.exists()
    assert shard_count(path) == 6


def test_rerun_is_idempotent(stub_uma, tmp_path):
    """A completed shard is skipped, so re-running finishes an interrupted
    campaign instead of duplicating it."""
    first = _run(tmp_path, 6)
    assert sum(r["written"] for r in first) == 6

    second = _run(tmp_path, 6)
    assert all(r["skipped"] for r in second)
    assert sum(r["written"] for r in second) == 0
    assert len(dataset_frames(tmp_path)) == 6


def test_resume_completes_a_partial_shard(stub_uma, tmp_path):
    """Half a shard on disk, then a re-run to the full target: it appends the
    remainder and never repeats a config_index."""
    generate_shard(
        out_dir=tmp_path, shard=0, n_configs=3, seed=1,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=1,
    )
    assert shard_count(shard_path(tmp_path, 0)) == 3

    res = generate_shard(
        out_dir=tmp_path, shard=0, n_configs=8, seed=1,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=1,
    )
    assert res["written"] == 5 and res["total"] == 8

    frames = read(shard_path(tmp_path, 0), index=":")
    indices = [int(f.info["config_index"]) for f in frames]
    assert indices == list(range(8))


def test_resume_reproduces_the_uninterrupted_run(stub_uma, tmp_path):
    """Per-config seeding means an interrupted campaign is byte-identical to
    one that ran straight through -- the property a shard-wide RNG loses."""
    kw = dict(
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=1,
    )
    generate_shard(out_dir=tmp_path, shard=0, n_configs=2, seed=9, **kw)
    generate_shard(out_dir=tmp_path, shard=0, n_configs=5, seed=9, **kw)
    resumed = read(shard_path(tmp_path, 0), index=":")

    straight_dir = tmp_path / "straight"
    straight_dir.mkdir()
    generate_shard(out_dir=straight_dir, shard=0, n_configs=5, seed=9, **kw)
    whole = read(shard_path(straight_dir, 0), index=":")

    assert len(resumed) == len(whole) == 5
    for a, b in zip(resumed, whole):
        np.testing.assert_allclose(a.get_positions(), b.get_positions(), atol=1e-12)


def test_different_shard_seeds_give_different_configurations(stub_uma, tmp_path):
    kw = dict(
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=4,
    )
    generate_shard(out_dir=tmp_path, shard=0, n_configs=2, seed=100, **kw)
    generate_shard(out_dir=tmp_path, shard=1, n_configs=2, seed=101, **kw)
    a = read(shard_path(tmp_path, 0), index=":")
    b = read(shard_path(tmp_path, 1), index=":")
    assert not np.allclose(a[0].get_positions(), b[0].get_positions())


def test_shard_count_tolerates_a_torn_final_frame(stub_uma, tmp_path):
    """A run killed mid-write leaves a partial frame; resume must drop only
    that frame, not the whole shard."""
    generate_shard(
        out_dir=tmp_path, shard=0, n_configs=4, seed=1,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=4,
    )
    path = shard_path(tmp_path, 0)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-5]) + "\n")  # truncate mid-frame

    assert shard_count(path) == 3


def test_campaign_stamps_a_config(stub_uma, tmp_path):
    _run(tmp_path, 4)
    cfg = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert cfg["n_configs"] == 4
    assert cfg["settings"]["max_com_distance"] == SamplingSettings().max_com_distance
    assert cfg["settings"]["radial_sampling"] == "volume-uniform"


def test_unknown_radial_sampling_is_rejected(stub_uma, tmp_path):
    with pytest.raises(ValueError, match="radial_sampling"):
        generate_dataset(
            n_configs=2,
            out_dir=tmp_path,
            settings=SamplingSettings(radial_sampling="nonsense"),
            max_workers=1,
        )


def test_saved_frames_carry_the_training_targets(stub_uma, tmp_path):
    """What a fit actually reads back: both energies, orientations, geometry."""
    _run(tmp_path, 4)
    frames = dataset_frames(tmp_path)
    assert len(frames) == 4
    for f in frames:
        assert np.asarray(f.info["or_vec"]).shape == (2, 3)
        assert np.asarray(f.info["molecular_com"]).shape == (2, 3)
        assert np.isfinite(f.info["interaction_energy"])
        assert np.isfinite(f.info["gbq_interaction_energy"])
        assert f.info["rigid_monomers"]
        # the Delta-learning target is formable from the frame alone
        delta = f.info["interaction_energy"] - f.info["gbq_interaction_energy"]
        assert np.isfinite(delta)


def test_progress_queue_counts_every_configuration(stub_uma, tmp_path):
    """The bar is only useful if it is honest: the posted total must equal the
    configurations actually generated, or a long run reads as stalled or as
    finished early."""
    import queue

    reported = queue.Queue()
    result = generate_shard(
        out_dir=str(tmp_path), shard=0, n_configs=7, seed=3,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=2,
        progress_queue=reported,
    )

    total = 0
    while not reported.empty():
        total += reported.get_nowait()
    assert result["written"] == 7
    assert total == 7


def test_progress_queue_reports_frames_already_on_disk(stub_uma, tmp_path):
    """A resumed campaign must start its bar where the data does. A completed
    shard returns immediately, so it has to post its count before returning."""
    import queue

    common = dict(
        out_dir=str(tmp_path), shard=0, seed=3,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", flush_every=2,
    )
    generate_shard(n_configs=5, **common)

    resumed = queue.Queue()
    result = generate_shard(n_configs=5, progress_queue=resumed, **common)

    total = 0
    while not resumed.empty():
        total += resumed.get_nowait()
    assert result["skipped"]
    assert total == 5, "a skipped shard still has to account for its frames"


def test_drain_returns_everything_queued_without_blocking():
    import queue

    from asmcmc.delta_learning.dataset import _drain

    q = queue.Queue()
    for n in (1, 1, 3, 5):
        q.put(n)
    assert _drain(q) == 10
    assert _drain(q) == 0  # empty, and must not hang


@pytest.fixture
def rotation_rng():
    return np.random.default_rng(20260811)


# --- rotations ---------------------------------------------------------------


def test_orthonormal_basis_is_orthonormal(rotation_rng):
    u = rotation_rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    e1, e2 = orthonormal_basis(u)

    for a, b in [(e1, e1), (e2, e2)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 1.0)
    for a, b in [(e1, e2), (e1, u), (e2, u)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 0.0, atol=1e-12)


def test_rotation_carries_the_reference_normal_onto_the_target(rotation_rng):
    targets = rotation_rng.normal(size=(300, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    spin = rotation_rng.uniform(0, 2 * np.pi, size=300)

    rotations = rotation_to_normal(targets, spin)
    mapped = np.einsum("nij,j->ni", rotations, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(mapped, targets, atol=1e-10)
    assert np.allclose(np.linalg.det(rotations), 1.0)


def test_rotation_handles_the_antipodal_target():
    """cos = -1 is where the Rodrigues 1/(1+cos) form blows up."""
    rotations = rotation_to_normal(np.array([[0.0, 0.0, -1.0]]), np.array([0.0]))
    mapped = rotations[0] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(mapped, [0.0, 0.0, -1.0], atol=1e-10)
    assert np.linalg.det(rotations[0]) == pytest.approx(1.0)


def test_spin_leaves_the_normal_alone_but_moves_the_molecule():
    """Spin is a real atomistic degree of freedom the CG coordinates cannot see."""
    target = np.array([[0.0, 0.0, 1.0]])
    a = rotation_to_normal(target, np.array([0.0]))[0]
    b = rotation_to_normal(target, np.array([0.7]))[0]

    positions = build_reference_benzene().get_positions()
    assert not np.allclose(positions @ a.T, positions @ b.T)
    assert np.allclose(a @ np.array([0, 0, 1.0]), b @ np.array([0, 0, 1.0]), atol=1e-12)
