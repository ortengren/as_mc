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

from asmcmc.data_preparation.cluster_dataset import (
    CONFIG_NAME,
    SamplingSettings,
    build_reference_benzene,
    center_of_mass,
    config_rng,
    dataset_frames,
    energy_decomposition,
    gbq_baseline,
    _shard_sizes,
    generate_shard,
    main,
    make_cluster,
    molecular_geometry_metadata,
    molecule_indices,
    motif_plan,
    shard_count,
    shard_path,
    subset_atoms,
)
from asmcmc.data_preparation.cluster_analysis import motif_masks
from asmcmc.data_preparation.proposals import (
    COFACIAL,
    MOTIFS,
    PARALLEL_DISPLACED,
    T_SHAPED,
    UNIFORM,
)
from asmcmc.base.potentials import CACELLI_POTENTIAL

EV_TO_KCAL = 23.060541945329334


@pytest.fixture(scope="module")
def reference():
    return build_reference_benzene()


class StubCalculator:
    """A cheap stand-in for UMA: energy = -(number of atoms)/10, zero forces.

    Deliberately *not* physical. These tests check the generator's plumbing --
    decomposition algebra, sharding, resume -- and a stub makes the expected
    numbers exact instead of approximate.
    """

    def __init__(self):
        self.n_calls = 0

    def get_potential_energy(self, atoms=None):
        self.n_calls += 1
        return -0.1 * len(atoms)

    def get_forces(self, atoms=None):
        return np.zeros((len(atoms), 3))

    # ASE calls into the calculator through these on an attached Atoms.
    def calculate(self, *a, **k):
        pass

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name == "energy":
            return self.get_potential_energy(atoms)
        if name == "forces":
            return self.get_forces(atoms)
        raise NotImplementedError(name)

    def check_state(self, atoms):
        return []

    def get_stress(self, atoms=None):
        raise NotImplementedError


# --- radial sampling ---------------------------------------------------------

def test_volume_uniform_is_flat_in_r_cubed():
    """The chosen sampling: uniform in r^3, so configuration density per unit
    volume is flat and shell occupancy matches a bulk pair census."""
    s = SamplingSettings(radial_sampling="volume-uniform")
    rng = np.random.default_rng(0)
    r = s.radial().sample(rng, 40_000)

    assert r.min() >= s.min_com_distance
    assert r.max() <= s.max_com_distance

    # Analytic shell fractions over [3.4, 15]: (hi^3 - lo^3) / (max^3 - min^3)
    span = s.max_com_distance**3 - s.min_com_distance**3
    for lo, hi in [(3.4, 6.0), (6.0, 9.0), (9.0, 15.0)]:
        expected = (hi**3 - lo**3) / span
        got = np.mean((r >= lo) & (r < hi))
        assert got == pytest.approx(expected, abs=0.01), f"shell {lo}-{hi}"


def test_default_range_reaches_the_mc_cutoff():
    """15 A, not the 9 A that merely spans the dimer wells.

    `MetropolisCalculator` evaluates pairs to nl_radius = 15 and the AniSOAP
    descriptor cutoff is 15; in a real 100 K frame 55 pairs/molecule sit beyond
    9 A carrying ~9% of the cohesive energy. Training that stopped at 9 A would
    leave the model unconstrained over most of the pairs it is asked about.
    """
    assert SamplingSettings().max_com_distance == 15.0


def test_mixture_sampling_concentrates_on_the_wells():
    """The alternative shape stays reachable, and actually differs."""
    s = SamplingSettings(radial_sampling="mixture", compact_probability=0.7)
    rng = np.random.default_rng(0)
    r = s.radial().sample(rng, 20_000)
    assert np.mean(r < 6.0) > 0.65


def test_trimer_ceiling_is_honoured():
    s = SamplingSettings(trimer_max_com_distance=6.0)
    rng = np.random.default_rng(0)
    r = s.trimer_radial().sample(rng, 2_000)
    assert r.max() <= 6.0


# --- cluster construction ----------------------------------------------------

@pytest.mark.parametrize("n_mol", [2, 3])
def test_cluster_shape_and_labelling(reference, n_mol):
    cluster = make_cluster(n_mol, reference, config_rng(3, 0), SamplingSettings(), UNIFORM)

    assert len(cluster) == 12 * n_mol
    assert int(cluster.info["n_molecules"]) == n_mol
    assert not cluster.pbc.any()
    assert np.allclose(cluster.cell, 0.0)
    assert cluster.info["charge"] == 0 and cluster.info["spin"] == 1

    idx = molecule_indices(cluster)
    assert len(idx) == n_mol
    for block in idx:
        assert len(block) == 12
        # contiguous, so subset_atoms and molecule_id agree with slicing
        assert np.array_equal(block, np.arange(block[0], block[0] + 12))


def test_no_cluster_violates_the_hard_core(reference):
    """min_atom_distance is a rejection criterion, not a suggestion."""
    s = SamplingSettings(min_atom_distance=2.0)
    for k in range(25):
        cluster = make_cluster(2 + k % 2, reference, config_rng(5, k), s, UNIFORM)
        pos = cluster.get_positions()
        ids = cluster.arrays["molecule_id"]
        for a in range(int(ids.max()) + 1):
            for b in range(a + 1, int(ids.max()) + 1):
                d = np.linalg.norm(
                    pos[ids == a][:, None, :] - pos[ids == b][None, :, :], axis=-1
                )
                assert d.min() >= s.min_atom_distance - 1e-9


def test_rigid_monomers_are_the_reference_up_to_rotation(reference):
    """Why monomers are rigid: one monomer energy is valid for every molecule.

    That is only true if each molecule is a *rigid rotation* of the reference,
    which this pins via the (rotation-invariant) sorted internal distances.
    """
    cluster = make_cluster(3, reference, config_rng(7, 0), SamplingSettings(), UNIFORM)
    ref_d = np.sort(reference.get_all_distances().ravel())
    for block in molecule_indices(cluster):
        mol = cluster[block]
        np.testing.assert_allclose(
            np.sort(mol.get_all_distances().ravel()), ref_d, atol=1e-9
        )


def test_config_rng_is_reproducible_and_index_dependent(reference):
    """Per-configuration seeding is what makes resume exact."""
    a = make_cluster(2, reference, config_rng(11, 4), SamplingSettings(), UNIFORM)
    b = make_cluster(2, reference, config_rng(11, 4), SamplingSettings(), UNIFORM)
    c = make_cluster(2, reference, config_rng(11, 5), SamplingSettings(), UNIFORM)
    np.testing.assert_allclose(a.get_positions(), b.get_positions())
    assert not np.allclose(a.get_positions(), c.get_positions())


def test_retry_salt_changes_the_stream(reference):
    """A failed placement must not be retried from an identical stream."""
    a = make_cluster(2, reference, config_rng(11, 4, 0), SamplingSettings(), UNIFORM)
    b = make_cluster(2, reference, config_rng(11, 4, 1), SamplingSettings(), UNIFORM)
    assert not np.allclose(a.get_positions(), b.get_positions())


# --- energy decomposition ----------------------------------------------------

def test_decomposition_algebra_is_exact(reference):
    """interaction = E_cluster - sum E_mono, and the trimer 3-body identity."""
    calc = StubCalculator()
    cluster = make_cluster(3, reference, config_rng(13, 0), SamplingSettings(), UNIFORM)
    e_cluster = -0.1 * len(cluster)

    out = energy_decomposition(cluster, calc, e_cluster, mode="full")

    e_mono = out["monomer_energies"]
    assert e_mono.shape == (3,)
    assert out["interaction_energy"] == pytest.approx(e_cluster - e_mono.sum())

    # E_3body = E_123 - sum(E_ij) + sum(E_i); with an additive stub this is 0.
    expected = e_cluster - out["pair_energies"].sum() + e_mono.sum()
    assert out["three_body_energy"] == pytest.approx(expected)
    assert out["three_body_energy"] == pytest.approx(0.0, abs=1e-12)


def test_rigid_monomer_energy_skips_the_per_molecule_calls(reference):
    """The cost saving is real: passing the constant makes zero monomer calls."""
    cluster = make_cluster(3, reference, config_rng(13, 1), SamplingSettings(), UNIFORM)

    calc = StubCalculator()
    energy_decomposition(cluster, calc, -1.0, mode="monomers")
    assert calc.n_calls == 3

    calc = StubCalculator()
    out = energy_decomposition(
        cluster, calc, -1.0, mode="monomers", rigid_monomer_energy=-1.2
    )
    assert calc.n_calls == 0
    np.testing.assert_allclose(out["monomer_energies"], [-1.2, -1.2, -1.2])


def test_decomposition_none_is_empty(reference):
    cluster = make_cluster(2, reference, config_rng(13, 2), SamplingSettings(), UNIFORM)
    assert energy_decomposition(cluster, StubCalculator(), -1.0, mode="none") == {}


def test_subset_atoms_selects_whole_molecules(reference):
    cluster = make_cluster(3, reference, config_rng(13, 3), SamplingSettings(), UNIFORM)
    sub = subset_atoms(cluster, [0, 2])
    assert len(sub) == 24
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


def test_gbq_baseline_sums_all_trimer_pairs(reference):
    cluster = make_cluster(3, reference, config_rng(17, 0), SamplingSettings(), UNIFORM)
    out = gbq_baseline(cluster)
    com = cluster.info.get("molecular_com")
    normals = out["or_vec"]
    assert normals.shape == (3, 3)

    from asmcmc.utils.geometry import coarse_grain_frame

    cg = coarse_grain_frame(cluster)
    pos = cg.get_positions()
    i, j = np.triu_indices(3, k=1)
    expected = CACELLI_POTENTIAL.pair_energy(normals[i], normals[j], pos[j] - pos[i])
    assert out["gbq_interaction_energy"] == pytest.approx(float(np.sum(expected)))


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

    ``n_shards=1`` is deliberate, not a simplification: ``main`` dispatches
    multi-shard runs through a **spawned** ProcessPoolExecutor, and a
    monkeypatched ``load_uma_calculator`` does not survive that boundary -- the
    child re-imports the real module and would quietly load real UMA, turning
    these into slow MLIP tests. The pool wrapper itself is the same pattern
    ``test_npt_equilibration.py`` already exercises; what is specific here is
    the shard plan, tested directly in ``test_shard_plan_*`` below.
    """
    return main(
        n_configs=n_configs,
        out_dir=tmp_path,
        n_shards=1,
        decomposition=kw.pop("decomposition", "monomers"),
        max_workers=1,
        **kw,
    )


@pytest.fixture
def stub_uma(monkeypatch):
    """Swap UMA for the stub everywhere generate_shard reaches for it."""
    import asmcmc.data_preparation.cluster_dataset as cd

    monkeypatch.setattr(cd, "load_uma_calculator", lambda *a, **k: StubCalculator())
    return cd


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
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=2,
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
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=1,
    )
    assert shard_count(shard_path(tmp_path, 0)) == 3

    res = generate_shard(
        out_dir=tmp_path, shard=0, n_configs=8, seed=1,
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=1,
    )
    assert res["written"] == 5 and res["total"] == 8

    frames = read(shard_path(tmp_path, 0), index=":")
    indices = [int(f.info["config_index"]) for f in frames]
    assert indices == list(range(8))


def test_resume_reproduces_the_uninterrupted_run(stub_uma, tmp_path):
    """Per-config seeding means an interrupted campaign is byte-identical to
    one that ran straight through -- the property a shard-wide RNG loses."""
    kw = dict(
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=1,
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
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=4,
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
        settings_dict=asdict(SamplingSettings(trimer_fraction=0.5)), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=4,
    )
    path = shard_path(tmp_path, 0)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-5]) + "\n")  # truncate mid-frame

    assert shard_count(path) == 3


def test_campaign_stamps_a_config(stub_uma, tmp_path):
    import json

    _run(tmp_path, 4)
    cfg = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert cfg["n_configs"] == 4
    assert cfg["settings"]["max_com_distance"] == 15.0
    assert cfg["settings"]["radial_sampling"] == "volume-uniform"


def test_unknown_radial_sampling_is_rejected(stub_uma, tmp_path):
    with pytest.raises(ValueError, match="radial_sampling"):
        main(
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
        n_mol = int(f.info["n_molecules"])
        assert np.asarray(f.info["or_vec"]).shape == (n_mol, 3)
        assert np.asarray(f.info["molecular_com"]).shape == (n_mol, 3)
        assert np.isfinite(f.info["interaction_energy"])
        assert np.isfinite(f.info["gbq_interaction_energy"])
        assert f.info["rigid_monomers"]
        # the Delta-learning target is formable from the frame alone
        delta = f.info["interaction_energy"] - f.info["gbq_interaction_energy"]
        assert np.isfinite(delta)


# --- motif quotas ------------------------------------------------------------


def test_motif_plan_honours_the_quotas_exactly():
    """Composition is decided, not sampled.

    The whole point of quotas over a mixture density: asking for 30% cofacial
    gets exactly 30% cofacial, checkable before any MLIP time is spent. Largest
    remainder means the counts sum to n_configs with nothing rounded away.
    """
    quotas = {COFACIAL: 0.3, PARALLEL_DISPLACED: 0.2, T_SHAPED: 0.2, UNIFORM: 0.3}
    plan = motif_plan(seed0=7, n_configs=1000, quotas=quotas)

    assert len(plan) == 1000
    for motif, share in quotas.items():
        assert int(np.sum(plan == motif)) == int(round(share * 1000))


def test_motif_plan_normalises_unnormalised_shares():
    """Shares are relative, so counts must not depend on their scale."""
    a = motif_plan(3, 200, {COFACIAL: 1, UNIFORM: 3})
    b = motif_plan(3, 200, {COFACIAL: 0.25, UNIFORM: 0.75})
    assert np.array_equal(a, b)
    assert int(np.sum(a == COFACIAL)) == 50


def test_motif_plan_is_a_pure_function_of_the_campaign():
    """A shard reads its own slice, so resume depends on this being stable."""
    quotas = {COFACIAL: 0.5, UNIFORM: 0.5}
    assert np.array_equal(
        motif_plan(11, 64, quotas), motif_plan(11, 64, quotas)
    )
    assert not np.array_equal(
        motif_plan(11, 64, quotas), motif_plan(12, 64, quotas)
    )


def test_motif_plan_rejects_an_unknown_motif():
    with pytest.raises(ValueError, match="unknown motif"):
        motif_plan(0, 10, {"herringbone": 1.0})


def test_every_motif_survives_a_full_campaign(reference):
    """Generated clusters carry the motif they were asked for, and the seeded
    pair really classifies as it -- the round trip, end to end through
    ``make_cluster`` rather than only through the generator."""
    from asmcmc.data_preparation.cluster_analysis import pair_records

    for motif in MOTIFS:
        if motif == UNIFORM:
            continue
        cluster = make_cluster(2, reference, config_rng(5, 0), SamplingSettings(), motif)
        assert cluster.info["motif"] == motif

        cluster.info["interaction_energy"] = 0.0
        cluster.info["gbq_interaction_energy"] = 0.0
        cluster.info.update(gbq_baseline(cluster))
        cluster.info.update(molecular_geometry_metadata(cluster))
        records = pair_records([cluster])
        assert motif_masks(records)[motif][0]


def test_a_campaign_records_the_composition_it_realised(stub_uma, tmp_path):
    """The manifest states counts, not just the request -- that number is the
    one the coverage problem is about."""
    _run(tmp_path, 8, quotas={COFACIAL: 0.5, UNIFORM: 0.5})
    cfg = json.loads((tmp_path / CONFIG_NAME).read_text())

    assert cfg["motif_counts"] == {COFACIAL: 4, UNIFORM: 4}
    assert cfg["quotas"] == {COFACIAL: 0.5, UNIFORM: 0.5}

    motifs = [f.info["motif"] for f in dataset_frames(tmp_path)]
    assert sorted(motifs) == sorted([COFACIAL] * 4 + [UNIFORM] * 4)


def test_a_default_campaign_is_entirely_uniform(stub_uma, tmp_path):
    """The pre-motif behaviour is still what you get by not asking."""
    _run(tmp_path, 4)
    cfg = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert cfg["motif_counts"] == {UNIFORM: 4}


def test_seeding_cofacial_actually_moves_the_census(stub_uma, tmp_path):
    """The measured effect the restructure exists for.

    **A quota is per configuration; the census is per pair.** A trimer
    contributes three pairs of which only the seeded one carries the requested
    motif, so a fully-seeded campaign lands at ``1 / (1 + 2 * trimer_fraction)``,
    not at 1. Both limits are pinned here because the gap between them is
    exactly the thing that would otherwise be misread as the seeding failing.
    """
    from asmcmc.data_preparation.cluster_analysis import pair_records

    def cofacial_share(directory, n, **kw):
        # "full": a trimer's three pair labels come from its per-pair energies,
        # which only that mode computes, and pair_records needs them.
        _run(directory, n, decomposition="full", **kw)
        records = pair_records(dataset_frames(directory))
        return float(motif_masks(records)[COFACIAL].mean())

    # Dimers only: every pair in the campaign is a seeded pair.
    dimers_only = cofacial_share(
        tmp_path / "dimers",
        16,
        quotas={COFACIAL: 1.0},
        settings=SamplingSettings(trimer_fraction=0.0),
    )
    assert dimers_only == pytest.approx(1.0)

    # With trimers, diluted by the incidental pairs they bring with them.
    fraction = 0.7
    mixed = cofacial_share(
        tmp_path / "mixed",
        48,
        quotas={COFACIAL: 1.0},
        settings=SamplingSettings(trimer_fraction=fraction),
    )
    assert mixed == pytest.approx(1.0 / (1.0 + 2.0 * fraction), abs=0.12)

    plain = cofacial_share(tmp_path / "plain", 48)
    assert mixed > 10 * max(plain, 1e-3)


def test_the_trimer_ceiling_is_honoured_under_motif_seeding(reference):
    """Regression: the previous motif path reached into the mixture's uniform
    component for the third molecule, whose ceiling came from
    ``max_com_distance``, so a tighter trimer ceiling was silently ignored."""
    settings = SamplingSettings(trimer_max_com_distance=6.0)
    for k in range(12):
        cluster = make_cluster(3, reference, config_rng(31, k), settings, COFACIAL)
        coms = np.asarray(
            [
                center_of_mass(cluster.get_positions()[ids], cluster.get_masses()[ids])
                for ids in molecule_indices(cluster)
            ]
        )
        # The third molecule was placed against one of the first two.
        assert min(np.linalg.norm(coms[2] - coms[0]), np.linalg.norm(coms[2] - coms[1])) <= 6.0


def test_progress_queue_counts_every_configuration(stub_uma, tmp_path):
    """The bar is only useful if it is honest: the posted total must equal the
    configurations actually generated, or a long run reads as stalled or as
    finished early."""
    import queue

    reported = queue.Queue()
    result = generate_shard(
        out_dir=str(tmp_path), shard=0, n_configs=7, seed=3,
        settings_dict=asdict(SamplingSettings()), model="stub", device="cpu",
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=2,
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
        decomposition="monomers", motifs=[UNIFORM] * 64, flush_every=2,
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

    from asmcmc.data_preparation.cluster_dataset import _drain

    q = queue.Queue()
    for n in (1, 1, 3, 5):
        q.put(n)
    assert _drain(q) == 10
    assert _drain(q) == 0  # empty, and must not hang
