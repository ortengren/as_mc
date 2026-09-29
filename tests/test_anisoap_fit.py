"""AniSOAP Delta-learning: descriptor contracts, and the sweep's bookkeeping.

No campaign on disk and no UMA anywhere: every test either builds a handful of
ellipsoid frames directly or drives the sweep over a tiny synthetic campaign
written into ``tmp_path``, so the suite runs from a fresh clone. AniSOAP itself
does run, since it's what is being tested, but only on frames with two or three
beads.
"""

import csv
import importlib.util
import json

import numpy as np
import pytest
from ase.io import write

# Checked with find_spec rather than importorskip: importing AniSOAP or
# scikit-learn here, before asmcmc sets single-threaded BLAS, would make
# in-process fits run multi-threaded and differ in the last digits from the
# sweep's worker processes.
if not all(importlib.util.find_spec(name) for name in ("anisoap", "sklearn")):
    pytest.skip("AniSOAP or scikit-learn is not installed", allow_module_level=True)

from asmcmc.delta_learning.descriptors import (
    Hypers,
    descriptors,
    ellipsoid_frames,
    load_training_set,
    make_ellipsoid_frame,
    quaternions_from_normals,
)
from asmcmc.delta_learning.model import (
    AniSOAPDeltaPotential,
    fit_delta,
    train_test_split,
)
from asmcmc.delta_learning.sweep import (
    COMPARISON_NAME,
    MODEL_NAME,
    POINTS_DIRNAME,
    build_grid,
    fit_and_score_point,
    load_model,
    run_sweep,
)
from asmcmc.mc.potentials import CACELLI_POTENTIAL

Z_HAT = np.array([0.0, 0.0, 1.0])

# Enough frames for the RidgeCV folds to work. Below about 20, every alpha scores
# nan and the search picks one arbitrarily, so the resume checks would prove
# nothing.
N_CAMPAIGN = 24


def dimer_frame(separation, hypers, normals=None):
    """Two discs on the z-axis, ``separation`` apart."""
    normals = np.array([Z_HAT, Z_HAT]) if normals is None else np.asarray(normals, float)
    centres = np.stack([np.zeros(3), np.array([0.0, 0.0, float(separation)])])
    return make_ellipsoid_frame(centres, quaternions_from_normals(normals), hypers)


# --------------------------------------------------------------------------
# The descriptor contracts
# --------------------------------------------------------------------------


def test_spin_about_a_uniaxial_normal_moves_no_descriptor():
    """This invariance is what allows MC frames to be given an arbitrary azimuth.

    An MC frame only has ``or_vec``, a disc normal with 2 degrees of freedom, so
    ``quaternions_from_normals`` has to make up the third. That's only valid
    because both in-plane semiaxes are equal, so spinning a particle about its
    normal changes nothing. If this test fails, ``Hypers`` has become biaxial and
    the descriptors of MC frames would be silently wrong.
    """
    hypers = Hypers()
    rng = np.random.default_rng(0)
    normals = np.array([Z_HAT, Z_HAT])
    centres = np.stack([np.zeros(3), np.array([0.0, 0.0, 4.5])])

    reference = None
    for _ in range(4):
        frame = make_ellipsoid_frame(
            centres, quaternions_from_normals(normals, rng=rng), hypers
        )
        row = descriptors([frame], hypers)[0]
        if reference is None:
            reference = row
        else:
            assert np.abs(row - reference).max() < 1e-15


def test_a_batch_entirely_outside_the_cutoff_is_all_zero_not_an_error():
    """AniSOAP raises an error here instead of returning nothing, and our check handles it.

    This happens in normal use: ``dimer_scan`` moves a dimer past the cutoff
    every time the benchmark runs, and it's common at small ``cutoff_radius``.
    """
    hypers = Hypers()
    frames = [dimer_frame(r, hypers) for r in (20.0, 25.0)]
    X = descriptors(frames, hypers)
    assert X.shape == (2, hypers.n_features)
    assert not X.any()


def test_only_the_far_frame_is_zeroed_in_a_mixed_batch():
    hypers = Hypers()
    frames = [dimer_frame(r, hypers) for r in (4.0, 5.0, 20.0)]
    X = descriptors(frames, hypers)
    assert (np.abs(X).max(axis=1)[:2] > 0).all()
    assert not X[2].any()


@pytest.mark.parametrize("max_angular,max_radial", [(3, 3), (3, 6), (9, 3), (9, 6)])
def test_derived_feature_count_matches_the_realised_width(max_angular, max_radial):
    """``Hypers.n_features`` is our own formula, which AniSOAP doesn't provide, so check it.

    The case above needs a width when there's nothing to measure it from, so a
    wrong formula would silently give all-zero rows of the wrong shape.
    """
    hypers = Hypers(max_angular=max_angular, max_radial=max_radial)
    assert descriptors([dimer_frame(4.5, hypers)], hypers).shape[1] == hypers.n_features


def test_hypers_round_trip_through_json_and_key_is_stable():
    hypers = Hypers(max_angular=5, max_radial=4, cutoff_radius=7.5)
    assert Hypers.from_dict(json.loads(json.dumps(hypers.to_dict()))) == hypers
    # from_dict ignores foreign keys, so a metrics blob can be passed whole.
    assert Hypers.from_dict({**hypers.to_dict(), "test_rmse": 1.0}) == hypers


# --------------------------------------------------------------------------
# The no-intercept contract
# --------------------------------------------------------------------------


def _toy_fit(hypers, n=24, seed=0):
    """A cheap fitted model over dimers spanning the cutoff."""
    rng = np.random.default_rng(seed)
    separations = np.linspace(3.6, hypers.cutoff_radius + 3.0, n)
    frames = [dimer_frame(r, hypers) for r in separations]
    X = descriptors(frames, hypers)
    delta = 0.01 * np.exp(-separations / 3.0) + 1e-4 * rng.standard_normal(n)
    train, test = train_test_split(n, 0.25, 0)
    return fit_delta(X, delta, train, test, hypers=hypers)


def test_a_zero_descriptor_predicts_exactly_zero_delta():
    """Exactly zero, not just small.

    In MC the per-centre energies are summed over N = 400 particles, so a
    constant per centre would become a large, spurious shift in the total
    energy. This is why the model has no intercept and no feature centring.
    """
    hypers = Hypers()
    model = _toy_fit(hypers).model
    assert model.predict(np.zeros((1, hypers.n_features)))[0] == 0.0


def test_the_correction_vanishes_at_large_separation():
    hypers = Hypers()
    model = _toy_fit(hypers).model
    potential = AniSOAPDeltaPotential(model)

    uhat = np.array([Z_HAT])
    r = np.array([[0.0, 0.0, 40.0]])
    assert potential.delta(uhat, uhat, r)[0] == 0.0
    # ... so the pair energy is exactly the baseline out there.
    assert potential.pair_energy(uhat, uhat, r) == pytest.approx(
        CACELLI_POTENTIAL.pair_energy(uhat, uhat, r)
    )


# --------------------------------------------------------------------------
# The sweep's bookkeeping
# --------------------------------------------------------------------------


def test_grid_is_the_full_product_ordered_most_expensive_first():
    grid = build_grid(angular=(3, 9), radial=(3, 6), cutoffs=(6.0, 9.0))
    assert len(grid) == 8
    assert len({h.key for h in grid}) == 8
    # The long pole starts at t=0 so the pool does not wait on a straggler.
    assert grid[0].cutoff_radius == 9.0 and grid[0].max_angular == 9
    assert grid[-1].cutoff_radius == 6.0 and grid[-1].max_angular == 3


@pytest.fixture
def campaign(tmp_path):
    """A tiny synthetic campaign in the shape ``load_training_set`` reads."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N_CAMPAIGN):
        separation = 3.6 + 0.5 * (i % 8)
        normals = np.array([Z_HAT, Z_HAT])
        centres = np.stack([np.zeros(3), np.array([0.0, 0.0, separation])])
        frame = make_ellipsoid_frame(centres, quaternions_from_normals(normals), Hypers())
        axes = np.tile(np.eye(3), (2, 1, 1))
        gbq = float(0.02 * rng.standard_normal())
        frame.info.update(
            molecular_com=centres,
            principal_axes=axes,
            interaction_energy=gbq + 0.01 * np.exp(-separation / 3.0),
            gbq_interaction_energy=gbq,
            n_molecules=2,
        )
        frames.append(frame)

    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    write(campaign_dir / "clusters_shard00.xyz", frames, format="extxyz")
    return campaign_dir


def test_geometry_reads_the_synthetic_campaign(campaign):
    training_set = load_training_set(campaign, cache_dir=None)
    assert len(training_set) == N_CAMPAIGN
    assert training_set.com.shape == (2 * N_CAMPAIGN, 3)
    assert len(ellipsoid_frames(training_set, Hypers())) == N_CAMPAIGN


def test_a_point_writes_its_three_artifacts_and_the_model_round_trips(campaign, tmp_path):
    hypers = Hypers(max_angular=3, max_radial=3, cutoff_radius=6.0)
    train, test = train_test_split(N_CAMPAIGN, 0.25, 0)
    cfg = {
        "campaign": str(campaign),
        "out_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "train_idx": train.tolist(),
        "test_idx": test.tolist(),
        "cv_seed": 0,
        "gate": False,
        "dimers": None,
        "meta": {},
    }
    record = fit_and_score_point(hypers.to_dict(), cfg)
    assert record["skipped"] is False
    assert record["extras"]["n_features"] == hypers.n_features

    point_dir = tmp_path / "out" / POINTS_DIRNAME / hypers.key
    for name in ("hypers.json", "metrics.json", "model.npz"):
        assert (point_dir / name).exists()

    model = load_model(point_dir)
    assert model.hypers == hypers
    assert model.alpha == record["alpha"]
    assert model.coef.shape == (hypers.n_features,)


def test_the_descriptor_build_and_the_fit_are_timed_separately(campaign, tmp_path):
    """Only the two stages that depend on the hyperparameters are timed.

    The check is ``>= 0`` rather than ``> 0`` because on a 24-frame synthetic
    campaign the fit really does round to 0.000 s. What's being tested is that
    the two stages are timed separately, not how long they take.
    """
    hypers = Hypers(max_angular=3, max_radial=3, cutoff_radius=6.0)
    train, test = train_test_split(N_CAMPAIGN, 0.25, 0)
    cfg = {
        "campaign": str(campaign),
        "out_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "train_idx": train.tolist(),
        "test_idx": test.tolist(),
        "cv_seed": 0,
        "gate": False,
        "dimers": None,
        "meta": {},
    }
    timing = fit_and_score_point(hypers.to_dict(), cfg)["timing"]

    # Exactly these two. Loading the geometry and running the benchmark aren't
    # timed, so an extra key means one of them has started being timed.
    assert set(timing) == set(TIMING_COLUMNS)
    assert all(np.isfinite(v) and v >= 0 for v in timing.values())
    assert timing["descriptors_s"] > 0


def test_rerunning_a_point_skips_instead_of_refitting(campaign, tmp_path):
    hypers = Hypers(max_angular=3, max_radial=3, cutoff_radius=6.0)
    train, test = train_test_split(N_CAMPAIGN, 0.25, 0)
    cfg = {
        "campaign": str(campaign),
        "out_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "train_idx": train.tolist(),
        "test_idx": test.tolist(),
        "cv_seed": 0,
        "gate": False,
        "dimers": None,
        "meta": {},
    }
    first = fit_and_score_point(hypers.to_dict(), cfg)
    second = fit_and_score_point(hypers.to_dict(), cfg)

    assert first["skipped"] is False and second["skipped"] is True
    assert second["test"] == first["test"]
    assert second["alpha"] == first["alpha"]


# Every measured column, so the resume comparison drops exactly the fields that
# are timings and no others. Kept next to the helper that pops them so the two
# cannot drift apart.
TIMING_COLUMNS = ("descriptors_s", "fit_s")


def _comparison_rows(path):
    """The comparison table minus the timings, which are wall-clock by design."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column in TIMING_COLUMNS:
            row.pop(column)
    return rows


def test_an_interrupted_sweep_resumes_to_the_same_comparison(campaign, tmp_path):
    """Re-running is the resume path, so it must reproduce the finished table."""
    kwargs = dict(
        campaign=str(campaign),
        angular=(3,),
        radial=(3,),
        cutoffs=(6.0, 9.0),
        test_frac=0.25,
        gate=False,
        progress=False,
    )
    full = tmp_path / "full"
    run_sweep(out_dir=str(full), **kwargs)

    # A sweep that only got through one of the two points, then re-run whole.
    partial = tmp_path / "partial"
    run_sweep(out_dir=str(partial), **{**kwargs, "cutoffs": (6.0,)})
    records = run_sweep(out_dir=str(partial), **kwargs)

    assert sum(r["skipped"] for r in records) == 1
    assert _comparison_rows(partial / COMPARISON_NAME) == _comparison_rows(
        full / COMPARISON_NAME
    )


def test_the_sweep_records_the_gate_without_filtering_on_it(campaign, tmp_path):
    """Points that fail the benchmark are recorded like any other; nothing is dropped."""
    records = run_sweep(
        campaign=str(campaign),
        out_dir=str(tmp_path / "gated"),
        angular=(3,),
        radial=(3,),
        cutoffs=(6.0,),
        test_frac=0.25,
        gate=True,
        progress=False,
    )
    assert len(records) == 1
    gate = records[0]["gate"]
    assert "gate_error" not in gate
    assert isinstance(gate["stacking_bound"], bool)
    assert set(gate) >= {"cofacial", "parallel_displaced", "t_shaped"}
    # Running the gate does not add a timing key: only the two
    # hyperparameter-dependent stages are measured, gated or not.
    assert set(records[0]["timing"]) == set(TIMING_COLUMNS)


def test_regate_rescores_a_finished_point_without_refitting(campaign, tmp_path):
    """The reference only affects the benchmark, not the fit.

    This is the test that matters for re-scoring a finished sweep. A finished
    point returns its stored benchmark result unchanged, so changing
    ``reference`` on its own still reports the old reference's verdict.
    ``regate`` is what actually switches it, and it must do so without touching
    the model or the fit metrics.
    """
    kwargs = dict(
        campaign=str(campaign),
        out_dir=str(tmp_path / "regate"),
        angular=(3,),
        radial=(3,),
        cutoffs=(6.0,),
        test_frac=0.25,
        gate=True,
        progress=False,
    )
    model_path = (
        tmp_path
        / "regate"
        / POINTS_DIRNAME
        / Hypers(max_angular=3, max_radial=3, cutoff_radius=6.0).key
        / MODEL_NAME
    )

    first = run_sweep(reference="mp2", **kwargs)[0]
    model_before = model_path.read_bytes()

    # Without regate, the old benchmark result comes back unchanged.
    stale = run_sweep(reference="uma", **kwargs)[0]
    assert stale["gate"]["reference"] == "mp2"

    again = run_sweep(reference="uma", regate=True, **kwargs)[0]

    assert again["skipped"] is True, "regate must not refit"
    assert model_path.read_bytes() == model_before, "regate must not touch the model"
    assert first["gate"]["reference"] == "mp2"
    assert again["gate"]["reference"] == "uma"
    assert again["meta"]["reference"] == "uma"
    # Delta = E_UMA - E_GBQ whatever the probe scores against, so the fit is
    # untouched while the verdict is recomputed.
    assert again["test"] == first["test"]
    assert again["gate"]["well_rmse_kcal"] != first["gate"]["well_rmse_kcal"]
    assert again["gate"]["baseline_well_rmse_kcal"] != pytest.approx(
        first["gate"]["baseline_well_rmse_kcal"]
    )
