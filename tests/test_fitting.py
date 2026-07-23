import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ase
import pytest

from asmcmc.potentials import gb, quadrupole, GB_PARAMS, QQ
from asmcmc.fitting_gbq.data import gbq, extract_periodic_pairs, FitData
from asmcmc.fitting_gbq.fit import (
    predict_per_mol,
    boltzmann_weights,
    objective_function,
    train_test_split,
    default_bounds,
    run_fit,
    DEFAULT_ALPHA,
    DEFAULT_BOUNDS,
    PARAM_NAMES,
    PENALTY,
)
from asmcmc.fitting_gbq.report import (
    regression_metrics,
    evaluate_fit,
    params_to_dict,
    sanity_checks,
    write_artifacts,
    PARAM_UNITS,
    BENZENE_SUBLIMATION_EV_PER_MOL,
)
from asmcmc.fitting_gbq import run as run_mod
from asmcmc.fitting_gbq.run import (
    build_parser,
    cli,
    DEFAULT_DATA,
    DEFAULT_CUTOFF,
    DEFAULT_OUT,
)

# theta order: [sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q, E_intra]
THETA = [*GB_PARAMS.values(), QQ, 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _random_unit_vectors(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _synthetic_fitdata(n_frames=4, pairs_per_frame=6, seed=1):
    """A small FitData with plausible-range geometry and absolute targets.

    The pair invariants are drawn in their valid ranges (|a|, |b| <= 1); their
    exact values are irrelevant to the algebra these tests exercise (shape, the
    E_intra offset, the residual/merit identity), only that gbq stays finite.
    """
    rng = np.random.default_rng(seed)
    P = n_frames * pairs_per_frame
    return FitData(
        r_mag=rng.uniform(6.0, 14.0, P),
        a_i=rng.uniform(-1.0, 1.0, P),
        a_j=rng.uniform(-1.0, 1.0, P),
        b_ij=rng.uniform(-1.0, 1.0, P),
        frame_index=np.repeat(np.arange(n_frames), pairs_per_frame),
        n_mol=rng.integers(1, 3, n_frames).astype(float),
        target_per_mol=rng.uniform(-1602.0, -1600.0, n_frames),
        cutoff=15.0,
    )


# ---------------------------------------------------------------------------
# Math fidelity: the duplicated gbq must equal the canonical potentials.py
# ---------------------------------------------------------------------------


def test_gbq_matches_potentials_gb_plus_quadrupole():
    """gbq(invariants) == potentials.gb + quadrupole(vectors) on random pairs.

    Guards the vectorised re-implementation in data.py from drifting away from
    the exact functions the MC uses.
    """
    rng = np.random.default_rng(0)
    n = 300
    u1 = _random_unit_vectors(n, rng)
    u2 = _random_unit_vectors(n, rng)
    r_mag = rng.uniform(5.5, 13.0, n)
    r_vec = _random_unit_vectors(n, rng) * r_mag[:, None]

    r_hat = r_vec / r_mag[:, None]
    a_i = np.einsum("pk,pk->p", r_hat, u1)
    a_j = np.einsum("pk,pk->p", r_hat, u2)
    b_ij = np.einsum("pk,pk->p", u1, u2)

    got = gbq(r_mag, a_i, a_j, b_ij, *GB_PARAMS.values(), QQ)
    ref = gb(u1, u2, r_vec, **GB_PARAMS) + np.squeeze(quadrupole(u1, u2, r_vec, QQ))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Lattice-sum foundation: extraction + the 1/2 vs an independent image sum
# ---------------------------------------------------------------------------


def test_self_image_lattice_sum_matches_bruteforce():
    """predict_per_mol on a 1-particle crystal == a brute-force self-image sum.

    A single particle in a small cubic cell interacts only with its own
    periodic images. extract_periodic_pairs + predict_per_mol's 0.5 factor must
    reproduce an independent triple-loop over image shifts evaluated with the
    canonical potentials.py functions.
    """
    L = 7.0
    cutoff = 16.0
    u = _unit([1.0, 2.0, 3.0])
    frame = ase.Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=np.diag([L, L, L]), pbc=True
    )
    frame.new_array("or_vec", u[None, :].copy())

    pairs = extract_periodic_pairs(frame, "or_vec", cutoff)
    data = FitData(
        r_mag=pairs[:, 0],
        a_i=pairs[:, 1],
        a_j=pairs[:, 2],
        b_ij=pairs[:, 3],
        frame_index=np.zeros(len(pairs), dtype=int),
        n_mol=np.array([1.0]),
        target_per_mol=np.array([0.0]),
        cutoff=cutoff,
    )
    pred = predict_per_mol(THETA, data)[0]

    # Independent self-image lattice sum (potentials.py math). Each image is one
    # directed pair; the per-molecule energy is half their sum.
    total = 0.0
    n_images = 0
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                if n1 == n2 == n3 == 0:
                    continue
                r_vec = np.array([[n1 * L, n2 * L, n3 * L]])
                if np.linalg.norm(r_vec) >= cutoff:
                    continue
                total += gb(u[None], u[None], r_vec, **GB_PARAMS).item()
                total += np.squeeze(quadrupole(u[None], u[None], r_vec, QQ)).item()
                n_images += 1

    np.testing.assert_allclose(pred, 0.5 * total, rtol=1e-9, atol=1e-12)
    # All shells with sum(n^2) <= 5 fit inside 16 Å at L = 7 -> 56 images.
    assert len(pairs) == n_images == 56


# ---------------------------------------------------------------------------
# Boltzmann weights: finite on absolute energies, physically ordered, invariant
# ---------------------------------------------------------------------------


def test_boltzmann_weights_stable_and_invariant():
    """Weights stay finite on ~-1600 eV targets and are shift-invariant."""
    target = np.array([-1601.0, -1600.5, -1602.3, -1599.8, -1601.7])
    w = boltzmann_weights(target)

    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    # most-bound (lowest-energy) frame receives the largest weight
    assert np.argmax(w) == np.argmin(target)
    # exact stabilised form: w = exp(-alpha (E - min E))
    # (rtol loosened from machine eps for the differing float associativity:
    # the code subtracts (-alpha E).max(), the reference factors out E.min())
    np.testing.assert_allclose(
        w, np.exp(-DEFAULT_ALPHA * (target - target.min())), rtol=1e-9
    )
    # a naive exp(-alpha E) overflows at these energies; the stabilised one must not
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(-DEFAULT_ALPHA * target)).all()
    # normalised weights are invariant to a global shift of every energy
    w_shift = boltzmann_weights(target + 137.0)
    np.testing.assert_allclose(w / w.sum(), w_shift / w_shift.sum(), rtol=1e-9)


# ---------------------------------------------------------------------------
# predict_per_mol: shape and the additive E_intra offset
# ---------------------------------------------------------------------------


def test_predict_per_mol_shape_and_offset():
    """One prediction per frame; E_intra shifts every prediction equally."""
    data = _synthetic_fitdata()
    pred = predict_per_mol(THETA, data)

    assert pred.shape == (data.n_frames,)
    assert np.all(np.isfinite(pred))

    shifted = list(THETA)
    shifted[-1] = THETA[-1] + 5.0
    np.testing.assert_allclose(predict_per_mol(shifted, data) - pred, 5.0, atol=1e-9)


# ---------------------------------------------------------------------------
# objective_function: the scalar Cacelli merit F/2 that DE minimises
# ---------------------------------------------------------------------------


def test_objective_equals_half_cacelli_merit():
    """objective_function == 0.5 * sum_k (w_k / sum w)(pred_k - E_k)**2 = F/2."""
    data = _synthetic_fitdata()
    weights = boltzmann_weights(data.target_per_mol)
    obj = objective_function(THETA, data, weights)

    pred = predict_per_mol(THETA, data)
    w = weights / weights.sum()
    F = np.sum(w * (pred - data.target_per_mol) ** 2)
    np.testing.assert_allclose(obj, 0.5 * F, rtol=1e-12)


def test_objective_idx_subsets_and_renormalises():
    """idx restricts to a frame subset and renormalises weights within it."""
    data = _synthetic_fitdata()
    weights = boltzmann_weights(data.target_per_mol)
    idx = np.array([0, 2])
    obj = objective_function(THETA, data, weights, idx=idx)

    pred = predict_per_mol(THETA, data)[idx]
    target = data.target_per_mol[idx]
    w = weights[idx] / weights[idx].sum()
    np.testing.assert_allclose(obj, 0.5 * np.sum(w * (pred - target) ** 2), rtol=1e-12)


def test_objective_penalises_nonfinite():
    """A parameter region that yields NaN/inf returns the finite PENALTY.

    mu -> 0 drives the GB energy's 1/mu term singular; the objective must report
    a large finite value so differential_evolution avoids it rather than crash.
    """
    data = _synthetic_fitdata()
    weights = boltzmann_weights(data.target_per_mol)
    bad = list(THETA)
    bad[PARAM_NAMES.index("mu")] = 0.0
    obj = objective_function(bad, data, weights)
    assert np.isfinite(obj)
    assert obj == PENALTY


# ---------------------------------------------------------------------------
# fitting machinery: split, bounds, and the differential_evolution driver
# ---------------------------------------------------------------------------


def test_train_test_split_deterministic_and_partitions():
    """Same seed -> same split; train/test are disjoint and cover every frame."""
    n = 50
    train, test = train_test_split(n, test_frac=0.2, seed=7)
    train2, test2 = train_test_split(n, test_frac=0.2, seed=7)

    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)
    assert len(test) == 10
    assert set(train).isdisjoint(test)
    np.testing.assert_array_equal(np.union1d(train, test), np.arange(n))
    # a different seed gives a different partition
    assert not np.array_equal(test, train_test_split(n, test_frac=0.2, seed=8)[1])


def test_default_bounds_box():
    """7 physical params from DEFAULT_BOUNDS; E_intra window straddles the mean."""
    data = _synthetic_fitdata()
    bounds = default_bounds(data, e_intra_half_window=5.0)

    assert len(bounds) == len(PARAM_NAMES)
    for name, b in zip(PARAM_NAMES[:-1], bounds[:-1]):
        assert b == DEFAULT_BOUNDS[name]
    lo, hi = bounds[-1]
    mean = float(np.mean(data.target_per_mol))
    np.testing.assert_allclose([lo, hi], [mean - 5.0, mean + 5.0])
    # idx restricts the E_intra centre to the given frame subset
    idx = np.array([0, 1])
    lo_i, hi_i = default_bounds(data, idx=idx)[-1]
    np.testing.assert_allclose(
        0.5 * (lo_i + hi_i), float(np.mean(data.target_per_mol[idx]))
    )


def test_run_fit_recovers_synthetic_params():
    """DE recovers known params from noise-free synthetic targets.

    Build targets as predict_per_mol(true_theta) on real-geometry pairs, then
    fit with bounds bracketing the truth. With a clean signal DE should drive
    the merit to ~0 and match predictions; E_intra (the dominant ~-1601 eV/mol
    term) is well identified so we check it directly.
    """
    data = _synthetic_fitdata(n_frames=8, pairs_per_frame=10, seed=3)
    # order: [sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q, E_intra]
    true = [6.0, 0.05, 0.4, 0.7, 2.0, 1.0, 1.0, -3.0, -1601.0]
    data.target_per_mol[:] = predict_per_mol(true, data)

    # bracket each true value; E_intra inferred from the (now exact) targets
    bounds = [
        (5.0, 7.0),
        (1e-3, 0.2),
        (0.2, 0.7),
        (0.3, 1.2),
        (1.0, 3.0),
        (0.5, 1.5),
        (0.5, 1.5),
        (-5.0, 0.0),
    ]
    bounds.append(default_bounds(data)[-1])

    weights = np.ones(data.n_frames)  # uniform: noise-free recovery, no reweight
    res = run_fit(
        data,
        weights=weights,
        bounds=bounds,
        seed=0,
        maxiter=300,
        tol=1e-10,
        polish=True,
    )

    assert res.fun < 1e-6
    np.testing.assert_allclose(
        predict_per_mol(res.x, data), data.target_per_mol, atol=1e-3
    )
    np.testing.assert_allclose(res.x[PARAM_NAMES.index("E_intra")], -1601.0, atol=1e-2)


# ---------------------------------------------------------------------------
# report.regression_metrics: scalar error summary
# ---------------------------------------------------------------------------


def test_regression_metrics_perfect_fit():
    """pred == target -> zero errors and R^2 == 1."""
    target = np.array([-1.0, 0.5, 2.0, 3.5])
    m = regression_metrics(target.copy(), target)
    assert m["n"] == 4
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["max_abs_err"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_hand_computed():
    """Known (pred, target) -> hand-computed rmse/mae/max/R^2."""
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.0])
    # err = [-0.5, 0, 1]; ss_res = 1.25; ss_tot = 0.1666...; r2 = 1 - 7.5
    m = regression_metrics(pred, target)
    assert m["n"] == 3
    assert m["rmse"] == pytest.approx(np.sqrt(1.25 / 3))
    assert m["mae"] == pytest.approx(0.5)
    assert m["max_abs_err"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(-6.5)


def test_regression_metrics_empty_and_constant_target():
    """Empty subset -> all NaN; constant target -> R^2 NaN (variance 0)."""
    empty = regression_metrics(np.array([]), np.array([]))
    assert empty["n"] == 0
    assert all(np.isnan(empty[k]) for k in ("rmse", "mae", "max_abs_err", "r2"))

    const = regression_metrics(np.array([0.0, 1.0]), np.array([2.0, 2.0]))
    assert np.isnan(const["r2"])
    assert const["rmse"] == pytest.approx(np.sqrt((4.0 + 1.0) / 2))


# ---------------------------------------------------------------------------
# report.evaluate_fit: per-partition metrics, with/without a split and pred reuse
# ---------------------------------------------------------------------------


def test_evaluate_fit_all_partition_default():
    """No split -> a single 'all' partition covering every frame."""
    data = _synthetic_fitdata(n_frames=10)
    out = evaluate_fit(THETA, data)
    assert set(out) == {"all"}
    assert out["all"]["n"] == data.n_frames


def test_evaluate_fit_split_partitions():
    """train/test keys; counts partition the frames without overlap."""
    data = _synthetic_fitdata(n_frames=10)
    train, test = train_test_split(data.n_frames, test_frac=0.2, seed=0)
    out = evaluate_fit(THETA, data, train_idx=train, test_idx=test)
    assert set(out) == {"train", "test"}
    assert out["train"]["n"] == len(train)
    assert out["test"]["n"] == len(test)
    assert out["train"]["n"] + out["test"]["n"] == data.n_frames


def test_evaluate_fit_pred_reuse_matches_theta_path():
    """Passing precomputed pred (theta=None) equals recomputing from theta."""
    data = _synthetic_fitdata(n_frames=10)
    pred = predict_per_mol(THETA, data)
    from_theta = evaluate_fit(THETA, data)
    from_pred = evaluate_fit(None, data, pred=pred)
    assert from_theta == from_pred


# ---------------------------------------------------------------------------
# report.params_to_dict: self-describing, JSON-serialisable parameter mapping
# ---------------------------------------------------------------------------


def test_params_to_dict_values_units_and_json_roundtrip():
    """Every param maps to its value+unit, in order, and survives JSON."""
    import json

    d = params_to_dict(THETA)
    assert list(d) == PARAM_NAMES  # order preserved
    for name, value in zip(PARAM_NAMES, THETA):
        assert d[name]["value"] == pytest.approx(value)
        assert d[name]["unit"] == PARAM_UNITS[name]

    # round-trips through JSON unchanged (plain floats, no numpy scalars)
    assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# report.sanity_checks: physical reasonableness of a fitted theta
# ---------------------------------------------------------------------------


def test_sanity_checks_offset_and_min_lattice():
    """E_intra is read from its slot; min lattice = min(pred) - E_intra."""
    data = _synthetic_fitdata(n_frames=8)
    s = sanity_checks(THETA, data)

    e_intra = THETA[PARAM_NAMES.index("E_intra")]
    pred = predict_per_mol(THETA, data)
    assert s["E_intra_eV_per_mol"] == pytest.approx(e_intra)
    assert s["mean_target_eV_per_mol"] == pytest.approx(data.target_per_mol.mean())
    # lattice energy strips the offset: min(pred - E_intra) == min(pred) - E_intra
    assert s["min_lattice_energy_eV_per_mol"] == pytest.approx(pred.min() - e_intra)
    assert s["benzene_sublimation_ref_eV_per_mol"] == BENZENE_SUBLIMATION_EV_PER_MOL


def test_sanity_checks_lattice_invariant_to_e_intra():
    """E_intra tracks the offset, but the lattice energy strips it out.

    pred = 0.5*frame_e/n_mol + E_intra, so lattice = pred - E_intra cancels the
    offset: shifting E_intra moves the reported offset but leaves the cohesive
    (lattice) energy unchanged.
    """
    data = _synthetic_fitdata(n_frames=8)
    shifted = list(THETA)
    shifted[PARAM_NAMES.index("E_intra")] += 3.0
    base = sanity_checks(THETA, data)
    bumped = sanity_checks(shifted, data)
    assert bumped["E_intra_eV_per_mol"] == pytest.approx(
        base["E_intra_eV_per_mol"] + 3.0
    )
    assert bumped["min_lattice_energy_eV_per_mol"] == pytest.approx(
        base["min_lattice_energy_eV_per_mol"]
    )


def test_sanity_checks_pred_reuse_matches():
    """Passing precomputed pred (theta still supplies E_intra) matches."""
    data = _synthetic_fitdata(n_frames=8)
    pred = predict_per_mol(THETA, data)
    assert sanity_checks(THETA, data, pred=pred) == sanity_checks(THETA, data)


# ---------------------------------------------------------------------------
# report.write_artifacts: the three on-disk files + returned dicts
# ---------------------------------------------------------------------------


def test_write_artifacts_files_and_contents(tmp_path):
    """Writes params/metrics JSON + markdown report; returns matching dicts."""
    import json

    data = _synthetic_fitdata(n_frames=10)
    train, test = train_test_split(data.n_frames, test_frac=0.2, seed=0)
    out = write_artifacts(
        str(tmp_path),
        THETA,
        data,
        train_idx=train,
        test_idx=test,
        meta={"seed": 0, "dataset": "synthetic"},
    )

    params_file = tmp_path / "params.json"
    metrics_file = tmp_path / "metrics.json"
    report_file = tmp_path / "fit_report.md"
    assert params_file.exists() and metrics_file.exists() and report_file.exists()

    # JSON on disk matches the returned in-memory dicts
    assert json.loads(params_file.read_text()) == out["params"]
    on_disk = json.loads(metrics_file.read_text())
    assert on_disk["metrics"] == out["metrics"]
    assert on_disk["sanity"] == out["sanity"]
    assert set(out["metrics"]) == {"train", "test"}

    # report rendered (format_report ran without error) and carries the run meta
    text = report_file.read_text()
    assert text.startswith("# GBQ fit report")
    assert "dataset" in text and "sigma0" in text


# ---------------------------------------------------------------------------
# run.build_parser / run.cli: flag parsing and the namespace -> main() mapping
# ---------------------------------------------------------------------------


def test_build_parser_defaults_and_de_knob_grouping():
    """Unset args take main()'s defaults; unset DE knobs stay None (dropped later)."""
    args = build_parser().parse_args([])
    assert args.dataset == DEFAULT_DATA
    assert args.cutoff == DEFAULT_CUTOFF
    assert args.out_dir == DEFAULT_OUT
    assert args.weightings is None  # append default -> cli() supplies both variants
    assert args.index == ":"
    assert args.progress is True
    assert args.alpha == DEFAULT_ALPHA  # unset -> the Cacelli default
    assert args.workers == 1
    # DE knobs unset -> None so cli() omits them and SciPy's own defaults apply
    assert args.maxiter is None and args.popsize is None and args.tol is None


def _capture_main(monkeypatch):
    """Replace run.main with a recorder; returns the dict it captures kwargs into."""
    captured = {}

    def fake_main(**kwargs):
        captured.update(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(run_mod, "main", fake_main)
    return captured


def test_cli_translates_namespace_to_main(monkeypatch):
    """Set flags flow to main(); only the provided DE knobs ride in de_kwargs."""
    captured = _capture_main(monkeypatch)
    out = cli(
        [
            "--dataset",
            "foo.xyz",
            "--cutoff",
            "12",
            "--out-dir",
            "/tmp/x",
            "--weighting",
            "boltzmann",
            "--index",
            ":50",
            "--test-frac",
            "0.3",
            "--split-seed",
            "1",
            "--fit-seed",
            "2",
            "--workers",
            "-1",
            "--maxiter",
            "100",
            "--tol",
            "1e-3",
            "--alpha",
            "3.87",
        ]
    )
    assert out == {"sentinel": True}
    assert captured["dataset_path"] == "foo.xyz"
    assert captured["cutoff"] == 12.0
    assert captured["out_dir"] == "/tmp/x"
    assert captured["weightings"] == ("boltzmann",)
    assert captured["index"] == ":50"
    assert captured["test_frac"] == 0.3
    assert captured["split_seed"] == 1 and captured["fit_seed"] == 2
    assert captured["alpha"] == 3.87
    assert captured["workers"] == -1
    # only the knobs we passed appear; popsize (unset) must not
    assert captured["maxiter"] == 100 and captured["tol"] == 1e-3
    assert "popsize" not in captured


def test_cli_weighting_repeatable_and_no_progress(monkeypatch):
    """--weighting is repeatable; absent -> both; --no-progress flips progress."""
    captured = _capture_main(monkeypatch)
    cli(["--weighting", "boltzmann", "--weighting", "uniform", "--no-progress"])
    assert captured["weightings"] == ("boltzmann", "uniform")
    assert captured["progress"] is False

    captured.clear()
    cli([])  # no --weighting -> fall back to both variants, progress on by default
    assert captured["weightings"] == ("boltzmann", "uniform")
    assert captured["progress"] is True
