import os

import pytest

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from ase.db import connect

from asmcmc.utils.npt_equilibration import (
    point_dirname,
    _submission_order,
    equilibrate_point,
    _evaluate_point,
    plot_point_results,
    find_point_dirs,
    continue_point,
    extend_points,
    main,
)
from asmcmc.base.metropolis import MetropolisCalculator
from asmcmc.base.initialize import RandomLatticeInitializer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_cfg(out_dir):
    return {
        "n_particles": 27,
        "density": 0.3,
        "num_steps": 2 * 27,  # ~2 passes -> 2 recorded blocks
        "block_size": 27,
        "buffer_size": 100,
        "seed0": 100,
        "out_dir": out_dir,
    }


def _last_row(output_dir, db_name="equilibration.db"):
    with connect(os.path.join(output_dir, db_name)) as db:
        row = db.get(db.count())  # rows written in order: last id is most recent
    return {"total_energy": row.total_energy, "vol": row.vol, "step": row.step}


def _db_count(output_dir, db_name="equilibration.db"):
    with connect(os.path.join(output_dir, db_name)) as db:
        return db.count()


# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------


def test_point_dirname_formats_compactly():
    assert point_dirname(300.0, 0.0) == "T300_P0"
    assert point_dirname(300.0, 1e-3) == "T300_P0.001"
    assert point_dirname(250.5, 1e-6) == "T250.5_P1e-06"


def test_submission_order_expensive_first():
    """Cold/high-pressure (densest, hence slowest in NPT) points dispatch first,
    but it is a pure permutation of indices -- each point keeps its canonical k
    (seed/dir), only submission order changes."""
    # canonical grid is pressure-outer / temp-inner:
    # k=0 (200,0)  k=1 (400,0)  k=2 (200,5e-3)  k=3 (400,5e-3)
    grid = [(t, p) for p in (0.0, 5e-3) for t in (200.0, 400.0)]
    # expensive first => high P first, then low T first
    assert _submission_order(grid) == [2, 3, 0, 1]
    # never drops or duplicates a point
    assert sorted(_submission_order(grid)) == list(range(len(grid)))


# ---------------------------------------------------------------------------
# equilibrate_point: artifacts, resumability, guard rails
# ---------------------------------------------------------------------------


def test_equilibrate_point_writes_resumable_artifacts(tmp_path):
    """A point writes equilibration.db + a write-once run_config.json that
    from_equilibration can rebuild a continuable NPT sampler from."""
    output_dir = str(tmp_path / "T300_P0" / "1")
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=1)
    returned = equilibrate_point(
        300.0, 0.0, 4 * 27, init, block_size=27, output_dir=output_dir, seed=1
    )
    assert returned == output_dir
    assert os.path.exists(os.path.join(output_dir, "equilibration.db"))
    assert os.path.exists(os.path.join(output_dir, "run_config.json"))

    metro = MetropolisCalculator.from_equilibration(output_dir)
    assert metro.npt_ensemble is True  # NPT, not NVT
    assert metro.temp == 300.0
    assert metro.pressure == 0.0
    assert metro.step_count > 0  # picked up the last-row step to continue from


def test_equilibrate_point_refuses_existing_db(tmp_path):
    """Re-equilibrating into a populated dir would interleave trajectories, so
    it raises (resume goes through from_equilibration instead)."""
    output_dir = str(tmp_path / "p")
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=1)
    equilibrate_point(300.0, 0.0, 2 * 27, init, 27, output_dir, seed=1)

    init2 = RandomLatticeInitializer(n_particles=27, density=0.3, seed=1)
    with pytest.raises(FileExistsError):
        equilibrate_point(300.0, 0.0, 2 * 27, init2, 27, output_dir, seed=1)


def test_equilibrate_point_seed_determinism(tmp_path):
    """A point is fully determined by its seed -- the basis for safe parallelism.

    Same seed -> identical final frame regardless of where it ran; the global
    RNG reseed inside equilibrate_point is what guarantees this.
    """

    def run(dirname):
        output_dir = str(tmp_path / dirname)
        init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=7)
        equilibrate_point(300.0, 1e-3, 4 * 27, init, 27, output_dir, seed=7)
        return _last_row(output_dir)

    a = run("a")
    b = run("b")
    assert a["total_energy"] == b["total_energy"]
    assert a["vol"] == b["vol"]


# ---------------------------------------------------------------------------
# _evaluate_point: directory layout + idempotent re-run
# ---------------------------------------------------------------------------


def test_evaluate_point_layout_and_idempotent(tmp_path):
    cfg = _small_cfg(str(tmp_path / "scan"))
    k, output_dir = _evaluate_point(0, 300.0, 0.0, cfg)

    assert k == 0
    # out_dir / T{temp}_P{pressure} / {seed0 + k}
    assert output_dir == os.path.join(cfg["out_dir"], "T300_P0", "100")
    assert os.path.exists(os.path.join(output_dir, "equilibration.db"))
    assert os.path.exists(os.path.join(output_dir, "run_config.json"))

    # A finished point is skipped on re-run: same path, db untouched (no append).
    n_before = _db_count(output_dir)
    k2, dir2 = _evaluate_point(0, 300.0, 0.0, cfg)
    assert (k2, dir2) == (0, output_dir)
    assert _db_count(output_dir) == n_before


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------


def test_scan_runs_through_process_pool(tmp_path):
    """Points dispatched through a real spawn ProcessPoolExecutor all come back
    with their run dirs populated."""
    cfg = _small_cfg(str(tmp_path / "scan"))
    grid = [(300.0, 0.0), (400.0, 0.0)]
    done = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(_evaluate_point, k, t, p, cfg): k
            for k, (t, p) in enumerate(grid)
        }
        for fut in as_completed(futures):
            k, output_dir = fut.result()
            done[k] = output_dir
    assert set(done) == {0, 1}
    for output_dir in done.values():
        assert os.path.exists(os.path.join(output_dir, "equilibration.db"))
        assert os.path.exists(os.path.join(output_dir, "run_config.json"))


# ---------------------------------------------------------------------------
# Per-point diagnostics
# ---------------------------------------------------------------------------


def test_plot_point_results_writes_png(tmp_path):
    output_dir = str(tmp_path / "p")
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=2)
    equilibrate_point(300.0, 0.0, 4 * 27, init, 27, output_dir, seed=2)

    png = plot_point_results(output_dir)
    assert png == os.path.join(output_dir, "equilibration_diagnostics.png")
    assert os.path.exists(png)


# ---------------------------------------------------------------------------
# Full driver: main() end-to-end through the pool
# ---------------------------------------------------------------------------


def test_main_creates_point_dirs_and_diagnostics(tmp_path):
    """main() runs the grid through the real pool and leaves each point as a
    resumable dir with a convergence-diagnostics plot."""
    out_dir = str(tmp_path / "npt_scan")
    main(
        temp_grid=(300.0, 400.0),
        pressure_grid=(0.0,),
        n_particles=27,
        density=1.0,  # N=27 columnar caps ~1.30; stay under it
        num_steps=2 * 27,
        block_size=27,
        seed0=100,
        out_dir=out_dir,
        max_workers=2,
    )

    # grid is pressure-outer/temp-inner: k=0 -> (300, 0), k=1 -> (400, 0)
    for k, temp in enumerate((300.0, 400.0)):
        d = os.path.join(out_dir, f"T{temp:g}_P0", str(100 + k))
        assert os.path.exists(os.path.join(d, "equilibration.db"))
        assert os.path.exists(os.path.join(d, "run_config.json"))
        assert os.path.exists(os.path.join(d, "equilibration_diagnostics.png"))


def test_main_replicas_reuses_existing_and_adds_new(tmp_path):
    """Adding replicas to an existing scan reuses the original runs as replica 0
    (untouched, not recomputed) and only computes the new replicas, with seeds
    following seed0 + r*stride + k (stride = grid size)."""
    out_dir = str(tmp_path / "npt_scan")
    grid_kwargs = dict(
        temp_grid=(300.0, 400.0),
        pressure_grid=(0.0,),
        n_particles=27,
        density=1.0,  # N=27 columnar caps ~1.30; stay under it
        num_steps=2 * 27,
        block_size=27,
        seed0=100,
        out_dir=out_dir,
        max_workers=2,
    )

    # First pass: single replica -> seeds 100 (k=0), 101 (k=1).
    main(**grid_kwargs)
    rep0_dirs = {
        0: os.path.join(out_dir, "T300_P0", "100"),
        1: os.path.join(out_dir, "T400_P0", "101"),
    }
    before = {k: (_db_count(d), _last_row(d)) for k, d in rep0_dirs.items()}

    # Second pass: 3 replicas, same grid. stride = len(grid) = 2, so the seeds
    # are k=0 -> {100, 102, 104}, k=1 -> {101, 103, 105}.
    main(replicas=3, **grid_kwargs)

    # Replica 0 dirs are reused untouched: same db length and same last frame
    # (the idempotent skip means no fresh trajectory was appended).
    for k, d in rep0_dirs.items():
        n_after, row_after = _db_count(d), _last_row(d)
        assert n_after == before[k][0]
        assert row_after == before[k][1]

    # The two new replicas of each point exist with the expected seeds + artifacts.
    for point, k in (("T300_P0", 0), ("T400_P0", 1)):
        for r in (1, 2):
            seed = 100 + r * 2 + k
            d = os.path.join(out_dir, point, str(seed))
            assert os.path.exists(os.path.join(d, "equilibration.db"))
            assert os.path.exists(os.path.join(d, "run_config.json"))
            assert os.path.exists(os.path.join(d, "equilibration_diagnostics.png"))

    # Independent initial conditions -> the replicas are genuinely different runs,
    # not copies: replica 1 of point k=0 differs from its replica 0.
    assert _last_row(os.path.join(out_dir, "T300_P0", "102")) != before[0][1]


def _run_config(output_dir):
    import json

    with open(os.path.join(output_dir, "run_config.json")) as fh:
        return json.load(fh)


def test_main_defaults_to_columnar_packing(tmp_path):
    """The scan's default initializer is columnar, recorded in each run_config."""
    out_dir = str(tmp_path / "npt_scan")
    main(
        temp_grid=(300.0,),
        pressure_grid=(0.0,),
        n_particles=27,
        density=1.0,  # N=27 columnar caps ~1.30
        num_steps=2 * 27,
        block_size=27,
        seed0=100,
        out_dir=out_dir,
        max_workers=1,
    )
    cfg = _run_config(os.path.join(out_dir, "T300_P0", "100"))
    assert cfg["init"]["init_packing"] == "columnar"
    assert cfg["init"]["init_density"] == 1.0


def test_main_random_packing_is_honored(tmp_path):
    """packing='random' overrides the default and is recorded in the run_config."""
    out_dir = str(tmp_path / "npt_scan")
    main(
        temp_grid=(300.0,),
        pressure_grid=(0.0,),
        n_particles=27,
        packing="random",
        density=0.3,
        num_steps=2 * 27,
        block_size=27,
        seed0=100,
        out_dir=out_dir,
        max_workers=1,
    )
    cfg = _run_config(os.path.join(out_dir, "T300_P0", "100"))
    assert cfg["init"]["init_packing"] == "random"


def test_main_rejects_unknown_packing(tmp_path):
    with pytest.raises(ValueError, match="unknown packing"):
        main(packing="hexatic", out_dir=str(tmp_path / "scan"))


# ---------------------------------------------------------------------------
# Resuming: find_point_dirs / continue_point / extend_points
# ---------------------------------------------------------------------------


def test_find_point_dirs_discovers_resumable_points(tmp_path):
    """find_point_dirs returns exactly the T*_P*/{seed}/ dirs that hold both a
    run_config.json and an equilibration.db, and nothing else."""
    out_dir = str(tmp_path / "npt_scan")
    cfg = _small_cfg(out_dir)
    _evaluate_point(0, 300.0, 0.0, cfg)
    _evaluate_point(1, 400.0, 0.0, cfg)

    # a stray dir with no run/db must not be picked up
    os.makedirs(os.path.join(out_dir, "T999_P0", "999"), exist_ok=True)

    found = find_point_dirs(out_dir)
    assert found == sorted(
        [
            os.path.join(out_dir, "T300_P0", "100"),
            os.path.join(out_dir, "T400_P0", "101"),
        ]
    )


def test_continue_point_appends_and_advances_steps(tmp_path):
    """continue_point resumes in place: the db grows, the step axis advances
    monotonically past the original target, and the write-once run_config.json is
    preserved (not rewritten)."""
    output_dir = str(tmp_path / "T300_P0" / "1")
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=1)
    equilibrate_point(300.0, 0.0, 4 * 27, init, block_size=27, output_dir=output_dir, seed=1)

    n_before = _db_count(output_dir)
    step_before = _last_row(output_dir)["step"]
    with open(os.path.join(output_dir, "run_config.json")) as f:
        config_before = f.read()

    returned = continue_point(output_dir, extra_steps=4 * 27, block_size=27)
    assert returned == output_dir

    assert _db_count(output_dir) > n_before  # appended, not restarted
    assert _last_row(output_dir)["step"] >= step_before + 4 * 27  # absolute target

    # resuming preserves the original write-once config verbatim
    with open(os.path.join(output_dir, "run_config.json")) as f:
        assert f.read() == config_before


def test_continue_point_resets_vol_delt(tmp_path):
    """continue_point(vol_delt=X) overrides the carried volume width before resuming.

    The short extension stays under one fresh-window of volume moves, so the gated
    tuner does not fire and the reset value is what gets recorded -- confirming the
    override threads through from_equilibration rather than the db's tuned value.
    """
    output_dir = str(tmp_path / "T300_P0" / "1")
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=1)
    equilibrate_point(300.0, 0.0, 4 * 27, init, block_size=27, output_dir=output_dir, seed=1)

    with connect(os.path.join(output_dir, "equilibration.db")) as db:
        carried = db.get(db.count()).vol_delta
    reset_to = round(carried + 0.123, 6)  # a value the db could not have held

    continue_point(output_dir, extra_steps=4 * 27, block_size=27, vol_delt=reset_to)

    with connect(os.path.join(output_dir, "equilibration.db")) as db:
        assert db.get(db.count()).vol_delta == reset_to


def test_continue_point_deterministic(tmp_path):
    """Continuing the same finished point by the same step budget is reproducible
    -- the per-point RNG reseed makes the extension independent of process order."""

    def run(dirname):
        output_dir = str(tmp_path / dirname / "7")
        init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=7)
        equilibrate_point(300.0, 0.0, 4 * 27, init, 27, output_dir, seed=7)
        continue_point(output_dir, extra_steps=4 * 27, block_size=27)
        return _last_row(output_dir)

    a = run("a")
    b = run("b")
    assert a["total_energy"] == b["total_energy"]
    assert a["vol"] == b["vol"]


def test_extend_points_through_pool_replots(tmp_path):
    """extend_points continues a set of point dirs through the real spawn pool and
    re-renders each one's convergence diagnostics over the longer trajectory."""
    out_dir = str(tmp_path / "npt_scan")
    cfg = _small_cfg(out_dir)
    _evaluate_point(0, 300.0, 0.0, cfg)
    _evaluate_point(1, 400.0, 0.0, cfg)
    dirs = find_point_dirs(out_dir)

    counts_before = {d: _db_count(d) for d in dirs}
    done = extend_points(dirs, extra_steps=2 * 27, block_size=27, max_workers=2)

    assert sorted(done) == sorted(dirs)
    for d in dirs:
        assert _db_count(d) > counts_before[d]
        assert os.path.exists(os.path.join(d, "equilibration_diagnostics.png"))
