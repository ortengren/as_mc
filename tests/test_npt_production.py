import os

import pytest

from ase.db import connect

from asmcmc.npt_equilibration import _evaluate_point, plot_point_results
from asmcmc.npt_production import produce_point, produce_points


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


def _db_count(output_dir, db_name):
    with connect(os.path.join(output_dir, db_name)) as db:
        return db.count()


def _equilibrate_point(out_dir, k=0, temp=300.0, pressure=0.0):
    """Build one equilibrated point dir (equilibration.db + run_config.json)."""
    cfg = _small_cfg(out_dir)
    _, d = _evaluate_point(k, temp, pressure, cfg)
    return d


# ---------------------------------------------------------------------------
# produce_point: writes simulation.db, leaves equilibration intact, idempotent
# ---------------------------------------------------------------------------


def test_produce_point_writes_simulation_db(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    eq_count = _db_count(d, "equilibration.db")

    produce_point(d, num_steps=2 * 27, block_size=27)

    # production wrote its own db with recorded frames...
    assert os.path.exists(os.path.join(d, "simulation.db"))
    assert _db_count(d, "simulation.db") >= 1
    # ...and did not touch the equilibration db.
    assert _db_count(d, "equilibration.db") == eq_count


def test_produce_point_step_axis_starts_fresh(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    produce_point(d, num_steps=2 * 27, block_size=27)

    with connect(os.path.join(d, "simulation.db")) as db:
        steps = sorted(row.step for row in db.select())
    # production zeroes step_count, so its axis is independent of equilibration's.
    assert steps[0] <= 27
    assert steps == sorted(steps)


def test_produce_point_idempotent_skip(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    produce_point(d, num_steps=2 * 27, block_size=27)
    n_before = _db_count(d, "simulation.db")

    # a dir that already has a simulation.db is skipped, not appended to.
    d2 = produce_point(d, num_steps=2 * 27, block_size=27)
    assert d2 == d
    assert _db_count(d, "simulation.db") == n_before


def test_produce_point_default_block_size_is_per_sweep(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    # block_size=None -> one frame per sweep (= n_particles = 27); 2 sweeps -> 2 frames.
    produce_point(d, num_steps=2 * 27, block_size=None)
    assert _db_count(d, "simulation.db") == 2


# ---------------------------------------------------------------------------
# produce_points: parallel dispatch + diagnostics
# ---------------------------------------------------------------------------


def test_produce_points_through_pool(tmp_path):
    out_dir = str(tmp_path / "scan")
    d0 = _equilibrate_point(out_dir, k=0, temp=300.0)
    d1 = _equilibrate_point(out_dir, k=1, temp=400.0)
    # render equilibration diagnostics so we can assert production doesn't clobber them.
    for d in (d0, d1):
        plot_point_results(d)

    done = produce_points([d0, d1], num_steps=2 * 27, block_size=27, max_workers=2)

    assert sorted(done) == sorted([d0, d1])
    for d in (d0, d1):
        assert os.path.exists(os.path.join(d, "simulation.db"))
        # production diagnostics written to their own file...
        assert os.path.exists(os.path.join(d, "production_diagnostics.png"))
        # ...and the equilibration diagnostics survive.
        assert os.path.exists(os.path.join(d, "equilibration_diagnostics.png"))
