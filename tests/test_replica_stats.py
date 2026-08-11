"""Replica aggregation: per-replica ESS stats and between-replica error bars."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest
import numpy as np

from ase.db import connect

from asmcmc.npt_production import produce_point
from asmcmc.replica_stats import replica_observables, aggregate, OBSERVABLES

from asmcmc.npt_equilibration import _evaluate_point


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



def _equilibrate_replica(out_dir, r, k=0, temp=300.0, pressure=0.0):
    """Build one replica seed dir for a (T, P) point (distinct seed per r)."""
    cfg = _small_cfg(out_dir)
    cfg["replica_stride"] = 10  # r=0 -> seed 100, r=1 -> 110, ... (no collision)
    _, d = _evaluate_point(k, temp, pressure, cfg, r)
    return d



def test_replica_observables_returns_ess_stats(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    produce_point(d, num_steps=4 * 27, block_size=27)

    stats = replica_observables(d)
    assert set(stats) == set(OBSERVABLES)
    for name, s in stats.items():
        assert {"mean", "std", "ess", "tau", "sem", "num_samples"} <= set(s)
        assert s["num_samples"] >= 1


def test_aggregate_single_replica_uses_ess_fallback(tmp_path):
    out_dir = str(tmp_path / "scan")
    d = _equilibrate_point(out_dir)
    produce_point(d, num_steps=4 * 27, block_size=27)

    df = aggregate(out_dir, plot=False)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_replicas"] == 1
    assert row["temp"] == 300.0 and row["pressure"] == 0.0
    # single replica -> no between-replica spread, falls back to within-chain ESS
    for name in OBSERVABLES:
        assert row[f"{name}_method"] == "ess_single"
        assert row[f"{name}_sem"] >= 0.0
    assert os.path.exists(os.path.join(out_dir, "npt_production.csv"))


def test_aggregate_between_replica_error_and_consistency(tmp_path):
    out_dir = str(tmp_path / "scan")
    # two independent replicas of the *same* (T, P) point (sibling seed dirs)
    d0 = _equilibrate_replica(out_dir, r=0)
    d1 = _equilibrate_replica(out_dir, r=1)
    assert os.path.dirname(d0) == os.path.dirname(d1)  # same T_P point dir
    for d in (d0, d1):
        produce_point(d, num_steps=4 * 27, block_size=27)

    df = aggregate(out_dir, plot=True)

    assert len(df) == 1  # the two replicas collapse into one point row
    row = df.iloc[0]
    assert row["n_replicas"] == 2
    for name in OBSERVABLES:
        assert row[f"{name}_method"] == "between_replica"
        # consistency ratio (between-replica var / within-chain prediction) is finite
        assert np.isfinite(row[f"{name}_consistency"])
    assert os.path.exists(os.path.join(out_dir, "npt_production.png"))


