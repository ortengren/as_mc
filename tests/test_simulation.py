import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np

from asmcmc.simulation.run import run_grid, point_dir
from asmcmc.simulation.report import write_artifacts
from asmcmc.simulation.plots import write_plots, write_trace_plots


# A small, fast grid shared by the end-to-end tests below.
GRID_KWARGS = dict(
    temps=[150.0, 300.0],
    pressures=[1e-6],
    n_steps=200,
    block_size=50,
    num_eq_steps=100,
    buffer_size=1,
    n_particles=32,
    density=0.25,
    seed=0,
    rdf_r_max=5.0,
    rdf_bins=20,
    progress=False,
)

ARTIFACTS = [
    "run_config.json",
    "observables.json",
    "observables.npz",
    "report.md",
    "energy_trace.png",
    "volume_trace.png",
    "acceptance_trace.png",
    "nematic_order.png",
    "rdf.png",
    "orientational_correlation.png",
]


def test_run_grid_writes_full_artifact_set(tmp_path):
    """Every (T, P) point gets a leaf dir with the complete artifact set."""
    out_dir = str(tmp_path / "grid")
    results = run_grid(out_dir=out_dir, **GRID_KWARGS)

    # one result per (T, P) combination
    assert set(results.keys()) == {(150.0, 1e-6), (300.0, 1e-6)}

    for temp in GRID_KWARGS["temps"]:
        leaf = point_dir(out_dir, temp, 1e-6)
        assert os.path.isdir(leaf)
        for name in ARTIFACTS:
            assert os.path.exists(os.path.join(leaf, name)), f"missing {name} in {leaf}"


def test_observables_are_finite(tmp_path):
    """The scalar observables in observables.json must all be finite."""
    out_dir = str(tmp_path / "grid")
    run_grid(out_dir=out_dir, **GRID_KWARGS)

    leaf = point_dir(out_dir, 150.0, 1e-6)
    with open(os.path.join(leaf, "observables.json")) as f:
        obs = json.load(f)

    for key in ("avg_energy", "heat_capacity", "nematic_S", "energy_per_particle"):
        assert np.isfinite(obs[key]), f"{key} is not finite: {obs[key]}"


def test_run_config_records_provenance(tmp_path):
    """run_config.json captures temp/pressure/ensemble + acceptance rates."""
    out_dir = str(tmp_path / "grid")
    run_grid(out_dir=out_dir, **GRID_KWARGS)

    leaf = point_dir(out_dir, 300.0, 1e-6)
    with open(os.path.join(leaf, "run_config.json")) as f:
        config = json.load(f)

    assert config["temp"] == 300.0
    assert config["pressure"] == 1e-6
    assert config["ensemble"] == "npt"
    assert config["n_particles"] == 32
    assert set(config["acceptance"]) == {"position", "orientation", "volume"}


def test_repeat_writes_replicas_and_summary(tmp_path):
    """repeat>1 gives per-replica dirs (distinct seeds) + an aggregate summary."""
    out_dir = str(tmp_path / "grid")
    kwargs = dict(GRID_KWARGS, temps=[200.0], repeat=3)
    results = run_grid(out_dir=out_dir, **kwargs)

    leaf = point_dir(out_dir, 200.0, 1e-6)

    # one rep dir per replica, each with the full artifact set
    seeds = []
    for i in range(3):
        rep = os.path.join(leaf, "rep{:02d}".format(i))
        assert os.path.isdir(rep)
        for name in ARTIFACTS:
            assert os.path.exists(os.path.join(rep, name)), f"missing {name} in {rep}"
        with open(os.path.join(rep, "run_config.json")) as f:
            seeds.append(json.load(f)["seed"])
    # different initializations -> distinct, reproducible seeds (base seed 0 + i)
    assert seeds == [0, 1, 2]

    # cross-replica summary at the point dir
    for name in ("summary.json", "summary.md"):
        assert os.path.exists(os.path.join(leaf, name))
    with open(os.path.join(leaf, "summary.json")) as f:
        summary = json.load(f)
    assert summary["n_replicas"] == 3
    assert summary["seeds"] == [0, 1, 2]
    stats = summary["observables"]["energy_per_particle"]
    assert stats["n"] == 3
    assert np.isfinite(stats["mean"]) and np.isfinite(stats["std"])
    assert stats["std"] >= 0.0

    # the returned value exposes both replicas and the summary
    assert set(results[(200.0, 1e-6)]) == {"replicas", "summary"}
    assert len(results[(200.0, 1e-6)]["replicas"]) == 3
