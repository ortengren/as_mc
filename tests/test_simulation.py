import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np

from asmcmc.simulation.run import run_grid, equilibrate_grid, point_dir, cli
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


TRACE_PLOTS = [
    "energy_trace.png",
    "volume_trace.png",
    "acceptance_trace.png",
    "nematic_order.png",
]


def test_equilibrate_grid_writes_db_and_trace_plots(tmp_path):
    """equilibrate_grid leaves an equilibration.db + trace plots per (T, P)."""
    out_dir = str(tmp_path / "eq")
    eq_kwargs = dict(
        temps=[200.0],
        pressures=[1e-6],
        num_eq_steps=100,
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        progress=False,
    )
    results = equilibrate_grid(out_dir=out_dir, **eq_kwargs)

    leaf = point_dir(out_dir, 200.0, 1e-6)
    assert os.path.exists(os.path.join(leaf, "equilibration.db"))
    assert os.path.exists(os.path.join(leaf, "equilibration_config.json"))
    for name in TRACE_PLOTS:
        assert os.path.exists(os.path.join(leaf, name)), f"missing {name} in {leaf}"

    assert set(results[(200.0, 1e-6)]) == {"config", "plots", "db_path"}


def test_equilibrate_grid_repeat_writes_replica_dirs(tmp_path):
    """equilibrate_grid(repeat=n) gives rep dirs with distinct seeds + dbs."""
    out_dir = str(tmp_path / "eq")
    eq_kwargs = dict(
        temps=[200.0],
        pressures=[1e-6],
        num_eq_steps=100,
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        progress=False,
    )
    results = equilibrate_grid(out_dir=out_dir, repeat=3, **eq_kwargs)

    leaf = point_dir(out_dir, 200.0, 1e-6)
    seeds = []
    for i in range(3):
        rep = os.path.join(leaf, "rep{:02d}".format(i))
        assert os.path.exists(os.path.join(rep, "equilibration.db"))
        for name in TRACE_PLOTS:
            assert os.path.exists(os.path.join(rep, name)), f"missing {name} in {rep}"
        with open(os.path.join(rep, "equilibration_config.json")) as f:
            seeds.append(json.load(f)["seed"])
    assert seeds == [0, 1, 2]  # distinct, reproducible per-replica seeds

    assert set(results[(200.0, 1e-6)]) == {"replicas"}
    assert len(results[(200.0, 1e-6)]["replicas"]) == 3


def test_resume_production_with_repeat(tmp_path):
    """run_grid(resume_from=DIR, repeat=n) restarts each replica from its eq db."""
    eq_dir = str(tmp_path / "eq")
    equilibrate_grid(
        out_dir=eq_dir,
        temps=[200.0],
        pressures=[1e-6],
        num_eq_steps=100,
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        repeat=2,
        progress=False,
    )

    prod_dir = str(tmp_path / "prod")
    results = run_grid(
        out_dir=prod_dir,
        resume_from=eq_dir,
        repeat=2,
        **dict(GRID_KWARGS, temps=[200.0]),
    )

    leaf = point_dir(prod_dir, 200.0, 1e-6)
    for i in range(2):
        rep = os.path.join(leaf, "rep{:02d}".format(i))
        for name in ARTIFACTS:
            assert os.path.exists(os.path.join(rep, name)), f"missing {name} in {rep}"
        with open(os.path.join(rep, "run_config.json")) as f:
            config = json.load(f)
        assert config["num_eq_steps"] is None  # production-only restart
        assert config["initializer"] == "FrameInitializer"
        # replica i resumed from the matching rep{i} equilibration
        assert "rep{:02d}".format(i) in config["resumed_from"]

    # cross-replica summary still written
    assert os.path.exists(os.path.join(leaf, "summary.json"))
    assert set(results[(200.0, 1e-6)]) == {"replicas", "summary"}


def test_resume_equilibration_appends_and_extends_steps(tmp_path):
    """equilibrate_grid(resume_from=DIR) continues in place with more steps."""
    from ase.db import connect

    out_dir = str(tmp_path / "eq")
    eq_kwargs = dict(
        temps=[200.0],
        pressures=[1e-6],
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        progress=False,
    )
    equilibrate_grid(out_dir=out_dir, num_eq_steps=100, **eq_kwargs)

    leaf = point_dir(out_dir, 200.0, 1e-6)
    eq_db = os.path.join(leaf, "equilibration.db")

    def max_step(path):
        with connect(path) as db:
            return max(row.key_value_pairs["step"] for row in db.select())

    first_max = max_step(eq_db)

    # resume in place for 100 more steps -> db grows, step counter continues
    equilibrate_grid(resume_from=out_dir, num_eq_steps=100, **eq_kwargs)

    second_max = max_step(eq_db)
    assert second_max > first_max

    with open(os.path.join(leaf, "equilibration_config.json")) as f:
        config = json.load(f)
    assert config["start_step"] == first_max
    assert config["resumed_from"].endswith("equilibration.db")


def test_resume_equilibration_subset_of_points(tmp_path):
    """points= resumes only the chosen (T, P), leaving the others untouched."""
    from ase.db import connect

    out_dir = str(tmp_path / "eq")
    eq_kwargs = dict(
        temps=[200.0, 300.0],
        pressures=[1e-6],
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        progress=False,
    )
    equilibrate_grid(out_dir=out_dir, num_eq_steps=100, **eq_kwargs)

    def max_step(temp):
        db = os.path.join(point_dir(out_dir, temp, 1e-6), "equilibration.db")
        with connect(db) as conn:
            return max(row.key_value_pairs["step"] for row in conn.select())

    before = {t: max_step(t) for t in (200.0, 300.0)}

    # resume ONLY the 200 K point
    equilibrate_grid(
        resume_from=out_dir,
        points=[(200.0, 1e-6)],
        num_eq_steps=100,
        **eq_kwargs,
    )

    after = {t: max_step(t) for t in (200.0, 300.0)}
    assert after[200.0] > before[200.0]  # resumed
    assert after[300.0] == before[300.0]  # untouched


def test_resume_from_runs_production_only(tmp_path):
    """run_grid(resume_from=...) starts from the equilibrated frame, no eq pass."""
    eq_dir = str(tmp_path / "eq")
    equilibrate_grid(
        out_dir=eq_dir,
        temps=[200.0],
        pressures=[1e-6],
        num_eq_steps=100,
        block_size=50,
        buffer_size=1,
        n_particles=32,
        density=0.25,
        seed=0,
        progress=False,
    )

    prod_dir = str(tmp_path / "prod")
    run_grid(
        out_dir=prod_dir,
        resume_from=eq_dir,
        **dict(GRID_KWARGS, temps=[200.0]),
    )

    leaf = point_dir(prod_dir, 200.0, 1e-6)
    for name in ARTIFACTS:
        assert os.path.exists(os.path.join(leaf, name)), f"missing {name} in {leaf}"

    with open(os.path.join(leaf, "run_config.json")) as f:
        config = json.load(f)
    # production-only restart: no equilibration was run for this point
    assert config["num_eq_steps"] is None
    assert config["resumed_from"].endswith("equilibration.db")
    # the run must actually start from the equilibrated frame, not a fresh config
    assert config["initializer"] == "FrameInitializer"


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


def _max_step(db_path):
    from ase.db import connect

    with connect(db_path) as db:
        return max(row.key_value_pairs["step"] for row in db.select())


def test_cli_equilibrate_continue_produce(tmp_path):
    """The equilibrate / continue-eq / produce subcommands wire end-to-end."""
    eq_dir = str(tmp_path / "eq")
    grid = [
        "--temps", "200",
        "--pressures", "1e-6",
        "--num-eq-steps", "80",
        "--block-size", "40",
        "--buffer-size", "1",
        "--n-particles", "32",
        "--density", "0.25",
        "--seed", "0",
        "--no-progress",
    ]

    # 1. fresh equilibration grid
    cli(["equilibrate", "--out-dir", eq_dir, *grid])
    eq_db = os.path.join(point_dir(eq_dir, 200.0, 1e-6), "equilibration.db")
    assert os.path.exists(eq_db)
    first_max = _max_step(eq_db)

    # 2. continue that equilibration in place -> db grows
    cli(["continue-eq", "--from", eq_dir, *grid])
    assert _max_step(eq_db) > first_max

    # 3. produce from the equilibrated configs -> production-only restart
    prod_dir = str(tmp_path / "prod")
    cli(["produce", "--from", eq_dir, "--out-dir", prod_dir, "--n-steps", "120", *grid])
    leaf = point_dir(prod_dir, 200.0, 1e-6)
    for name in ARTIFACTS:
        assert os.path.exists(os.path.join(leaf, name)), f"missing {name} in {leaf}"
    with open(os.path.join(leaf, "run_config.json")) as f:
        config = json.load(f)
    assert config["num_eq_steps"] is None
    assert config["initializer"] == "FrameInitializer"
