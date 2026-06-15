"""Runner / CLI for MC simulations: build a sampler over a (T, P) grid and write
a self-describing artifact set per state point.

Ties together :class:`asmcmc.metropolis.MetropolisCalculator` (the sampler),
:mod:`asmcmc.simulation.report` (config + observables + markdown) and
:mod:`asmcmc.simulation.plots` (diagnostic PNGs), the sampler analogue of
:mod:`asmcmc.fitting.run`. Run with ``python -m asmcmc.simulation.run``.

The knobs that are constant across the grid (potential, neighbour list, ensemble,
particle count, move-tuning, analysis ranges) are bundled into a single
:class:`SimulationConfig`; only the things that genuinely vary per point/replica
(temperature, pressure, step counts, seed, restart frame) are threaded as call
arguments. ``run_grid``/``equilibrate_grid`` share one generic grid driver
(:func:`_drive_grid`) and one calculator builder (:func:`_build_calculator`).
"""

import argparse
import datetime
import json
import os
import traceback
from dataclasses import dataclass, replace

import numpy as np
from ase.db import connect

from asmcmc.metropolis import MetropolisCalculator
from asmcmc.initialize import RandomLatticeInitializer
from asmcmc.potentials import GBQPotential
from asmcmc.simulation.report import run_config, write_artifacts, write_summary
from asmcmc.simulation.plots import write_plots, write_trace_plots

DEFAULT_TEMPS = (100.0, 200.0, 300.0, 400.0)
DEFAULT_PRESSURES = (5e-6, 1e-5, 5e-5)

# 1 atm = 6.32e-6 eV / Å^3


@dataclass(frozen=True)
class SimulationConfig:
    """Knobs held constant across every (T, P) point and replica of a grid.

    Everything here describes *how* a sampler is built, tuned and analysed --
    not *which* state point or *how long* it runs (those vary per call and stay
    as arguments to the ``*_point`` / ``*_grid`` functions). Frozen so a replica
    can be derived cheaply and safely via :func:`dataclasses.replace` (only the
    ``seed`` changes between replicas).
    """

    # sampler construction
    potential: object = None
    nl_radius: float = 15.0
    nl_skin: float = 1.0
    npt_ensemble: bool = True
    n_particles: int = 210
    density: float = 0.3
    seed: int = None
    # block recording + move tuning
    block_size: int = 250
    buffer_size: int = 4
    max_scale: float = 1.1
    min_scale: float = 0.9
    progress: bool = True
    # production analysis
    recompute_energy: bool = True
    rdf_r_max: float = 15.0
    rdf_bins: int = 100


def default_out_dir(prefix="npt"):
    """Timestamped output root, e.g. ``results/simulations/npt_2026-06-14T10-15``.

    ``prefix`` labels the run kind (``"npt"``/``"nvt"`` production, ``"eq"`` for an
    equilibrate-only grid).
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    return os.path.join("results", "simulations", prefix + "_" + stamp)


def point_dir(out_dir, temp, pressure):
    """Leaf dir for one (T, P) point: ``<out_dir>/T{temp}_P{pressure}``."""
    return os.path.join(out_dir, "T{:g}_P{:g}".format(temp, pressure))


def _parse_points(point_strs):
    """Parse CLI ``--points`` strings (``"T,P"``) into ``[(temp, pressure), ...]``.

    Returns ``None`` when no points were given, so callers fall back to the
    ``temps`` x ``pressures`` product.
    """
    if not point_strs:
        return None
    points = []
    for s in point_strs:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("--points expects 'T,P' pairs, got {!r}".format(s))
        points.append((float(parts[0]), float(parts[1])))
    return points


def _grid_points(temps, pressures, points=None):
    """The list of (temp, pressure) state points to run.

    By default the full Cartesian product of ``temps`` x ``pressures``. An
    explicit ``points`` iterable of ``(temp, pressure)`` pairs overrides it,
    letting callers target an arbitrary subset of the grid -- e.g. resuming only
    the points that have not yet converged.
    """
    if points is not None:
        return [(float(t), float(p)) for t, p in points]
    return [(t, p) for t in temps for p in pressures]


def _restart_state(db_path):
    """Final equilibrated frame + tuned trial-move deltas from an equilibration db.

    Reads the last block written to ``db_path`` (chronological insertion order)
    and rebuilds an ``ase.Atoms`` carrying the per-particle ``c_q``/``or_vec``
    arrays (stored in the row ``data``, since ase.db does not round-trip custom
    arrays) plus the adapted ``pos/or/vol`` move widths (from the row's scalar
    data). Returns ``(atoms, deltas, last_step)`` -- ``last_step`` lets
    :func:`equilibrate_grid` continue the step counter so a resumed run's trace
    plots stay monotonic. Used to resume production (:func:`run_grid`) or further
    equilibration without repeating work or losing the tuning.
    """
    last = None
    with connect(db_path) as db:
        for row in db.select():
            last = row
    if last is None:
        raise ValueError("no frames in {}".format(db_path))

    atoms = last.toatoms()
    atoms.set_pbc(True)
    atoms.new_array("c_q", np.asarray(last.data["c_q"]))
    atoms.new_array("or_vec", np.asarray(last.data["or_vec"]))

    kv = last.key_value_pairs
    deltas = {
        "pos": kv.get("pos_delta"),
        "or": kv.get("or_delta"),
        "vol": kv.get("vol_delta"),
    }
    return atoms, deltas, int(kv.get("step", 0))


def _replica_seed(seed, replica):
    """Per-replica config seed: distinct-but-reproducible, or ``None`` for random.

    A fixed ``seed`` is offset by the replica index so each replica gets a
    different initial configuration that still reproduces; ``seed=None`` leaves
    every replica genuinely random (independent draws).
    """
    return None if seed is None else seed + replica


def _replica_dir(base, replica, repeat):
    """Per-replica leaf dir: the point dir itself for ``repeat == 1``, else
    ``<base>/rep{replica:02d}``. Shared by the equilibration and production grids
    so a replica's equilibration and its production restart land in matching
    layouts."""
    return base if repeat == 1 else os.path.join(base, "rep{:02d}".format(replica))


def _build_calculator(temp, pressure, cfg, dest_dir, init_frame=None, deltas=None):
    """Construct the :class:`MetropolisCalculator` for one (T, P) point.

    Shared by :func:`run_point` and :func:`equilibrate_point`. ``deltas`` (from a
    restart) seeds the trial-move widths; ``init_frame`` (a restart frame) is
    threaded through so the calculator starts from it -- otherwise a fresh
    :class:`RandomLatticeInitializer` builds a config from ``cfg``. Providing the
    frame as ``init_frame`` (not via an initializer) is essential: the calculator
    falls back to a random config if it is ``None``, silently discarding the
    equilibration.
    """
    delta_kwargs = {}
    if deltas is not None:
        for key, kwarg in (("pos", "pos_delt"), ("or", "or_delt"), ("vol", "vol_delt")):
            if deltas.get(key) is not None:
                delta_kwargs[kwarg] = float(deltas[key])
    initializer = (
        RandomLatticeInitializer(cfg.n_particles, cfg.density, cfg.seed)
        if init_frame is None
        else None
    )
    return MetropolisCalculator(
        temp,
        pressure,
        init_frame=init_frame,
        initializer=initializer,
        potential=cfg.potential,
        nl_radius=cfg.nl_radius,
        nl_skin=cfg.nl_skin,
        output_dir=dest_dir,
        npt_ensemble=cfg.npt_ensemble,
        **delta_kwargs,
    )


def run_point(
    temp,
    pressure,
    dest_dir,
    cfg,
    n_steps,
    num_eq_steps,
    init_frame=None,
    deltas=None,
    resumed_from=None,
):
    """Run one (T, P) simulation into ``dest_dir`` and write its artifact set.

    Builds the sampler from ``cfg`` (reproducible for a fixed ``cfg.seed``,
    independent random init when ``cfg.seed is None``), runs
    ``calculate_trajectory``, then hands the finished ``simulation.db`` to
    :func:`asmcmc.simulation.report.write_artifacts` and
    :func:`asmcmc.simulation.plots.write_plots`. Returns the in-memory
    ``{'config', 'observables'}`` from the report.

    When ``init_frame`` is supplied (a restart from a prior equilibration) it is
    used instead of generating a config, ``deltas`` seeds the trial-move widths
    with the equilibration's tuned values, and the caller should pass
    ``num_eq_steps=None`` for a production-only run; ``resumed_from`` records the
    source db path in the provenance.
    """
    metro = _build_calculator(temp, pressure, cfg, dest_dir, init_frame, deltas)
    # capture the trial-move widths as initially configured (block_update adapts
    # them in place during equilibration, so record before running).
    initial_deltas = {
        "pos": float(metro.pos_delt),
        "or": float(metro.or_delt),
        "vol": float(metro.vol_delt),
    }
    metro.calculate_trajectory(
        n_steps,
        block_size=cfg.block_size,
        num_eq_steps=num_eq_steps,
        buffer_size=cfg.buffer_size,
        progress=cfg.progress,
    )

    db_path = os.path.join(dest_dir, "simulation.db")
    meta = {
        "n_steps": n_steps,
        "block_size": cfg.block_size,
        "num_eq_steps": num_eq_steps,
        "seed": cfg.seed,
        "density_init": cfg.density,
        "initial_deltas": initial_deltas,
        "recompute_energy": cfg.recompute_energy,
        "rdf_r_max": cfg.rdf_r_max,
        "rdf_bins": cfg.rdf_bins,
        "resumed_from": resumed_from,
    }
    artifacts = write_artifacts(
        dest_dir, metro, db_path, meta=meta, progress=cfg.progress
    )
    write_plots(dest_dir, db_path, artifacts["observables"])
    return artifacts


def equilibrate_point(
    temp,
    pressure,
    dest_dir,
    cfg,
    num_eq_steps,
    start_step=0,
    init_frame=None,
    deltas=None,
    resumed_from=None,
):
    """Equilibrate one (T, P) point into ``dest_dir`` and write trace diagnostics.

    Builds the sampler from ``cfg``, runs
    :meth:`MetropolisCalculator.equilibrate` into ``dest_dir/equilibration.db``,
    writes the trajectory trace PNGs (including ``volume_trace.png`` for the
    cell-volume plateau check) and an ``equilibration_config.json`` provenance
    file. Returns ``{'config', 'plots', 'db_path'}``.

    To **resume** an equilibration that did not run long enough, pass the prior
    final frame as ``init_frame`` (with its tuned ``deltas`` and the last recorded
    ``start_step``): equilibration continues for ``num_eq_steps`` *more* steps,
    appending to ``dest_dir/equilibration.db`` with a continued step counter so the
    trace plots cover the whole combined run. ``resumed_from`` records the source.
    """
    metro = _build_calculator(temp, pressure, cfg, dest_dir, init_frame, deltas)
    # Continue the step counter from where the prior equilibration stopped so the
    # appended blocks keep increasing and the combined trace stays monotonic.
    metro.step_count = start_step
    metro.equilibrate(
        start_step + num_eq_steps,
        block_size=cfg.block_size,
        buffer_size=cfg.buffer_size,
        max_scale=cfg.max_scale,
        min_scale=cfg.min_scale,
        progress=cfg.progress,
    )

    db_path = os.path.join(dest_dir, "equilibration.db")
    # equilibrate() does not clear the decision lists, so run_config's acceptance
    # rates reflect the equilibration moves -- exactly what we want here.
    config = run_config(
        metro,
        db_path,
        meta={
            "num_eq_steps": num_eq_steps,
            "block_size": cfg.block_size,
            "seed": cfg.seed,
            "density_init": cfg.density,
            "start_step": start_step,
            "resumed_from": resumed_from,
        },
    )
    with open(os.path.join(dest_dir, "equilibration_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    plots = write_trace_plots(dest_dir, db_path)
    return {"config": config, "plots": plots, "db_path": db_path}


def _drive_grid(
    work_fn,
    pack_point,
    temps,
    pressures,
    out_dir,
    cfg,
    repeat,
    points,
    resume_from,
    label,
):
    """Run ``work_fn`` over a (T, P) grid x ``repeat`` replicas, with resume/skip.

    The single orchestration loop shared by :func:`run_grid` and
    :func:`equilibrate_grid`. For each state point and replica it computes the
    leaf dir (:func:`_replica_dir`), derives the per-replica seed
    (:func:`_replica_seed` via :func:`dataclasses.replace`), optionally loads the
    restart state from a prior grid (:func:`_restart_state`), and calls::

        work_fn(temp, pressure, dest, cfg_i, init_frame, deltas, start_step,
                resumed_from)

    Per-replica failures (and unreadable restart dbs) are logged and skipped so
    one bad run does not abort the grid. Each point's surviving replica results
    are handed to ``pack_point(rep_results, base, temp, pressure)`` to build the
    returned per-point value. ``label`` names the kind of work in the progress
    prints. Returns ``{(temp, pressure): packed}`` for points with >=1 replica.
    """
    results = {}
    for temp, pressure in _grid_points(temps, pressures, points):
        print("=== {} T={:g} K, P={:g} eV/A^3 ===".format(label, temp, pressure))
        base = point_dir(out_dir, temp, pressure)

        rep_results = []
        for i in range(repeat):
            dest = _replica_dir(base, i, repeat)
            if repeat > 1:
                print("--- replica {}/{} ---".format(i + 1, repeat))
            cfg_i = replace(cfg, seed=_replica_seed(cfg.seed, i))

            init_frame, deltas, start_step, resumed_from = None, None, 0, None
            if resume_from is not None:
                src = _replica_dir(point_dir(resume_from, temp, pressure), i, repeat)
                resumed_from = os.path.join(src, "equilibration.db")
                try:
                    init_frame, deltas, start_step = _restart_state(resumed_from)
                except Exception:
                    print(
                        "!!! T={:g}, P={:g}, replica {}: cannot resume from {}; "
                        "skipping:".format(temp, pressure, i, resumed_from)
                    )
                    traceback.print_exc()
                    continue

            try:
                rep_results.append(
                    work_fn(
                        temp,
                        pressure,
                        dest,
                        cfg_i,
                        init_frame,
                        deltas,
                        start_step,
                        resumed_from,
                    )
                )
            except Exception:
                print(
                    "!!! T={:g}, P={:g}, replica {} failed; skipping:".format(
                        temp, pressure, i
                    )
                )
                traceback.print_exc()

        if rep_results:
            results[(temp, pressure)] = pack_point(rep_results, base, temp, pressure)
    return results


def equilibrate_grid(
    temps=DEFAULT_TEMPS,
    pressures=DEFAULT_PRESSURES,
    out_dir=None,
    num_eq_steps=20_000,
    block_size=500,
    buffer_size=10,
    potential=None,
    nl_radius=15,
    nl_skin=2.0,
    npt_ensemble=True,
    n_particles=210,
    density=0.3,
    seed=None,
    repeat=1,
    progress=True,
    max_scale=1.1,
    min_scale=0.9,
    resume_from=None,
    points=None,
):
    """Equilibrate every (T, P) point and write per-point convergence diagnostics.

    The pre-production half of the workflow: for each (T, P) it equilibrates a
    fresh config into ``<out_dir>/T{temp}_P{pressure}/equilibration.db`` and
    renders the trajectory trace PNGs so the cell-volume plateau (and energy /
    acceptance / nematic order) can be checked per point before committing to
    production. Pass ``out_dir`` back to :func:`run_grid` as ``resume_from=`` to
    start production from these equilibrated configs without repeating the
    equilibration or losing its tuned move widths.

    ``repeat`` equilibrates that many independent replicas **with different initial
    configurations** (seeds ``seed + i``) per point, each in its own
    ``rep{i:02d}/`` subdir, so the replicas you will later run production for can
    each be checked (and resumed) on their own. With ``repeat == 1`` (default) the
    single equilibration lands directly in the point dir.

    ``resume_from`` is the ``out_dir`` of a prior equilibration grid that did not
    run long enough: each replica continues **in place** from its final frame (and
    tuned move widths) for ``num_eq_steps`` *more* steps, appending to its existing
    ``equilibration.db`` so the regenerated trace plots show the whole combined
    run. ``out_dir`` is ignored in this mode (it equals ``resume_from``); pass the
    same ``repeat`` used to create the grid.

    ``points`` is an explicit iterable of ``(temp, pressure)`` pairs that overrides
    the ``temps`` x ``pressures`` product -- combine it with ``resume_from`` to
    resume only the subset of points that have not yet converged. Per-replica
    failures are logged and skipped. Returns ``{(temp, pressure): value}`` where
    ``value`` is the single ``{'config', 'plots', 'db_path'}`` dict when
    ``repeat == 1``, else ``{'replicas': [...]}``.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1, got {}".format(repeat))
    if resume_from is not None:
        out_dir = resume_from
    elif out_dir is None:
        out_dir = default_out_dir(prefix="eq")
    os.makedirs(out_dir, exist_ok=True)

    cfg = SimulationConfig(
        potential=potential,
        nl_radius=nl_radius,
        nl_skin=nl_skin,
        npt_ensemble=npt_ensemble,
        n_particles=n_particles,
        density=density,
        seed=seed,
        block_size=block_size,
        buffer_size=buffer_size,
        max_scale=max_scale,
        min_scale=min_scale,
        progress=progress,
    )

    def work(temp, pressure, dest, cfg_i, init_frame, deltas, start_step, resumed_from):
        return equilibrate_point(
            temp,
            pressure,
            dest,
            cfg_i,
            num_eq_steps=num_eq_steps,
            start_step=start_step,
            init_frame=init_frame,
            deltas=deltas,
            resumed_from=resumed_from,
        )

    def pack(rep_results, base, temp, pressure):
        return rep_results[0] if repeat == 1 else {"replicas": rep_results}

    results = _drive_grid(
        work,
        pack,
        temps,
        pressures,
        out_dir,
        cfg,
        repeat,
        points,
        resume_from,
        label="equilibrate",
    )
    print("equilibration grid written to", out_dir)
    print(
        "inspect each T*_P*/volume_trace.png; then run production with "
        "resume_from={!r}".format(out_dir)
    )
    return results


def run_grid(
    temps=DEFAULT_TEMPS,
    pressures=DEFAULT_PRESSURES,
    out_dir=None,
    n_steps=10_000,
    block_size=250,
    num_eq_steps=20_000,
    buffer_size=4,
    potential=None,
    nl_radius=15,
    nl_skin=1.0,
    npt_ensemble=True,
    n_particles=210,
    density=0.3,
    seed=None,
    repeat=1,
    resume_from=None,
    points=None,
    recompute_energy=True,
    rdf_r_max=15.0,
    rdf_bins=100,
    progress=True,
):
    """Run every (T, P) combination on ``temps`` x ``pressures`` and report each.

    Replaces the old ``run_multi_temp_trial``: it sweeps the Cartesian product of
    temperatures and pressures, writing one artifact set per point under
    ``<out_dir>/T{temp}_P{pressure}/``.

    ``repeat`` runs that many independent replicas **with different initial
    configurations** at each (T, P) to quantify run-to-run uncertainty. With
    ``repeat == 1`` (default) the artifact set lands directly in the point dir.
    With ``repeat > 1`` each replica gets its own ``rep{i:02d}/`` subdir and a
    cross-replica ``summary.json``/``summary.md`` (mean/std/sem of every scalar
    observable) is written to the point dir.

    ``resume_from`` is the ``out_dir`` of a prior :func:`equilibrate_grid`: each
    replica starts production from its final equilibrated frame (and its tuned
    move widths), running production-only (``num_eq_steps`` is ignored). With
    ``repeat > 1`` replica ``i``'s production resumes from that grid's matching
    ``rep{i:02d}/equilibration.db`` (so pass the same ``repeat``); replicas whose
    ``equilibration.db`` is missing are skipped. ``points`` is an explicit iterable
    of ``(temp, pressure)`` pairs that overrides the ``temps`` x ``pressures``
    product, to run (or resume) only a chosen subset of the grid.

    Per-point/per-replica failures are logged and skipped so one bad run does not
    abort the rest of the grid. Returns ``{(temp, pressure): value}`` for the
    points that produced at least one replica -- ``value`` is the single
    artifacts dict when ``repeat == 1``, else
    ``{'replicas': [...], 'summary': {...}}``.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1, got {}".format(repeat))
    if out_dir is None:
        out_dir = default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    cfg = SimulationConfig(
        potential=potential,
        nl_radius=nl_radius,
        nl_skin=nl_skin,
        npt_ensemble=npt_ensemble,
        n_particles=n_particles,
        density=density,
        seed=seed,
        block_size=block_size,
        buffer_size=buffer_size,
        progress=progress,
        recompute_energy=recompute_energy,
        rdf_r_max=rdf_r_max,
        rdf_bins=rdf_bins,
    )

    def work(temp, pressure, dest, cfg_i, init_frame, deltas, start_step, resumed_from):
        # a resume means the frame is already equilibrated -> production only
        eff_eq_steps = None if resumed_from is not None else num_eq_steps
        return run_point(
            temp,
            pressure,
            dest,
            cfg_i,
            n_steps=n_steps,
            num_eq_steps=eff_eq_steps,
            init_frame=init_frame,
            deltas=deltas,
            resumed_from=resumed_from,
        )

    def pack(rep_results, base, temp, pressure):
        if repeat == 1:
            return rep_results[0]
        summary = write_summary(base, temp, pressure, rep_results)
        return {"replicas": rep_results, "summary": summary}

    return _drive_grid(
        work,
        pack,
        temps,
        pressures,
        out_dir,
        cfg,
        repeat,
        points,
        resume_from,
        label="state point",
    )


def _grid_args(p):
    """Add the flags every subcommand shares (grid selection + sampler knobs)."""
    p.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPS),
        help="temperatures in K (space separated; default: %(default)s)",
    )
    p.add_argument(
        "--pressures",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRESSURES),
        help="pressures in eV/A^3 (space separated; default: %(default)s)",
    )
    p.add_argument(
        "--points",
        nargs="+",
        default=None,
        metavar="T,P",
        help="explicit 'T,P' state points (overrides --temps/--pressures), e.g. to "
        "act on only a subset of the grid",
    )
    p.add_argument(
        "--ensemble",
        choices=["npt", "nvt"],
        default="npt",
        help="ensemble; nvt holds the box fixed (default: %(default)s)",
    )
    p.add_argument(
        "--block-size", type=int, default=250, help="steps per recorded block"
    )
    p.add_argument(
        "--num-eq-steps",
        type=int,
        default=20_000,
        help="equilibration steps (for continue-eq: additional steps to append)",
    )
    p.add_argument("--buffer-size", type=int, default=4, help="db write buffer size")
    p.add_argument("--n-particles", type=int, default=210, help="number of particles")
    p.add_argument("--density", type=float, default=0.3, help="initial reduced density")
    p.add_argument("--seed", type=int, default=None, help="config RNG seed")
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="replicas per (T, P) with different initial configs (default: %(default)s)",
    )
    p.add_argument(
        "--nl-radius", type=float, default=15, help="neighbour-list radius (A)"
    )
    p.add_argument("--nl-skin", type=float, default=1.0, help="neighbour-list skin (A)")
    p.add_argument(
        "--potential",
        default=None,
        help="params.json to load via GBQPotential.from_json (default: DEFAULT_POTENTIAL)",
    )
    p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="silence the tqdm progress bars",
    )
    return p


def build_parser():
    """Argparse front-end with one subcommand per action.

    Three verbs replace the old overloaded ``--equilibrate-only``/``--resume-from``
    pair so each invocation does exactly one thing:

    - ``equilibrate`` -- equilibrate a fresh grid + write convergence trace plots
    - ``continue-eq`` -- continue an existing equilibration in place for more steps
    - ``produce`` -- run production, optionally starting from an equilibrated grid
    """
    p = argparse.ArgumentParser(
        prog="python -m asmcmc.simulation.run",
        description="Run MC simulations over a (T, P) grid and write artifacts + plots.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    eq = sub.add_parser(
        "equilibrate",
        help="equilibrate a fresh (T, P) grid + write convergence trace plots",
    )
    _grid_args(eq)
    eq.add_argument(
        "--out-dir",
        default=None,
        help="output root (default: results/simulations/eq_<datetime>)",
    )

    ce = sub.add_parser(
        "continue-eq",
        help="continue an existing equilibration grid in place for more steps",
    )
    _grid_args(ce)
    ce.add_argument(
        "--from",
        dest="resume_from",
        required=True,
        help="prior equilibration grid dir to continue (appends to its dbs)",
    )

    pr = sub.add_parser(
        "produce", help="run production, optionally from an equilibrated grid"
    )
    _grid_args(pr)
    pr.add_argument(
        "--out-dir",
        default=None,
        help="output root (default: results/simulations/npt_<datetime>)",
    )
    pr.add_argument(
        "--from",
        dest="resume_from",
        default=None,
        help="prior equilibration grid dir to start production from (production-only)",
    )
    pr.add_argument("--n-steps", type=int, default=10_000, help="production steps")
    pr.add_argument("--rdf-r-max", type=float, default=15.0, help="RDF/OCF max r (A)")
    pr.add_argument("--rdf-bins", type=int, default=100, help="RDF/OCF bin count")
    pr.add_argument(
        "--no-recompute",
        dest="recompute_energy",
        action="store_false",
        help="use the tracked energy instead of recomputing per frame",
    )
    return p


def cli(argv=None):
    """Parse ``argv`` (default ``sys.argv``) and dispatch to the chosen subcommand.

    ``equilibrate`` -> :func:`equilibrate_grid` (fresh); ``continue-eq`` ->
    :func:`equilibrate_grid` with ``resume_from`` (continues in place);
    ``produce`` -> :func:`run_grid`, with ``--from`` starting production from a
    prior equilibration grid.
    """
    args = build_parser().parse_args(argv)
    potential = (
        GBQPotential.from_json(args.potential) if args.potential is not None else None
    )
    # the knobs every subcommand shares -- assembled once
    common = dict(
        temps=args.temps,
        pressures=args.pressures,
        points=_parse_points(args.points),
        num_eq_steps=args.num_eq_steps,
        block_size=args.block_size,
        buffer_size=args.buffer_size,
        potential=potential,
        nl_radius=args.nl_radius,
        nl_skin=args.nl_skin,
        npt_ensemble=(args.ensemble == "npt"),
        n_particles=args.n_particles,
        density=args.density,
        seed=args.seed,
        repeat=args.repeat,
        progress=args.progress,
    )
    if args.command == "equilibrate":
        return equilibrate_grid(out_dir=args.out_dir, **common)
    if args.command == "continue-eq":
        return equilibrate_grid(resume_from=args.resume_from, **common)
    return run_grid(
        out_dir=args.out_dir,
        resume_from=args.resume_from,
        n_steps=args.n_steps,
        recompute_energy=args.recompute_energy,
        rdf_r_max=args.rdf_r_max,
        rdf_bins=args.rdf_bins,
        **common,
    )


if __name__ == "__main__":
    cli()
