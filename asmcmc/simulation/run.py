"""Runner / CLI for MC simulations: build a sampler over a (T, P) grid and write
a self-describing artifact set per state point.

Ties together :class:`asmcmc.metropolis.MetropolisCalculator` (the sampler),
:mod:`asmcmc.simulation.report` (config + observables + markdown) and
:mod:`asmcmc.simulation.plots` (diagnostic PNGs), the sampler analogue of
:mod:`asmcmc.fitting.run`. Run with ``python -m asmcmc.simulation.run``.
"""

import argparse
import datetime
import json
import os
import traceback

import numpy as np
from ase.db import connect

from asmcmc.metropolis import MetropolisCalculator
from asmcmc.initialize import generate_random_config
from asmcmc.potentials import GBQPotential
from asmcmc.simulation.report import run_config, write_artifacts, write_summary
from asmcmc.simulation.plots import write_plots, write_trace_plots

DEFAULT_TEMPS = (100.0, 200.0, 300.0, 400.0)
DEFAULT_PRESSURES = (5e-6, 1e-5, 1.5e-5, 2e-5)

# 1 atm = 6.32e-6 eV / Å^3


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


def _restart_state(db_path):
    """Final equilibrated frame + tuned trial-move deltas from an equilibration db.

    Reads the last block written to ``db_path`` (chronological insertion order)
    and rebuilds an ``ase.Atoms`` carrying the per-particle ``c_q``/``or_vec``
    arrays (stored in the row ``data``, since ase.db does not round-trip custom
    arrays) plus the adapted ``pos/or/vol`` move widths (from the row's scalar
    data). Used by :func:`run_grid` to resume production from a prior
    :func:`equilibrate_grid` without repeating equilibration or losing its tuning.
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
    return atoms, deltas


def run_point(
    temp,
    pressure,
    dest_dir,
    n_steps,
    block_size,
    num_eq_steps,
    buffer_size,
    potential,
    nl_radius,
    nl_skin,
    npt_ensemble,
    n_particles,
    density,
    seed,
    recompute_energy,
    rdf_r_max,
    rdf_bins,
    progress,
    init_frame=None,
    deltas=None,
    resumed_from=None,
):
    """Run one (T, P) simulation into ``dest_dir`` and write its artifact set.

    Builds a fresh config (reproducible for a fixed ``seed``, independent random
    init when ``seed is None``), runs ``calculate_trajectory``, then hands the
    finished ``simulation.db`` to
    :func:`asmcmc.simulation.report.write_artifacts` and
    :func:`asmcmc.simulation.plots.write_plots`. Returns the in-memory
    ``{'config', 'observables'}`` from the report.

    When ``init_frame`` is supplied (a restart from a prior equilibration) it is
    used instead of generating a config, ``deltas`` seeds the trial-move widths
    with the equilibration's tuned values, and the caller should pass
    ``num_eq_steps=None`` for a production-only run; ``resumed_from`` records the
    source db path in the provenance.
    """
    if init_frame is None:
        init_frame = generate_random_config(
            n_particles=n_particles, density=density, seed=seed
        )
    delta_kwargs = {}
    if deltas is not None:
        for key, kwarg in (("pos", "pos_delt"), ("or", "or_delt"), ("vol", "vol_delt")):
            if deltas.get(key) is not None:
                delta_kwargs[kwarg] = float(deltas[key])
    metro = MetropolisCalculator(
        temp,
        pressure,
        init_frame=init_frame,
        potential=potential,
        nl_radius=nl_radius,
        nl_skin=nl_skin,
        output_dir=dest_dir,
        npt_ensemble=npt_ensemble,
        **delta_kwargs,
    )
    # capture the trial-move widths as initially configured (block_update adapts
    # them in place during equilibration, so record before running).
    initial_deltas = {
        "pos": float(metro.pos_delt),
        "or": float(metro.or_delt),
        "vol": float(metro.vol_delt),
    }
    metro.calculate_trajectory(
        n_steps,
        block_size=block_size,
        num_eq_steps=num_eq_steps,
        buffer_size=buffer_size,
        progress=progress,
    )

    db_path = os.path.join(dest_dir, "simulation.db")
    meta = {
        "n_steps": n_steps,
        "block_size": block_size,
        "num_eq_steps": num_eq_steps,
        "seed": seed,
        "density_init": density,
        "initial_deltas": initial_deltas,
        "recompute_energy": recompute_energy,
        "rdf_r_max": rdf_r_max,
        "rdf_bins": rdf_bins,
        "resumed_from": resumed_from,
    }
    artifacts = write_artifacts(dest_dir, metro, db_path, meta=meta, progress=progress)
    write_plots(dest_dir, db_path, artifacts["observables"])
    return artifacts


def equilibrate_point(
    temp,
    pressure,
    dest_dir,
    num_eq_steps,
    block_size,
    buffer_size,
    potential,
    nl_radius,
    nl_skin,
    npt_ensemble,
    n_particles,
    density,
    seed,
    progress,
    max_scale=1.1,
    min_scale=0.9,
):
    """Equilibrate one (T, P) point into ``dest_dir`` and write trace diagnostics.

    Builds a fresh config, runs :meth:`MetropolisCalculator.equilibrate` into
    ``dest_dir/equilibration.db``, writes the trajectory trace PNGs (including
    ``volume_trace.png`` for the cell-volume plateau check) and an
    ``equilibration_config.json`` provenance file. Returns
    ``{'config', 'plots', 'db_path'}``.
    """
    init_frame = generate_random_config(
        n_particles=n_particles, density=density, seed=seed
    )
    metro = MetropolisCalculator(
        temp,
        pressure,
        init_frame=init_frame,
        potential=potential,
        nl_radius=nl_radius,
        nl_skin=nl_skin,
        output_dir=dest_dir,
        npt_ensemble=npt_ensemble,
    )
    metro.equilibrate(
        num_eq_steps,
        block_size=block_size,
        buffer_size=buffer_size,
        max_scale=max_scale,
        min_scale=min_scale,
        progress=progress,
    )

    db_path = os.path.join(dest_dir, "equilibration.db")
    # equilibrate() does not clear the decision lists, so run_config's acceptance
    # rates reflect the equilibration moves -- exactly what we want here.
    config = run_config(
        metro,
        db_path,
        meta={
            "num_eq_steps": num_eq_steps,
            "block_size": block_size,
            "seed": seed,
            "density_init": density,
        },
    )
    with open(os.path.join(dest_dir, "equilibration_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    plots = write_trace_plots(dest_dir, db_path)
    return {"config": config, "plots": plots, "db_path": db_path}


def equilibrate_grid(
    temps=DEFAULT_TEMPS,
    pressures=DEFAULT_PRESSURES,
    out_dir=None,
    num_eq_steps=20_000,
    block_size=250,
    buffer_size=5,
    potential=None,
    nl_radius=15,
    nl_skin=2.0,
    npt_ensemble=True,
    n_particles=210,
    density=0.3,
    seed=None,
    progress=True,
    max_scale=1.1,
    min_scale=0.9,
):
    """Equilibrate every (T, P) point and write per-point convergence diagnostics.

    The pre-production half of the workflow: for each (T, P) it equilibrates a
    fresh config into ``<out_dir>/T{temp}_P{pressure}/equilibration.db`` and
    renders the trajectory trace PNGs so the cell-volume plateau (and energy /
    acceptance / nematic order) can be checked per point before committing to
    production. Pass ``out_dir`` back to :func:`run_grid` as ``resume_from=`` to
    start production from these equilibrated configs without repeating the
    equilibration or losing its tuned move widths. Per-point failures are logged
    and skipped. Returns ``{(temp, pressure): {'config', 'plots', 'db_path'}}``.
    """
    if out_dir is None:
        out_dir = default_out_dir(prefix="eq")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for temp in temps:
        for pressure in pressures:
            print("=== equilibrate T={:g} K, P={:g} eV/A^3 ===".format(temp, pressure))
            dest = point_dir(out_dir, temp, pressure)
            try:
                results[(temp, pressure)] = equilibrate_point(
                    temp,
                    pressure,
                    dest,
                    num_eq_steps=num_eq_steps,
                    block_size=block_size,
                    buffer_size=buffer_size,
                    potential=potential,
                    nl_radius=nl_radius,
                    nl_skin=nl_skin,
                    npt_ensemble=npt_ensemble,
                    n_particles=n_particles,
                    density=density,
                    seed=seed,
                    progress=progress,
                    max_scale=max_scale,
                    min_scale=min_scale,
                )
            except Exception:
                print(
                    "!!! T={:g}, P={:g} equilibration failed; skipping:".format(
                        temp, pressure
                    )
                )
                traceback.print_exc()

    print("equilibration grid written to", out_dir)
    print(
        "inspect each T*_P*/volume_trace.png; then run production with "
        "resume_from={!r}".format(out_dir)
    )
    return results


def _replica_seed(seed, replica):
    """Per-replica config seed: distinct-but-reproducible, or ``None`` for random.

    A fixed ``seed`` is offset by the replica index so each replica gets a
    different initial configuration that still reproduces; ``seed=None`` leaves
    every replica genuinely random (independent draws).
    """
    return None if seed is None else seed + replica


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
    point starts production from that point's final equilibrated frame (and its
    tuned move widths), running production-only (``num_eq_steps`` is ignored).
    Points missing an ``equilibration.db`` under ``resume_from`` are skipped.
    ``resume_from`` is incompatible with ``repeat > 1`` (one equilibrated config
    per point).

    Per-point/per-replica failures are logged and skipped so one bad run does not
    abort the rest of the grid. Returns ``{(temp, pressure): value}`` for the
    points that produced at least one replica -- ``value`` is the single
    artifacts dict when ``repeat == 1``, else
    ``{'replicas': [...], 'summary': {...}}``.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1, got {}".format(repeat))
    if resume_from is not None and repeat > 1:
        raise ValueError("resume_from is incompatible with repeat>1")
    if out_dir is None:
        out_dir = default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for temp in temps:
        for pressure in pressures:
            print("=== state point T={:g} K, P={:g} eV/A^3 ===".format(temp, pressure))
            base = point_dir(out_dir, temp, pressure)

            init_frame, deltas, resumed_from = None, None, None
            eff_eq_steps = num_eq_steps
            if resume_from is not None:
                resumed_from = os.path.join(
                    point_dir(resume_from, temp, pressure), "equilibration.db"
                )
                try:
                    init_frame, deltas = _restart_state(resumed_from)
                except Exception:
                    print(
                        "!!! T={:g}, P={:g}: cannot resume from {}; skipping:".format(
                            temp, pressure, resumed_from
                        )
                    )
                    traceback.print_exc()
                    continue
                eff_eq_steps = None  # production-only: already equilibrated

            rep_results = []
            for i in range(repeat):
                dest = (
                    base if repeat == 1 else os.path.join(base, "rep{:02d}".format(i))
                )
                if repeat > 1:
                    print("--- replica {}/{} ---".format(i + 1, repeat))
                try:
                    rep_results.append(
                        run_point(
                            temp,
                            pressure,
                            dest,
                            n_steps=n_steps,
                            block_size=block_size,
                            num_eq_steps=eff_eq_steps,
                            buffer_size=buffer_size,
                            potential=potential,
                            nl_radius=nl_radius,
                            nl_skin=nl_skin,
                            npt_ensemble=npt_ensemble,
                            n_particles=n_particles,
                            density=density,
                            seed=_replica_seed(seed, i),
                            recompute_energy=recompute_energy,
                            rdf_r_max=rdf_r_max,
                            rdf_bins=rdf_bins,
                            progress=progress,
                            init_frame=init_frame,
                            deltas=deltas,
                            resumed_from=resumed_from,
                        )
                    )
                except Exception:
                    print(
                        "!!! T={:g}, P={:g}, replica {} failed; skipping:".format(
                            temp, pressure, i
                        )
                    )
                    traceback.print_exc()

            if not rep_results:
                continue
            if repeat == 1:
                results[(temp, pressure)] = rep_results[0]
            else:
                summary = write_summary(base, temp, pressure, rep_results)
                results[(temp, pressure)] = {
                    "replicas": rep_results,
                    "summary": summary,
                }
    return results


def main(
    temps=DEFAULT_TEMPS,
    pressures=DEFAULT_PRESSURES,
    out_dir=None,
    ensemble="npt",
    **kwargs,
):
    """Entry point: run the default (or supplied) (T, P) grid.

    ``ensemble`` selects NPT (volume moves on) or NVT (box fixed); remaining
    kwargs forward to :func:`run_grid`.
    """
    return run_grid(
        temps=temps,
        pressures=pressures,
        out_dir=out_dir,
        npt_ensemble=(ensemble == "npt"),
        **kwargs,
    )


def build_parser():
    """Argparse front-end mapping CLI flags onto :func:`main`."""
    p = argparse.ArgumentParser(
        prog="python -m asmcmc.simulation.run",
        description="Run MC simulations over a (T, P) grid and write artifacts + plots.",
    )
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
        "--out-dir",
        default=None,
        help="output root (default: results/simulations/npt_<datetime>)",
    )
    p.add_argument(
        "--ensemble",
        choices=["npt", "nvt"],
        default="npt",
        help="ensemble; nvt holds the box fixed (default: %(default)s)",
    )
    p.add_argument("--n-steps", type=int, default=10_000, help="production steps")
    p.add_argument(
        "--block-size", type=int, default=250, help="steps per recorded block"
    )
    p.add_argument(
        "--num-eq-steps",
        type=int,
        default=20_000,
        help="equilibration steps (0 or omit-as-None for production-only)",
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
    p.add_argument("--rdf-r-max", type=float, default=15.0, help="RDF/OCF max r (A)")
    p.add_argument("--rdf-bins", type=int, default=100, help="RDF/OCF bin count")
    p.add_argument(
        "--no-recompute",
        dest="recompute_energy",
        action="store_false",
        help="use the tracked energy instead of recomputing per frame",
    )
    p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="silence the tqdm progress bars",
    )
    return p


def cli(argv=None):
    """Parse ``argv`` (default ``sys.argv``) and dispatch to :func:`main`."""
    args = build_parser().parse_args(argv)
    potential = (
        GBQPotential.from_json(args.potential) if args.potential is not None else None
    )
    return main(
        temps=args.temps,
        pressures=args.pressures,
        out_dir=args.out_dir,
        ensemble=args.ensemble,
        n_steps=args.n_steps,
        block_size=args.block_size,
        num_eq_steps=args.num_eq_steps,
        buffer_size=args.buffer_size,
        potential=potential,
        nl_radius=args.nl_radius,
        nl_skin=args.nl_skin,
        n_particles=args.n_particles,
        density=args.density,
        seed=args.seed,
        repeat=args.repeat,
        recompute_energy=args.recompute_energy,
        rdf_r_max=args.rdf_r_max,
        rdf_bins=args.rdf_bins,
        progress=args.progress,
    )


if __name__ == "__main__":
    cli()
