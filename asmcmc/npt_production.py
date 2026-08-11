"""
npt_production.py — Run production NPT trajectories on already-equilibrated
(temperature, pressure) points.

This module is the *production* counterpart to ``npt_equilibration``: that module
equilibrates a (T, P) grid into resumable run dirs; this one *consumes* those run
dirs. Each ``out_dir/T{temp}_P{pressure}/{seed}/`` holding an ``equilibration.db``
+ ``run_config.json`` is reloaded via ``MetropolisCalculator.from_equilibration``
(which restores the equilibrated frame and the *tuned* move widths) and a
production trajectory is recorded to ``simulation.db`` alongside it::

    out_dir/T{temp}_P{pressure}/{seed}/
        equilibration.db                # from npt_equilibration
        run_config.json                 # write-once static run definition
        simulation.db                   # <- written here (production)
        production_diagnostics.png      # energy & volume vs step

Production runs with **frozen** move widths (``dynamic_delta=False``): the widths
inherited from the end of equilibration are held fixed so the sampled ensemble is
not perturbed by ongoing tuning. Recording cadence (``block_size``) defaults to
one frame per sweep (``= n_particles``), finer than equilibration's tuning block,
so the recorded series resolves the autocorrelation needed for ESS/error bars.

Each ``{seed}`` subdir is an independent replica (its independence was injected at
the initial condition, before equilibration — see ``npt_equilibration``), so the
spread across a point's replicas is a meaningful error / ergodicity estimate.
Reducing replicas to observables + error bars is ``asmcmc.replica_stats`` — kept
separate so a notebook (or this module's own ``--aggregate`` CLI path) can pull it
in without every production worker also importing pandas/matplotlib.

Run:  python -m asmcmc.npt_production --num-steps N [--out-dir DIR]
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import random

# Cap each worker's BLAS/threadpool to one thread BEFORE numpy loads: production
# is parallelised across processes, so letting each also spin up N math threads
# would oversubscribe the cores and fight the parallelisation.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from tqdm.auto import tqdm

from asmcmc.metropolis import MetropolisCalculator
from asmcmc.npt_equilibration import find_point_dirs, plot_point_results

# Offset the production RNG stream off the equilibration one. Equilibration reseeds
# the global streams with the point's seed; reusing that exact seed for production
# would replay the same random sequence. Offsetting keeps production reproducible
# (per seed subdir, independent of worker order) while decorrelating its draws from
# the equilibration run's.
PRODUCTION_SEED_OFFSET = 2**20


def produce_point(
    output_dir,
    num_steps,
    block_size=None,
    buffer_size=100,
):
    """Run a production trajectory on one equilibrated point dir and return its path.

    Rebuilds the sampler from the point's ``run_config.json`` + last
    ``equilibration.db`` frame via ``from_equilibration`` (restoring the tuned,
    now-frozen move widths and pointing ``output_dir`` back at the same dir), then
    runs ``calculate_trajectory(..., num_eq_steps=None)`` — production only,
    ``dynamic_delta=False`` — which writes ``simulation.db`` into ``output_dir``.

    ``block_size`` defaults to the particle count (one recorded frame per sweep),
    finer than equilibration's tuning block so the series resolves autocorrelation.

    Reseeds the global RNG from the point's seed subdir (offset by
    ``PRODUCTION_SEED_OFFSET``) so the run is reproducible and independent of how
    many other points a worker produced first — the same parallel-safety property
    ``npt_equilibration`` relies on.

    ``simulation.db`` is written by appending, so a point that already has one is
    skipped (returning its path) rather than re-run, making the production pass
    re-runnable to finish an interrupted batch.
    """
    if os.path.exists(os.path.join(output_dir, "simulation.db")):
        return output_dir

    metro = MetropolisCalculator.from_equilibration(output_dir)

    seed_name = os.path.basename(os.path.normpath(output_dir))
    seed = int(seed_name) if seed_name.isdigit() else abs(hash(output_dir))
    prod_seed = seed + PRODUCTION_SEED_OFFSET
    random.seed(prod_seed)
    np.random.seed(prod_seed % (2**32))

    if block_size is None:
        block_size = len(metro.current_frame)
    metro.calculate_trajectory(
        num_steps=num_steps,
        block_size=block_size,
        num_eq_steps=None,  # production only: do not re-equilibrate
        buffer_size=buffer_size,
        progress=False,
    )
    return output_dir


def _produce_point(output_dir, cfg):
    """Run production on a single point in a worker process (picklable, module-level)."""
    return produce_point(
        output_dir,
        cfg["num_steps"],
        block_size=cfg["block_size"],
        buffer_size=cfg["buffer_size"],
    )


def produce_points(
    output_dirs,
    num_steps,
    block_size=None,
    buffer_size=100,
    max_workers=8,
):
    """Run production on a set of equilibrated point dirs in parallel, then render
    each one's diagnostics.

    Same pool machinery as the equilibration scan (spawn context, capped workers);
    a point that raises is logged and skipped. Returns the dirs successfully
    produced.
    """
    cfg = {
        "num_steps": num_steps,
        "block_size": block_size,
        "buffer_size": buffer_size,
    }
    num_workers = min(max_workers, os.cpu_count() or 1, len(output_dirs))

    done, failures = [], []
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(_produce_point, d, cfg): d for d in output_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Producing"):
            d = futures[fut]
            try:
                done.append(fut.result())
            except Exception as exc:
                failures.append(d)
                print(f"\n  point {d} failed: {exc!r}")

    for d in sorted(done):
        try:
            # energy/volume vs step from the production db, to its own PNG so the
            # equilibration diagnostics are left intact.
            plot_point_results(
                d,
                db_name="simulation.db",
                png_name="production_diagnostics.png",
            )
        except Exception as exc:
            print(f"  diagnostics for {d} failed: {exc!r}")

    print(f"\nProduced {len(done)}/{len(output_dirs)} point(s)")
    if failures:
        print(f"{len(failures)} point(s) failed: {sorted(failures)}")
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate already-produced points under --out-dir into a CSV + plots "
        "(instead of running production).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        metavar="N",
        help="Number of production steps to record per point (required unless --aggregate).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/npt_scan",
        help="Scan directory to read equilibrated points from (default: results/npt_scan).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        metavar="B",
        help="Recording cadence in steps (default: one frame per sweep, = n_particles).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Max concurrent worker processes (default: 12).",
    )
    args = parser.parse_args()

    if args.aggregate:
        # Local import: keeps pandas/matplotlib out of every production worker's
        # process image (they're only needed for this CLI path).
        from asmcmc.replica_stats import aggregate

        aggregate(args.out_dir)
    else:
        if args.num_steps is None:
            parser.error("--num-steps is required unless --aggregate is given")
        dirs = find_point_dirs(args.out_dir)
        if not dirs:
            parser.error(f"no equilibrated point dirs found under {args.out_dir}")
        print(
            f"Producing {len(dirs)} point(s) under {args.out_dir} for {args.num_steps} steps"
        )
        produce_points(
            dirs,
            args.num_steps,
            block_size=args.block_size,
            max_workers=args.max_workers,
        )
