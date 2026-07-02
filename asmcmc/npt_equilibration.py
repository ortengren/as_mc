"""
npt_equilibration.py — Equilibrate a grid of (temperature, pressure) NPT state
points in parallel.

Works in *physical* units — temperature in Kelvin, pressure in eV/Å³ — and only
**equilibrates** each point (no production). The deliverable is one resumable run
directory per point::

    results/npt_scan/
        T{temp}_P{pressure}/
            {seed}/
                equilibration.db                # the equilibration trajectory
                run_config.json                 # write-once static run definition
                equilibration_diagnostics.png   # energy & volume vs step

The diagnostics plot is the point of the scan: it lets you eyeball whether each
trajectory's energy and volume have flattened out, i.e. whether the point has
equilibrated sufficiently. A point can later be continued or have production run
on it via ``MetropolisCalculator.from_equilibration({seed}-dir)``; the ``{seed}``
subdirectory leaves room for multiple independent replicas of the same point.

Because volume floats in NPT, density is *not* an input axis (as it is in the NVT
scan): every point starts from one config (a columnar, near-equilibrium ordered
start by default) and relaxes to whatever volume (T, P) dictates.

The serial, single-point building blocks these workers drive — ``equilibrate_point``,
``continue_point``, ``find_point_dirs``, ``plot_point_results``, ``point_dirname`` —
live in ``asmcmc.equilibration`` and are re-exported here for backward compatibility.

Run:  python -m asmcmc.npt_equilibration
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import random

# Cap each worker's BLAS/threadpool to one thread BEFORE numpy loads: the grid
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

from asmcmc.initialize import (
    RandomLatticeInitializer,
    ColumnarLatticeInitializer,
    DEFAULT_DENSITY,
    DEFAULT_COLUMNAR_DENSITY,
)

# Single-point primitives now live in asmcmc.equilibration; re-exported here so
# existing `from asmcmc.npt_equilibration import ...` callers keep working.
from asmcmc.equilibration import (
    point_dirname,
    equilibrate_point,
    find_point_dirs,
    continue_point,
    plot_point_results,
)

# Selectable initial-config packings. Columnar is the default: for these oblate
# particles it starts at near-equilibrium (ordered) density — the *fast*
# equilibration direction (melting an ordered start is barrier-free; freezing a
# disordered one is not). Random (dilute simple-cubic) is kept for
# transition-bracketing checks that approach a boundary from the disordered side.
_INITIALIZERS = {
    "columnar": ColumnarLatticeInitializer,
    "random": RandomLatticeInitializer,
}
_DEFAULT_DENSITY = {
    "columnar": DEFAULT_COLUMNAR_DENSITY,
    "random": DEFAULT_DENSITY,
}


def _continue_point(output_dir, extra_steps, cfg):
    """Continue a single point in a worker process (picklable, module-level)."""
    return continue_point(
        output_dir,
        extra_steps,
        block_size=cfg["block_size"],
        buffer_size=cfg["buffer_size"],
        dynamic_delta=cfg["dynamic_delta"],
        vol_delt=cfg["vol_delt"],
    )


def extend_points(
    output_dirs,
    extra_steps,
    block_size=None,
    buffer_size=100,
    dynamic_delta=True,
    vol_delt=None,
    max_workers=8,
):
    """Equilibrate a chosen set of already-run point dirs ``extra_steps`` further,
    in parallel, then re-render each point's convergence diagnostics.

    ``vol_delt`` (default ``None``) is forwarded to each point's ``continue_point``
    to optionally reset the carried volume move width before continuing.

    Same pool machinery as ``main`` (spawn context, capped workers); a point that
    raises is logged and skipped. Returns the dirs that were successfully extended.
    """
    cfg = {
        "block_size": block_size,
        "buffer_size": buffer_size,
        "dynamic_delta": dynamic_delta,
        "vol_delt": vol_delt,
    }
    num_workers = min(max_workers, os.cpu_count() or 1, len(output_dirs))

    done, failures = [], []
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(_continue_point, d, extra_steps, cfg): d for d in output_dirs
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Extending equilibration"
        ):
            d = futures[fut]
            try:
                done.append(fut.result())
            except Exception as exc:
                failures.append(d)
                print(f"\n  point {d} failed: {exc!r}")

    for d in sorted(done):
        try:
            plot_point_results(d)
        except Exception as exc:
            print(f"  diagnostics for {d} failed: {exc!r}")

    print(f"\nExtended {len(done)}/{len(output_dirs)} point(s) by {extra_steps} steps")
    if failures:
        print(f"{len(failures)} point(s) failed: {sorted(failures)}")
    return done


def _evaluate_point(k, temp, pressure, cfg, r=0):
    """Equilibrate a single (grid point, replica) in a worker process.

    Defined at module level so it can be pickled and shipped to the process pool.
    ``cfg`` carries the settings shared by every point; ``k`` is the point's grid
    index, returned alongside the path so ``main`` can track which points came
    back. ``r`` is the replica index (0 = the first/only replica).

    The seed — which fixes both the initial lattice and the MC stream, and names
    the ``{seed}`` subdir — is ``seed0 + r*stride + k`` with ``stride =
    cfg["replica_stride"]`` (set by ``main`` to ≥ grid size, so no two
    (point, replica) pairs collide). Replica ``r=0`` is ``seed0 + k`` regardless
    of stride — i.e. exactly the single-replica seed — so adding replicas to an
    existing scan reuses the original runs as replica 0 (see the idempotent skip
    below) and only computes the new ones.

    Idempotent: if the point directory already holds a finished run
    (``run_config.json`` present), it is skipped rather than re-run, so ``main``
    can be re-invoked to finish / re-plot an interrupted scan, or to add replicas,
    without redoing work.
    """
    seed = cfg["seed0"] + r * cfg.get("replica_stride", 0) + k
    output_dir = os.path.join(cfg["out_dir"], point_dirname(temp, pressure), str(seed))

    if os.path.exists(os.path.join(output_dir, "run_config.json")):
        return k, output_dir

    init_cls = _INITIALIZERS[cfg.get("packing", "columnar")]
    initializer = init_cls(
        n_particles=cfg["n_particles"], density=cfg["density"], seed=seed
    )
    equilibrate_point(
        temp,
        pressure,
        num_steps=cfg["num_steps"],
        initializer=initializer,
        block_size=cfg["block_size"],
        output_dir=output_dir,
        seed=seed,
        buffer_size=cfg["buffer_size"],
    )
    return k, output_dir


def _submission_order(grid):
    """Grid indices ordered most-expensive-first (cold, high-pressure first).

    NPT per-point cost increases with the equilibrated density. Dispatching the cold,
    high-pressure points first means that when the grid outnumbers the workers
    the long-pole points run from t=0 instead of tailing the batch. This
    reorders *submission* only: each point keeps its canonical grid index ``k``
    (and thus its seed and output dir), so the outputs are identical regardless
    of order. ``grid`` holds ``(temp, pressure)`` tuples.
    """
    return sorted(range(len(grid)), key=lambda k: (-grid[k][1], grid[k][0]))


def main(
    temp_grid=(10.0, 50.0, 100.0, 150.0, 200.0, 250.0),
    pressure_grid=(5e-7, 1e-6, 1.5e-6),
    n_particles=125,
    density=0.75,
    packing="random",
    num_steps=500_000,
    block_size=None,
    buffer_size=100,
    seed0=12345,
    replicas=2,
    out_dir="results/low_p_scan",
    max_workers=12,
):
    """Equilibrate the (T, P) grid in parallel, one resumable run dir per
    (point, replica).

    ``temp`` is in Kelvin and ``pressure`` in eV/Å³, handed straight to the
    sampler. The default grid (9 T × 3 P = 27 points) is shaped by the first
    production scan's phase diagram: 15 K spacing across the transition window
    (70–175 K) where both the melting and clearing boundaries live, a single cold
    crystal anchor at 40 K, and only three pressures (P* ≈ 0.45, 4.5, 13.5) — the
    near-duplicate low-P isobars are pruned to one anchor. Every point
    starts from one ``n_particles`` config built by ``packing`` (``"columnar"``,
    the default ordered near-equilibrium start, or ``"random"`` dilute simple
    cubic) at reduced density ``density`` and relaxes its volume. ``density=None``
    picks the packing-appropriate default (columnar ~1.4, random ~0.6).
    ``num_steps`` is the (re-entrant) equilibration target and ``block_size``
    defaults to ``n_particles`` (~one recorded frame per pass).

    ``replicas`` independent equilibrations are run per (T, P) point — each from
    its own seed (and so its own initial config + MC stream), written to its own
    ``{seed}`` subdir. These are *true* independent replicas (the divergence is
    injected at the initial condition, before equilibration), which is what makes
    a later between-replica spread a meaningful error / ergodicity estimate.
    Seeds are ``seed0 + r*stride + k`` (``stride = len(grid)``), so replica
    ``r=0`` is the single-replica seed ``seed0 + k``: **adding replicas to an
    existing scan reuses the original runs as replica 0** (the idempotent skip
    leaves finished dirs untouched) and only computes the new ones. Reproducing /
    extending a scan therefore requires keeping the same grid (so ``k`` and the
    stride are stable).

    Each (point, replica) writes a resumable run dir at
    ``out_dir/T{temp}_P{pressure}/{seed}/`` (equilibration.db + run_config.json),
    plus an equilibration_diagnostics.png to eyeball convergence. Those
    directories are real outputs and are *not* cleared between runs — re-run
    main() to pick up already-finished points and re-plot them.

    Points are dispatched cold/high-pressure first (the densest — and so, in
    NPT, the slowest — points) and the pool is capped at ``max_workers``
    (default 8): processes contend for memory bandwidth / CPU frequency, so throughput
    plateaus around 8 concurrent points regardless of core count.
    """
    if packing not in _INITIALIZERS:
        raise ValueError(
            f"unknown packing {packing!r}; choose from {sorted(_INITIALIZERS)}"
        )
    if density is None:
        density = _DEFAULT_DENSITY[packing]
    if block_size is None:
        block_size = n_particles
    if replicas < 1:
        raise ValueError(f"replicas must be >= 1, got {replicas}")

    random.seed(seed0)
    np.random.seed(seed0)
    os.makedirs(out_dir, exist_ok=True)

    # pressure-outer / temperature-inner so points group into isobars
    grid = [(t, p) for p in pressure_grid for t in temp_grid]

    # Settings shared by every point, pickled and shipped to each worker.
    # replica_stride >= grid size guarantees distinct seeds across all
    # (point, replica) pairs while keeping r=0 at seed0+k (the single-replica seed).
    cfg = {
        "n_particles": n_particles,
        "density": density,
        "packing": packing,
        "num_steps": num_steps,
        "block_size": block_size,
        "buffer_size": buffer_size,
        "seed0": seed0,
        "replica_stride": len(grid),
        "out_dir": out_dir,
    }

    # One worker per core, never more than there are work units, and capped at
    # max_workers: beyond ~8 concurrent numpy-heavy MC processes, added workers
    # mostly thrash (memory-bandwidth / CPU-frequency contention) rather than add
    # throughput, so a higher core count buys little here.
    num_workers = min(max_workers, os.cpu_count() or 1, len(grid) * replicas)

    # Hand every point to the pool at once; collect each as its worker finishes.
    # A point that raises is logged and skipped so one bad point can't sink the
    # whole scan.
    #
    # spawn (not the Linux-default fork): forking a process that has already
    # started threads -- e.g. a numpy/BLAS pool -- can deadlock the child. spawn
    # launches clean interpreters; the __main__ guard at the bottom keeps them
    # from re-running the scan on import. Per-point reseeding makes results
    # independent of the start method and completion order.
    done = {}
    failures = []
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    ) as pool:
        # Submit cold/high-pressure (densest, slowest) points first so the long
        # pole starts at t=0 instead of tailing the batch when work units >
        # workers; all replicas of a point share its cost, so order by point.
        futures = {}
        for k in _submission_order(grid):
            t, p = grid[k]
            for r in range(replicas):
                futures[pool.submit(_evaluate_point, k, t, p, cfg, r)] = (k, r)
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Equilibrating (T, P)"
        ):
            k, r = futures[fut]
            try:
                _, output_dir = fut.result()
                done[(k, r)] = output_dir
            except Exception as exc:  # report and continue with the rest
                failures.append((k, r))
                print(f"\n  point {k:03d} replica {r} failed: {exc!r}")

    # Per-(point, replica) convergence diagnostics for everything that finished.
    for key in sorted(done):
        try:
            plot_point_results(done[key])
        except Exception as exc:
            print(f"  diagnostics for {done[key]} failed: {exc!r}")

    total = len(grid) * replicas
    print(f"\nEquilibrated {len(done)}/{total} (point, replica) run(s) under {out_dir}")
    if failures:
        print(f"{len(failures)} run(s) failed: {sorted(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extend",
        type=int,
        metavar="N",
        help="Resume every finished point under --out-dir for N more equilibration "
        "steps (instead of running a fresh scan).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/npt_scan",
        help="Scan directory to read points from / write to (default: results/npt_scan).",
    )
    parser.add_argument(
        "--packing",
        default="columnar",
        choices=sorted(_INITIALIZERS),
        help="Initial-config packing for a fresh scan (default: columnar — ordered, "
        "near-equilibrium density; 'random' is the dilute simple-cubic start for "
        "bracketing a transition from the disordered side).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Reduced starting density rho* for a fresh scan "
        "(default: packing-appropriate — columnar ~1.4, random ~0.6).",
    )
    parser.add_argument(
        "--reset-vol-delt",
        type=float,
        default=None,
        metavar="DELTA",
        help="With --extend, reset each point's volume move width to DELTA before "
        "resuming (default: keep the tuned value). Use when an old run's vol_delt "
        "is far off and the tuner would take many windows to recover.",
    )
    args = parser.parse_args()

    if args.extend is not None:
        dirs = find_point_dirs(args.out_dir)
        if not dirs:
            parser.error(f"no resumable point dirs found under {args.out_dir}")
        print(
            f"Extending {len(dirs)} point(s) under {args.out_dir} by {args.extend} steps"
        )
        extend_points(dirs, args.extend, vol_delt=args.reset_vol_delt)
    else:
        main(out_dir=args.out_dir, packing=args.packing, density=args.density)
