"""
npt_production.py — Run production NPT trajectories on already-equilibrated
(temperature, pressure) points and (eventually) aggregate observables across
replicas.

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
spread across a point's replicas is a meaningful error / ergodicity estimate. That
aggregation is the planned second half of this module.

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
import pandas as pd
from scipy.stats import t as student_t
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from asmcmc.metropolis import MetropolisCalculator
from asmcmc.config import RunConfig
from asmcmc.measurements import (
    TrajectoryAnalyzer,
    EffectiveSampleSize,
    nematic_q_tensor,
)
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


# ---------------------------------------------------------------------------
# Aggregation: reduce each (T, P) point's replicas to observables + error bars
# ---------------------------------------------------------------------------

# Per-frame scalar observables, as callables over a recorded frame's
# (frame, scalar_data, array_data). Reduced per replica via EffectiveSampleSize,
# which returns each one's mean + within-chain ESS/SEM; the replica means are then
# combined across replicas in `_combine_observable`.
OBSERVABLES = {
    "energy_per_N": lambda f, s, a: s["total_energy"] / s["num_particles"],
    "volume": lambda f, s, a: s["vol"],
    "density": lambda f, s, a: s["num_particles"] / s["vol"],
    "nematic_S": lambda f, s, a: float(
        np.linalg.eigvalsh(nematic_q_tensor(a["or_vec"]))[-1]
    ),
}


def replica_observables(seed_dir, db_name="simulation.db"):
    """Reduce one replica's production trajectory to per-observable statistics.

    Runs a ``TrajectoryAnalyzer`` over ``seed_dir/db_name`` with an
    ``EffectiveSampleSize`` for each observable in ``OBSERVABLES``; returns a dict
    ``{observable: finalize_dict}`` carrying the replica's mean, std, ESS, tau and
    within-chain SEM. (Production is run on already-equilibrated configs, so all
    recorded frames are used — no burn-in is dropped.)
    """
    analyzer = TrajectoryAnalyzer(os.path.join(seed_dir, db_name))
    for name, obs in OBSERVABLES.items():
        analyzer.add_measurement(name, EffectiveSampleSize(obs))
    return analyzer.run_analysis(progress=False)


def _combine_observable(replica_stats):
    """Combine one observable across a point's replicas into a single estimate.

    ``replica_stats`` is the list of per-replica ``EffectiveSampleSize.finalize()``
    dicts for one observable. With R>=2 independent replicas the reported error is
    the **between-replica** SEM — ``std(replica means, ddof=1)/sqrt(R)`` with a
    Student-t 95% interval (R-1 dof) — which needs no autocorrelation model and
    captures slow modes / basin differences. The within-chain ESS SEMs are used as
    a *cross-check*: ``consistency`` = between-replica variance / mean within-chain
    variance-of-the-mean should be ~1 if the replicas sample the same basin and
    grows >>1 under ergodicity breaking. With a single replica there is no spread,
    so it falls back to that replica's within-chain ESS SEM (a normal 95% interval).
    """
    means = np.array([r["mean"] for r in replica_stats], dtype=float)
    R = len(means)
    grand_mean = float(means.mean())
    pooled_ess = float(sum(r["ess"] for r in replica_stats))

    if R >= 2:
        s = float(means.std(ddof=1))
        sem = s / np.sqrt(R)
        ci95 = float(student_t.ppf(0.975, R - 1)) * sem
        within_var = float(np.mean([r["sem"] ** 2 for r in replica_stats]))
        consistency = (s**2 / within_var) if within_var > 0 else float("nan")
        method = "between_replica"
    else:
        sem = float(replica_stats[0]["sem"])
        ci95 = 1.96 * sem
        consistency = float("nan")
        method = "ess_single"

    return {
        "mean": grand_mean,
        "sem": sem,
        "ci95": ci95,
        "ess": pooled_ess,
        "method": method,
        "consistency": consistency,
    }


def _group_replicas_by_point(out_dir, db_name="simulation.db"):
    """Map each (T, P) point dir to its list of produced replica seed dirs.

    A replica counts only once it has a ``db_name`` (i.e. production has run). The
    point is the parent of the ``{seed}`` dir, so sibling seed dirs group together.
    """
    points = {}
    for d in find_point_dirs(out_dir):
        if os.path.exists(os.path.join(d, db_name)):
            point = os.path.dirname(os.path.normpath(d))
            points.setdefault(point, []).append(d)
    return {p: sorted(dirs) for p, dirs in points.items()}


def aggregate(
    out_dir, db_name="simulation.db", csv_name="npt_production.csv", plot=True
):
    """Reduce every produced (T, P) point under ``out_dir`` to observables with
    error bars and write a CSV (+ plots).

    Groups replicas by point, reduces each replica via ``replica_observables``,
    combines across replicas via ``_combine_observable`` (between-replica error,
    ESS cross-check), and reads each point's (T, P) from a replica's
    ``run_config.json``. One row per point; columns per observable are
    ``{obs}_mean/_sem/_ci95/_ess/_consistency/_method``. Returns the DataFrame.
    """
    points = _group_replicas_by_point(out_dir, db_name)
    if not points:
        raise FileNotFoundError(f"no produced points (with {db_name}) under {out_dir}")

    rows = []
    for point in sorted(points):
        seed_dirs = points[point]
        cfg = RunConfig.load(os.path.join(seed_dirs[0], "run_config.json"))
        per_replica = [replica_observables(d, db_name) for d in seed_dirs]

        row = {
            "point": os.path.basename(point),
            "temp": cfg.temp,
            "pressure": cfg.pressure,
            "n_replicas": len(seed_dirs),
        }
        for name in OBSERVABLES:
            c = _combine_observable([rep[name] for rep in per_replica])
            row[f"{name}_mean"] = c["mean"]
            row[f"{name}_sem"] = c["sem"]
            row[f"{name}_ci95"] = c["ci95"]
            row[f"{name}_ess"] = c["ess"]
            row[f"{name}_consistency"] = c["consistency"]
            row[f"{name}_method"] = c["method"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["pressure", "temp"]).reset_index(drop=True)
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} point(s))")
    if plot:
        plot_aggregate(df, out_dir)
    return df


def plot_aggregate(df, out_dir, png_name="npt_production.png"):
    """Observable vs temperature, one curve per pressure, with 95% error bars.

    Renders E/N, density and nematic S — the EOS + order-parameter view of the
    phase behaviour, now with the replica/ESS uncertainty made visible. Writes
    ``png_name`` into ``out_dir`` and returns its path.
    """
    panels = [
        ("energy_per_N", "E / N  (eV)"),
        ("density", "Density  N/V  (Å⁻³)"),
        ("nematic_S", "Nematic order  S"),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    pressures = sorted(df["pressure"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(pressures), 1)))

    for ax, (name, label) in zip(axs, panels):
        for p, c in zip(pressures, colors):
            sub = df[df["pressure"] == p].sort_values("temp")
            ax.errorbar(
                sub["temp"],
                sub[f"{name}_mean"],
                yerr=sub[f"{name}_ci95"],
                marker="o",
                capsize=3,
                color=c,
                label=f"P={p:g}",
            )
        ax.set_xlabel("Temperature  (K)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axs[-1].legend(title="P (eV/Å³)", fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, png_name)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


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
