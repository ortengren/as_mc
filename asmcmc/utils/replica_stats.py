"""
replica_stats.py — Reduce a produced (T, P) point's replicas to observables with
error bars.

Consumes the run dirs that ``scripts/npt_scan.py`` produces: each
``out_dir/T{temp}_P{pressure}/{seed}/`` holding a ``simulation.db`` is one
replica, and sibling ``{seed}`` dirs are independent replicas of the same point
(their independence was injected at the initial condition, before equilibration).
That makes the spread across a point's replicas a meaningful error / ergodicity
estimate rather than a restatement of within-chain autocorrelation.

This is the analysis layer, built on ``measurements.TrajectoryAnalyzer`` /
``EffectiveSampleSize`` — deliberately separate from the parallel *drivers* in
``scripts/npt_scan.py`` and from the per-run-dir primitives in
``asmcmc.utils.npt_production``, so a notebook can import it without pulling in a process
pool or an argparse entry point.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from asmcmc.base.config import RunConfig
from asmcmc.utils.measurements import (
    TrajectoryAnalyzer,
    EffectiveSampleSize,
    nematic_q_tensor,
)
from asmcmc.utils.equilibration import find_point_dirs

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
