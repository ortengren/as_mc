"""
nvt_scan.py — Locate where THIS Gay-Berne + quadrupole model orders/melts.

Instead of assuming benzene's real Kelvin temperatures land in the right place,
we map the model's own phase behaviour in *reduced units*:

    reduced temperature   T*   = kT / eps0
    reduced density       rho* = N * sigma0^3 / V

At each (T*, rho*) grid point we run constant-volume (NVT) Monte Carlo and
measure three observables that flag a phase change:

    E*/N      = <E> / (N * eps0)              reduced energy per particle
    Cv/kB     = Var(E) / ((kB T)^2 * N)       excess heat capacity (peaks at a transition)
    S         = largest eigenvalue of the     nematic/discotic order parameter
                nematic Q-tensor              (0 = isotropic, 1 = perfectly aligned)

A boundary shows up as a step in E*/N, a peak in Cv, and/or a rise in S as T*
drops. Everything is dimensionless, so the result is independent of the eps0
calibration we were questioning.

Run:  ~/.local/share/mamba/envs/asmcmc/bin/python nvt_scan.py
Outputs: scan_results/nvt_scan.csv  and  scan_results/nvt_scan.png
"""

import os
import csv
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

# Cap each worker's BLAS/threadpool to one thread BEFORE numpy loads: the grid
# is parallelised across processes, so letting each also spin up N math threads
# would interfere with the parallelisation.
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
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from initialize import generate_random_config
from metropolis import MetropolisCalculator, BOLTZCONST
from potentials import GB_PARAMS
from measurements import TrajectoryAnalyzer, AverageEnergy, NematicOrderParameter

EPS0 = GB_PARAMS["eps0"]  # eV  — energy scale, sets T* = kT/eps0
SIGMA0 = GB_PARAMS["sigma0"]  # Å   — length scale, sets rho* = N sigma0^3 / V


def tstar_to_kelvin(t_star):
    """Reduced temperature T* -> the Kelvin value the sampler expects.

    T* = kT/eps0  =>  T[K] = T* * eps0 / kB.
    """
    return t_star * EPS0 / BOLTZCONST


def to_reduced(mean_e, var_e, t_star, n_particles):
    """Convert the mean/variance of the (eV) energy to reduced observables.

        E*/N  = <E> / (N eps0)
        Cv/kB = Var(E) / ((kB T)^2 N),   with kB T = T* eps0

    Simple arithmetic on quantities AverageEnergy already produces.
    """
    kT = t_star * EPS0  # = kB * T, in eV
    return {
        "E_star_per_N": mean_e / (n_particles * EPS0),
        "Cv_over_kB": var_e / (kT**2 * n_particles),
    }


def equilibration_steps(
    t_star, rho_star, base=10_000, t_ref=1.0, rho_ref=0.35, max_steps=60_000
):
    """Heuristic equilibration budget for one (T*, rho*) point.

    Cold, dense points need more equilibration than warm, dilute ones. Scale a
    baseline by 1/T* and linearly in density, then clamp to ``[base, max_steps]``:
    easy points sit at the ``base`` floor (no wasted effort) while the hardest
    corner is allowed up to ``max_steps``. ``t_ref``/``rho_ref`` set where each
    factor crosses unity. Just a stand-in for true convergence detection.
    """
    factor = (t_ref / t_star) * (rho_star / rho_ref)
    return int(np.clip(round(base * factor), base, max_steps))


def run_state_point(
    t_star,
    rho_star,
    n_particles,
    num_steps,
    num_eq_steps,
    block_size,
    seed,
    scratch_dir,
    buffer_size=100,
):
    """Run one (T*, rho*) NVT state point and return reduced observables.

    A new fixed-volume config is built at the target reduced density, then a
    single `calculate_trajectory` call equilibrates it and records a simulation
    trajectory to `simulation.db`. The stored frames are then reduced to E*/N,
    Cv/kB and the nematic order S via the measurement framework.

    Seeding ``seed`` makes the whole point reproducible and self-contained: the
    sampler and trial moves draw from the global ``random``/``np.random``
    streams, so reseeding both pins this point's trajectory regardless of how
    many other points ran before it. Without this, workers would inherit one
    shared stream and produce identical pseudorandom numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Build a fixed-volume starting config at the target reduced density.
    frame = generate_random_config(n_particles=n_particles, density=rho_star, seed=seed)

    metro = MetropolisCalculator(
        temp=tstar_to_kelvin(t_star),
        pressure=0.0,
        init_frame=frame,
        npt_ensemble=False,
        output_dir=scratch_dir,
    )

    # Equilibrate then sample production to the db.
    metro.calculate_trajectory(
        num_steps=num_steps,
        block_size=block_size,
        num_eq_steps=num_eq_steps,
        buffer_size=buffer_size,
        progress=False,
    )

    analyzer = TrajectoryAnalyzer(os.path.join(scratch_dir, "simulation.db"))
    analyzer.add_measurement(
        "energy",
        AverageEnergy(
            recompute=True,
            nl_radius=metro.nl_cutoffs[0],
            energy_func=metro.energy_func,
        ),
    )
    analyzer.add_measurement("nematic", NematicOrderParameter())
    results = analyzer.run_analysis(progress=False)

    mean_e, var_e = results["energy"]
    reduced = to_reduced(mean_e, var_e, t_star, n_particles)

    return {
        "T_star": t_star,
        "rho_star": rho_star,
        "E_star_per_N": reduced["E_star_per_N"],
        "Cv_over_kB": reduced["Cv_over_kB"],
        "S": results["nematic"]["S"],
        "pos_acc": (
            float(np.mean(metro.pos_decisions)) if metro.pos_decisions else np.nan
        ),
        "or_acc": float(np.mean(metro.or_decisions)) if metro.or_decisions else np.nan,
        "pos_delt": metro.pos_delt,  # tuned COM step size (Å)
        "or_delt": metro.or_delt,  # tuned rotation step size (rad)
    }


def plot_scan(rows, t_star_grid, rho_star_grid, out_dir):
    """Plot S, E*/N and Cv/kB versus T*, one curve per density."""

    def series(rho, key):
        pts = sorted((r["T_star"], r[key]) for r in rows if r["rho_star"] == rho)
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rho_star_grid)))
    panels = [
        ("S", "Nematic order  S"),
        ("E_star_per_N", "Reduced energy  E / (N·eps0)"),
        ("Cv_over_kB", "Heat capacity  Cv / kB"),
    ]
    for ax, (key, title) in zip(axs, panels):
        for color, rho in zip(colors, rho_star_grid):
            x, y = series(rho, key)
            ax.plot(x, y, marker="o", color=color, label=f"rho*={rho}")
        ax.set_xlabel("Reduced temperature  T*")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axs[0].legend(title="density")
    fig.tight_layout()
    png = os.path.join(out_dir, "nvt_scan.png")
    fig.savefig(png, dpi=150)
    print(f"Wrote {png}")


def _evaluate_point(k, t_star, rho_star, cfg):
    """Run a single grid point in a worker process.

    Defined at module level so it can be pickled and shipped to the process
    pool. ``cfg`` carries the settings shared by every point; ``k`` is the
    point's grid index, returned alongside the result so ``main`` can reassemble
    the rows in grid order even though points finish out of order.
    """
    num_eq_steps = equilibration_steps(
        t_star, rho_star, base=cfg["eq_base"], max_steps=cfg["eq_max"]
    )
    row = run_state_point(
        t_star,
        rho_star,
        cfg["n_particles"],
        num_steps=cfg["num_steps"],
        num_eq_steps=num_eq_steps,
        block_size=cfg["block_size"],
        seed=cfg["seed0"] + k,
        scratch_dir=os.path.join(cfg["scratch_root"], f"point_{k:03d}"),
        buffer_size=cfg["buffer_size"],
    )
    return k, row


def main(
    t_star_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.6),
    rho_star_grid=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
    n_particles=125,
    num_steps=8_000,
    eq_base=10_000,
    eq_max=60_000,
    seed0=12345,
    out_dir="scan_results",
):
    """Run the (T*, rho*) NVT scan and write nvt_scan.csv + nvt_scan.png.

    Defaults reproduce the standing production grid; pass smaller grids/budgets
    for a fast smoke test, or denser grids for a higher-resolution run. ``out_dir``
    redirects all outputs so a smoke run can't clobber a real scan.

    Grid notes: ``n_particles=125`` is 5^3 (fills the SC lattice exactly);
    rho_star kept < ~0.95 so the simple-cubic start is buildable. Step budgets are
    raw trial-move counts handed to calculate_trajectory (~n_particles moves touch
    every particle once). Equilibration is sized per point (see
    equilibration_steps): a floor (eq_base) for easy points, up to eq_max for the
    cold/dense corner. Production (num_steps) is fixed so every point contributes
    the same number of samples.
    """
    block_size = n_particles  # one frame per ~pass -> num_steps // block_size frames
    buffer_size = 100

    random.seed(seed0)
    np.random.seed(seed0)
    scratch_root = os.path.join(out_dir, "_scratch")
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(scratch_root, ignore_errors=True)  # fresh scratch for this run

    # density-outer / temperature-inner so each CSV block is one isochore
    grid = [(t, r) for r in rho_star_grid for t in t_star_grid]

    # Settings shared by every point, pickled and shipped to each worker. Each
    # point writes to its own scratch dir (point_{k:03d}, built in the worker);
    # ASE db.write appends, so a shared dir would pile every point into one db.
    cfg = {
        "n_particles": n_particles,
        "num_steps": num_steps,
        "block_size": block_size,
        "buffer_size": buffer_size,
        "eq_base": eq_base,
        "eq_max": eq_max,
        "seed0": seed0,
        "scratch_root": scratch_root,
    }

    # One worker per core, never more than there are points. Lower this if you
    # want to keep cores free for other work.
    num_workers = min(os.cpu_count() or 1, len(grid))

    # Hand every point to the pool at once; collect each as its worker finishes
    # (out of order) and stash it under its grid index k. A point that raises is
    # logged and skipped so one bad point can't sink the whole scan.
    rows_by_k = {}
    failures = []
    # spawn (not the Linux-default fork): forking a process that has already
    # started threads -- e.g. a numpy/BLAS pool -- can deadlock the child
    # (Python 3.12 warns about exactly this). spawn launches clean interpreters;
    # the __main__ guard at the bottom keeps them from re-running the scan on
    # import. Per-point reseeding means results don't depend on the method.
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(_evaluate_point, k, t_star, rho_star, cfg): k
            for k, (t_star, rho_star) in enumerate(grid)
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Scanning (T*, rho*)"
        ):
            k = futures[fut]
            try:
                _, row = fut.result()
                rows_by_k[k] = row
            except Exception as exc:  # report and continue with the rest
                failures.append(k)
                print(f"\n  point {k:03d} failed: {exc!r}")

    if failures:
        print(f"\n{len(failures)} point(s) failed: {sorted(failures)}")
    if not rows_by_k:
        print("All points failed; nothing to write.")
        return

    # Back into grid order so the CSV stays in tidy isochore blocks.
    rows = [rows_by_k[k] for k in sorted(rows_by_k)]

    csv_path = os.path.join(out_dir, "nvt_scan.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    plot_scan(rows, t_star_grid, rho_star_grid, out_dir)


if __name__ == "__main__":
    main()
