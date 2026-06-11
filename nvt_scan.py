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

    Pure arithmetic on quantities AverageEnergy already produces, so the
    reduced-unit knowledge stays here in the scan rather than leaking into the
    (unit-agnostic) measurement framework.
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

    Relaxation slows toward the cold, dense corner -- trial moves shrink and are
    rarely accepted, and the box is more jammed -- so those points need more
    equilibration than warm, dilute ones. We scale a trusted baseline by 1/T*
    and linearly in density, then clamp to ``[base, max_steps]``: easy points
    sit at the ``base`` floor (no wasted effort) while the hardest corner is
    allowed up to ``max_steps``. ``t_ref``/``rho_ref`` set where each factor
    crosses unity. This is a cheap stand-in for true convergence detection;
    tune the constants once you see whether the cold points actually settle.
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

    A fresh fixed-volume config is built at the target reduced density, then a
    single `calculate_trajectory` call equilibrates it (with adaptive trial-move
    tuning) and records a production trajectory to `simulation.db`. The step
    counts are raw trial-move counts forwarded straight to the sampler — as a
    rule of thumb ~`n_particles` moves touch every particle once. The stored
    frames are then reduced to E*/N, Cv/kB and the nematic order S via the
    measurement framework.
    """
    # Build a fixed-volume starting config at the target reduced density.
    frame = generate_random_config(n_particles=n_particles, density=rho_star, seed=seed)

    metro = MetropolisCalculator(
        temp=tstar_to_kelvin(t_star),
        pressure=0.0,
        init_frame=frame,
        npt_ensemble=False,
        output_dir=scratch_dir,
    )

    # One call: equilibrate (adaptive delta) then sample production to the db.
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
        # Production-only acceptance: calculate_trajectory clears the decision
        # lists after equilibration, so these are the tuned-move accept rates.
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


def main():
    # ---- scan settings: coarse + fast first look; raise for production ----
    n_particles = 125  # 5^3, fills the SC lattice exactly
    t_star_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
    rho_star_grid = [
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
    ]  # kept < ~0.95 so the SC start is buildable

    # Step budgets handed straight to calculate_trajectory (raw trial-move
    # counts). For intuition, ~n_particles moves touch every particle once.
    # Equilibration is sized per point (see equilibration_steps): a fixed floor
    # for easy points, more for the cold/dense corner. Production is fixed so
    # every point contributes the same number of samples.
    eq_base = 10_000  # equilibration floor for easy (warm/dilute) points
    eq_max = 60_000  # cap for the hardest (cold/dense) corner
    num_steps = 5_000  # production
    block_size = n_particles  # one frame per ~pass -> num_steps // block_size frames
    buffer_size = 100
    seed0 = 12345

    random.seed(seed0)
    np.random.seed(seed0)

    out_dir = "scan_results"
    scratch_root = os.path.join(out_dir, "_scratch")
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(scratch_root, ignore_errors=True)  # fresh scratch for this run

    # density-outer / temperature-inner so each CSV block is one isochore
    grid = [(t, r) for r in rho_star_grid for t in t_star_grid]
    rows = []
    for k, (t_star, rho_star) in enumerate(tqdm(grid, desc="Scanning (T*, rho*)")):
        # Each point writes to its own scratch dir; ASE db.write appends, so a
        # shared dir would accumulate every point's frames into one db.
        point_dir = os.path.join(scratch_root, f"point_{k:03d}")
        num_eq_steps = equilibration_steps(
            t_star, rho_star, base=eq_base, max_steps=eq_max
        )
        rows.append(
            run_state_point(
                t_star,
                rho_star,
                n_particles,
                num_steps=num_steps,
                num_eq_steps=num_eq_steps,
                block_size=block_size,
                seed=seed0 + k,
                scratch_dir=point_dir,
                buffer_size=buffer_size,
            )
        )

    csv_path = os.path.join(out_dir, "nvt_scan.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    plot_scan(rows, t_star_grid, rho_star_grid, out_dir)


if __name__ == "__main__":
    main()
