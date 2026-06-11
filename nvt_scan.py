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
from potentials import calc_total_energy, GB_PARAMS

EPS0 = GB_PARAMS["eps0"]      # eV  — energy scale, sets T* = kT/eps0
SIGMA0 = GB_PARAMS["sigma0"]  # Å   — length scale, sets rho* = N sigma0^3 / V


def tstar_to_kelvin(t_star):
    """Reduced temperature T* -> the Kelvin value the sampler expects.

    T* = kT/eps0  =>  T[K] = T* * eps0 / kB.
    """
    return t_star * EPS0 / BOLTZCONST


def nematic_order_parameter(or_vecs):
    """Discotic/nematic order parameter S for a set of unit orientation axes.

    Builds the symmetric, traceless Q-tensor
        Q_ab = (1/N) sum_i (3 u_ia u_ib - delta_ab) / 2
    and returns its largest eigenvalue. S ~ 0 for random orientations
    (isotropic), S -> 1 when all axes align (orientationally ordered).
    """
    u = np.asarray(or_vecs)
    n = len(u)
    Q = (3.0 * np.einsum("ia,ib->ab", u, u) - n * np.eye(3)) / (2.0 * n)
    return np.linalg.eigvalsh(Q)[-1]  # eigvalsh returns ascending eigenvalues


def run_state_point(t_star, rho_star, n_particles, n_equil_sweeps,
                    n_prod_sweeps, sample_every, seed, scratch_dir,
                    tune_every_sweeps=5, tune_max_scale=2.0):
    """Equilibrate and sample one (T*, rho*) NVT state point.

    Returns a dict of reduced observables. One "sweep" = n_particles trial
    moves, so the box sees each particle ~once per sweep on average.
    """
    # 1. Build a fixed-volume starting config at the target reduced density.
    frame = generate_random_config(
        n_particles=n_particles, density=rho_star, seed=seed
    )

    # 2. NVT sampler: the same tested engine, with volume moves switched off so
    #    the box (and therefore rho*) is held fixed. pressure is unused in NVT.
    metro = MetropolisCalculator(
        temp=tstar_to_kelvin(t_star),
        pressure=0.0,
        init_frame=frame,
        volume_moves=False,
        output_dir=scratch_dir,
    )

    n_steps_prod = n_prod_sweeps * n_particles
    sample_stride = sample_every * n_particles

    # 3. Equilibrate with the sampler's own routine. With volume_moves=False
    #    this is NVT; dynamic_delta=True adapts the trial-move sizes toward
    #    ~27.5% acceptance (every tune_every_sweeps sweeps) so the chain
    #    decorrelates instead of crawling at ~98% acceptance. buffer_size is
    #    large so the scratch db is flushed only once per state point.
    metro.equilibrate(
        n_equil_sweeps * n_particles,
        block_size=tune_every_sweeps * n_particles,
        dynamic_delta=True,
        buffer_size=10_000,
        progress=False,
        max_scale=tune_max_scale,
        min_scale=1.0 / tune_max_scale,
    )

    # Reset the acceptance bookkeeping so the reported rates are production-only.
    metro.pos_decisions.clear()
    metro.or_decisions.clear()

    # 4. Production: keep sampling energy and orientational order. We recompute
    #    the full energy at each sample (cheap, and avoids any drift in the
    #    incremental tracker over a long run).
    energies, s_values = [], []
    for i in range(n_steps_prod):
        metro.step()
        if (i + 1) % sample_stride == 0:
            energies.append(
                calc_total_energy(metro.current_frame, metro.nl_cutoffs,
                                  metro.energy_func)
            )
            s_values.append(
                nematic_order_parameter(metro.current_frame.arrays["or_vec"])
            )

    energies = np.asarray(energies)
    kT = BOLTZCONST * tstar_to_kelvin(t_star)  # == T* * eps0, in eV

    return {
        "T_star": t_star,
        "rho_star": rho_star,
        "E_star_per_N": np.mean(energies) / (n_particles * EPS0),
        "Cv_over_kB": np.var(energies) / (kT ** 2 * n_particles),
        "S": float(np.mean(s_values)),
        "pos_acc": float(np.mean(metro.pos_decisions)) if metro.pos_decisions else np.nan,
        "or_acc": float(np.mean(metro.or_decisions)) if metro.or_decisions else np.nan,
        "pos_delt": metro.pos_delt,   # tuned COM step size (Å)
        "or_delt": metro.or_delt,     # tuned rotation step size (rad)
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
    n_particles = 125                                 # 5^3, fills the SC lattice exactly
    t_star_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
    rho_star_grid = [0.15, 0.25, 0.35, 0.45, 0.55]    # kept < ~0.95 so the SC start is buildable
    n_equil_sweeps = 30
    n_prod_sweeps = 40
    sample_every = 1                                  # sweeps between samples
    seed0 = 12345

    random.seed(seed0)
    np.random.seed(seed0)

    out_dir = "scan_results"
    scratch_dir = os.path.join(out_dir, "_scratch")
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(scratch_dir, ignore_errors=True)  # fresh equilibration scratch

    # iterate density-outer / temperature-inner so each CSV block is one isochore
    grid = [(t, r) for r in rho_star_grid for t in t_star_grid]
    rows = []
    for k, (t_star, rho_star) in enumerate(
        tqdm(grid, desc="Scanning (T*, rho*)")
    ):
        rows.append(
            run_state_point(
                t_star, rho_star, n_particles,
                n_equil_sweeps, n_prod_sweeps, sample_every,
                seed=seed0 + k, scratch_dir=scratch_dir,
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
