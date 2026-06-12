import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random

import numpy as np
import pytest

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from asmcmc.nvt_scan import (
    tstar_to_kelvin,
    to_reduced,
    equilibration_steps,
    run_state_point,
    _evaluate_point,
    main,
    plot_heatmaps,
    load_rows,
    replot,
)
from asmcmc.metropolis import MetropolisCalculator, BOLTZCONST
from asmcmc.potentials import GB_PARAMS
from asmcmc.initialize import generate_random_config

# ---------------------------------------------------------------------------
# Reduced-unit conversion
# ---------------------------------------------------------------------------


def test_tstar_to_kelvin_roundtrip():
    eps0 = GB_PARAMS["eps0"]
    # T* = 1  <=>  kT = eps0
    assert np.isclose(tstar_to_kelvin(1.0), eps0 / BOLTZCONST)
    # kB * T[K] must recover T* * eps0
    for t_star in (0.2, 0.5, 1.3):
        assert np.isclose(BOLTZCONST * tstar_to_kelvin(t_star), t_star * eps0)


# ---------------------------------------------------------------------------
# Reduced-unit observables
# ---------------------------------------------------------------------------


def test_to_reduced_matches_definitions():
    """E*/N and Cv/kB are the textbook reductions of <E> and Var(E)."""
    eps0 = GB_PARAMS["eps0"]
    mean_e, var_e, t_star, n = -10.0, 4.0, 0.8, 25
    r = to_reduced(mean_e, var_e, t_star, n)
    assert np.isclose(r["E_star_per_N"], mean_e / (n * eps0))
    kT = t_star * eps0
    assert np.isclose(r["Cv_over_kB"], var_e / (kT**2 * n))


# ---------------------------------------------------------------------------
# Per-point equilibration budget
# ---------------------------------------------------------------------------


def test_equilibration_steps_floors_easy_points():
    # warm + dilute is the easy corner: stays at the baseline floor
    assert equilibration_steps(1.6, 0.15, base=10_000, max_steps=60_000) == 10_000


def test_equilibration_steps_caps_hard_points():
    # cold + dense overshoots the budget and is clamped to the cap
    assert equilibration_steps(0.2, 0.55, base=10_000, max_steps=60_000) == 60_000


def test_equilibration_steps_monotonic_in_difficulty():
    # colder needs more (fixed density); denser needs more (fixed temperature)
    assert equilibration_steps(0.4, 0.35) > equilibration_steps(0.8, 0.35)
    assert equilibration_steps(0.6, 0.55) > equilibration_steps(0.6, 0.25)


# ---------------------------------------------------------------------------
# NVT mode of the sampler
# ---------------------------------------------------------------------------


def test_nvt_holds_box_fixed(tmp_path):
    frame = generate_random_config(n_particles=27, density=0.3, seed=2)
    metro = MetropolisCalculator(
        temp=300,
        pressure=0.0,
        init_frame=frame,
        npt_ensemble=False,
        output_dir=str(tmp_path / "s"),
    )
    v0 = np.linalg.det(metro.current_frame.cell)
    for _ in range(400):
        metro.step()
    v1 = np.linalg.det(metro.current_frame.cell)
    assert np.isclose(v0, v1)  # box unchanged
    assert len(metro.vol_decisions) == 0  # no volume moves attempted


def test_npt_default_still_attempts_volume_moves(tmp_path):
    frame = generate_random_config(n_particles=27, density=0.3, seed=3)
    metro = MetropolisCalculator(
        temp=300,
        pressure=0.0,
        init_frame=frame,
        output_dir=str(tmp_path / "s"),
    )
    assert metro.npt_ensemble is True
    for _ in range(400):
        metro.step()
    assert len(metro.vol_decisions) > 0  # NPT default is unchanged


def test_nvt_equilibrate_tunes_deltas_and_holds_box(tmp_path):
    """equilibrate() in NVT mode adapts step sizes and keeps the box fixed."""
    frame = generate_random_config(n_particles=64, density=0.3, seed=5)
    metro = MetropolisCalculator(
        temp=300,
        pressure=0.0,
        init_frame=frame,
        npt_ensemble=False,
        output_dir=str(tmp_path / "s"),
    )
    v0 = np.linalg.det(metro.current_frame.cell)
    assert metro.pos_delt == 0.15  # timid default
    # runs block_update with dynamic_delta in NVT (no vol-decisions nan crash)
    metro.equilibrate(
        20 * 64,
        block_size=4 * 64,
        dynamic_delta=True,
        buffer_size=10_000,
        progress=False,
    )
    v1 = np.linalg.det(metro.current_frame.cell)
    assert np.isclose(v0, v1)  # NVT: box held fixed
    assert len(metro.vol_decisions) == 0  # no volume moves attempted
    assert metro.pos_delt > 0.15  # adaptive deltas grew the timid default


def test_larger_max_scale_converges_faster(tmp_path):
    """A bigger max_scale tunes the over-accepting moves faster per block."""

    def equil_pos_delt(max_scale):
        random.seed(1)
        np.random.seed(1)
        frame = generate_random_config(n_particles=64, density=0.3, seed=5)
        metro = MetropolisCalculator(
            temp=300,
            pressure=0.0,
            init_frame=frame,
            npt_ensemble=False,
            output_dir=str(tmp_path / f"s{max_scale}"),
        )
        metro.equilibrate(
            20 * 64,
            block_size=4 * 64,
            dynamic_delta=True,
            buffer_size=10_000,
            progress=False,
            max_scale=max_scale,
            min_scale=1.0 / max_scale,
        )
        return metro.pos_delt

    gentle = equil_pos_delt(1.1)
    aggressive = equil_pos_delt(2.0)
    assert aggressive > gentle


# ---------------------------------------------------------------------------
# run_state_point end-to-end smoke
# ---------------------------------------------------------------------------


def test_run_state_point_returns_sane_observables(tmp_path):
    res = run_state_point(
        t_star=0.8,
        rho_star=0.3,
        n_particles=27,
        num_steps=3 * 27,  # ~3 passes of production
        num_eq_steps=2 * 27,  # ~2 passes of equilibration
        block_size=27,  # one recorded frame per pass
        seed=4,
        scratch_dir=str(tmp_path / "scan"),
    )
    assert set(res) >= {
        "T_star",
        "rho_star",
        "E_star_per_N",
        "Cv_over_kB",
        "S",
        "pos_acc",
        "or_acc",
    }
    assert res["T_star"] == 0.8 and res["rho_star"] == 0.3
    assert np.isfinite(res["E_star_per_N"])
    assert res["Cv_over_kB"] >= 0.0  # variance is non-negative
    assert 0.0 <= res["S"] <= 1.0
    assert 0.0 <= res["pos_acc"] <= 1.0
    assert 0.0 <= res["or_acc"] <= 1.0


def test_run_state_point_seed_determines_results(tmp_path):
    """A point is fully determined by its seed -- the basis for safe parallelism.

    Same seed -> identical observables regardless of scratch dir or call order;
    a different seed -> a different trajectory.
    """
    kw = dict(
        t_star=0.8,
        rho_star=0.3,
        n_particles=27,
        num_steps=3 * 27,
        num_eq_steps=2 * 27,
        block_size=27,
    )
    a = run_state_point(**kw, seed=7, scratch_dir=str(tmp_path / "a"))
    b = run_state_point(**kw, seed=7, scratch_dir=str(tmp_path / "b"))
    assert a["E_star_per_N"] == b["E_star_per_N"]
    assert a["Cv_over_kB"] == b["Cv_over_kB"]
    assert a["S"] == b["S"]

    c = run_state_point(**kw, seed=8, scratch_dir=str(tmp_path / "c"))
    assert not np.isclose(c["E_star_per_N"], a["E_star_per_N"])


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------


def _small_cfg(scratch_root):
    return {
        "n_particles": 27,
        "num_steps": 2 * 27,
        "block_size": 27,
        "buffer_size": 100,
        "eq_base": 2 * 27,
        "eq_max": 4 * 27,
        "seed0": 100,
        "scratch_root": scratch_root,
    }


def test_evaluate_point_matches_direct_run(tmp_path):
    """The pool worker reproduces an equivalent direct run_state_point call."""
    cfg = _small_cfg(str(tmp_path / "w"))
    k, row = _evaluate_point(0, 0.8, 0.3, cfg)

    num_eq = equilibration_steps(0.8, 0.3, base=cfg["eq_base"], max_steps=cfg["eq_max"])
    direct = run_state_point(
        0.8,
        0.3,
        cfg["n_particles"],
        num_steps=cfg["num_steps"],
        num_eq_steps=num_eq,
        block_size=cfg["block_size"],
        seed=cfg["seed0"] + 0,
        scratch_dir=str(tmp_path / "d"),
        buffer_size=cfg["buffer_size"],
    )
    assert k == 0
    assert row["E_star_per_N"] == direct["E_star_per_N"]
    assert row["S"] == direct["S"]


def test_scan_runs_through_process_pool(tmp_path):
    """Points dispatched through a real ProcessPoolExecutor all come back intact."""
    cfg = _small_cfg(str(tmp_path / "p"))
    grid = [(0.8, 0.3), (1.0, 0.3)]
    rows_by_k = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(_evaluate_point, k, t, r, cfg): k
            for k, (t, r) in enumerate(grid)
        }
        for fut in as_completed(futures):
            k, row = fut.result()
            rows_by_k[k] = row
    assert set(rows_by_k) == {0, 1}
    for row in rows_by_k.values():
        assert np.isfinite(row["E_star_per_N"])
        assert 0.0 <= row["S"] <= 1.0


# ---------------------------------------------------------------------------
# Full driver: main() end-to-end through the pool
# ---------------------------------------------------------------------------


def test_main_writes_csv_and_plot(tmp_path):
    """main() runs the grid through the real ProcessPoolExecutor and writes a
    well-formed nvt_scan.csv (one row per grid point) plus nvt_scan.png."""
    out_dir = str(tmp_path / "scan")
    main(
        t_star_grid=(0.6, 1.0),
        rho_star_grid=(0.3,),
        n_particles=27,
        num_steps=2 * 27,
        eq_base=27,
        eq_max=54,
        out_dir=out_dir,
    )

    csv_path = os.path.join(out_dir, "nvt_scan.csv")
    png_path = os.path.join(out_dir, "nvt_scan.png")
    heatmap_path = os.path.join(out_dir, "nvt_scan_heatmap.png")
    assert os.path.exists(csv_path), "nvt_scan.csv was not written"
    assert os.path.exists(png_path), "nvt_scan.png was not written"
    assert os.path.exists(heatmap_path), "nvt_scan_heatmap.png was not written"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # one row per grid point: 2 T* x 1 rho*
    assert len(rows) == 2
    assert {"T_star", "rho_star", "E_star_per_N", "Cv_over_kB", "S"} <= set(rows[0])
    for r in rows:
        assert np.isfinite(float(r["E_star_per_N"]))
        assert 0.0 <= float(r["S"]) <= 1.0


# ---------------------------------------------------------------------------
# Heatmaps + re-plot from CSV
# ---------------------------------------------------------------------------


def _synthetic_rows(t_vals, rho_vals):
    """A full Cartesian grid of rows with arbitrary-but-finite observables."""
    return [
        {
            "T_star": t,
            "rho_star": r,
            "E_star_per_N": -1.0 / t - r,
            "Cv_over_kB": 1.0 / t,
            "S": max(0.0, 1.0 - t),
            "pos_acc": 0.3,
            "or_acc": 0.3,
            "pos_delt": 1.0,
            "or_delt": 1.0,
        }
        for r in rho_vals
        for t in t_vals
    ]


def test_plot_heatmaps_writes_png(tmp_path):
    """plot_heatmaps renders a phase-diagram PNG on a non-uniform T* grid."""
    out_dir = str(tmp_path / "h")
    os.makedirs(out_dir)
    # deliberately uneven T* spacing (0.1 steps then a jump) to exercise _cell_edges
    t_vals = [0.2, 0.3, 0.4, 1.0, 1.6]
    rho_vals = [0.15, 0.35, 0.55]
    plot_heatmaps(_synthetic_rows(t_vals, rho_vals), t_vals, rho_vals, out_dir)
    assert os.path.exists(os.path.join(out_dir, "nvt_scan_heatmap.png"))


def test_plot_heatmaps_tolerates_missing_points(tmp_path):
    """A grid with a hole (a failed point) still renders -- the cell goes blank."""
    out_dir = str(tmp_path / "h")
    os.makedirs(out_dir)
    t_vals = [0.2, 0.4, 0.6]
    rho_vals = [0.2, 0.4]
    rows = _synthetic_rows(t_vals, rho_vals)
    rows.pop()  # drop one grid point
    plot_heatmaps(rows, t_vals, rho_vals, out_dir)
    assert os.path.exists(os.path.join(out_dir, "nvt_scan_heatmap.png"))


def test_replot_regenerates_both_figures_from_csv(tmp_path):
    """replot() reads an existing CSV and rebuilds line plots + heatmap, with no
    grids passed in -- they are recovered from the CSV's unique T*/rho* values."""
    out_dir = str(tmp_path / "r")
    os.makedirs(out_dir)
    t_vals = [0.2, 0.4, 1.0]
    rho_vals = [0.15, 0.45]
    rows = _synthetic_rows(t_vals, rho_vals)

    csv_path = os.path.join(out_dir, "nvt_scan.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # load_rows round-trips the numeric columns back to floats
    loaded = load_rows(csv_path)
    assert len(loaded) == len(rows)
    assert loaded[0]["T_star"] == rows[0]["T_star"]

    replot(out_dir)
    assert os.path.exists(os.path.join(out_dir, "nvt_scan.png"))
    assert os.path.exists(os.path.join(out_dir, "nvt_scan_heatmap.png"))
