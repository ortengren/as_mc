"""Tiered resume of the low-pressure NPT *equilibration* scan
(results/low_press_npt_scan).

Same idea as extend_npt_scan.py, with one structural difference: this scan was
run with TWO replicas per (T, P) point, so a tier names a *point*
(``T{temp}_P{pressure}``) and BOTH replica seed dirs under it are extended by
that point's step budget. (extend_npt_scan.py's discover_points keyed a dict by
point name, which for a 2-replica scan would silently keep only one seed.)

Gate (unchanged): equilibration only has to remove the one-directional
relaxation away from the rho*=1.4 columnar start, NOT flatten fluctuations. The
625k production run that follows samples the fluctuations. So points are tiered
on whether a *persistent one-directional trend* (and/or a between-replica
density disagreement) remains over the back half -- not on a flat tail.

Reuses the tested ``_continue_point`` worker + ``plot_point_results`` from
asmcmc.npt_equilibration. All tiers share ONE spawn pool, longest job first, so
the +250k runs (the long pole) start at t=0 and the workers stay busy.

    python scripts/extend_low_press_npt_scan.py --dry-run   # print the plan, run nothing
    python scripts/extend_low_press_npt_scan.py             # launch the extension
    python scripts/extend_low_press_npt_scan.py --max-workers 8
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import ase.db
import numpy as np
from tqdm import tqdm

from asmcmc.npt_equilibration import _continue_point, plot_point_results

# Point dir names (T{temp}_P{pressure}); both replica seeds under each are
# extended by the tier's step budget.
EXTEND_375 = {"T10_P1.5e-06", "T10_P1e-06", "T10_P5e-07", "T50_P5e-07", "T100_P5e-07"}
EXTEND_250 = set()
EXTEND_125 = set()


def extra_steps_for(point_name):
    """Extra steps for a point, or None if it's equilibrated (-> production)."""
    if point_name in EXTEND_375:
        return 375_000
    if point_name in EXTEND_250:
        return 250_000
    if point_name in EXTEND_125:
        return 125_000
    return None


def back_half_trend(db_path):
    """(systematic trend %, fluctuation %) of volume over the back half.

    Printed for transparency only -- the tiers above are fixed, not recomputed
    from this. Trend = linear-fit net change across the back half / mean;
    fluctuation = std / mean.
    """
    rows = list(ase.db.connect(db_path).select())
    step = np.array([r.key_value_pairs["step"] for r in rows])
    vol = np.array([r.key_value_pairs["vol"] for r in rows])
    h = len(vol) // 2
    sx, vy = step[h:], vol[h:]
    slope = np.polyfit(sx, vy, 1)[0]
    trend = slope * (sx[-1] - sx[0]) / vy.mean() * 100.0
    fluct = vy.std() / vy.mean() * 100.0
    return trend, fluct


def discover_points(out_dir):
    """Return {point_name: [seed_dir, ...]} for every resumable point under
    out_dir -- ALL replica seed dirs per point (this scan has two)."""
    points = {}
    for db_path in glob.glob(os.path.join(out_dir, "T*_P*", "*", "equilibration.db")):
        seed_dir = os.path.dirname(db_path)
        name = os.path.basename(os.path.dirname(seed_dir))
        points.setdefault(name, []).append(seed_dir)
    return {name: sorted(seeds) for name, seeds in points.items()}


def main():
    points = discover_points(args.out_dir)
    if not points:
        print(f"No points found under {args.out_dir}")
        return

    # Sanity check: every named point must actually exist on disk.
    missing = (EXTEND_375 | EXTEND_250 | EXTEND_125) - points.keys()
    if missing:
        print(f"WARNING: named points not found on disk (skipped): {sorted(missing)}")

    todo = []  # (point_name, seed_dir, extra_steps) -- one entry per replica
    n_runs = sum(len(seeds) for seeds in points.values())
    print(f"\nLow-pressure NPT equilibration resume plan ({args.out_dir}):\n")
    print(f"  {'point':16s} {'seed':>7} {'tier':>7}   {'trend%':>7} {'fluct%':>7}")
    for name in sorted(points):
        extra = extra_steps_for(name)
        tier = "prod" if extra is None else f"+{extra // 1000}k"
        for seed_dir in points[name]:
            trend, fluct = back_half_trend(os.path.join(seed_dir, "equilibration.db"))
            print(
                f"  {name:16s} {os.path.basename(seed_dir):>7} {tier:>7}   "
                f"{trend:+7.2f} {fluct:7.2f}"
            )
            if extra is not None:
                todo.append((name, seed_dir, extra))

    todo.sort(key=lambda t: -t[2])  # longest first: long pole starts at t=0
    total_units = sum(extra for _, _, extra in todo)
    print(
        f"\n  {n_runs - len(todo)} run(s) -> production (no extra equil); "
        f"{len(todo)} replica run(s) to extend, {total_units:,} total step-units."
    )

    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return
    if not todo:
        print("\nNothing to extend.")
        return

    cfg = {
        "block_size": None,
        "buffer_size": 100,
        "dynamic_delta": True,
        "vol_delt": None,
    }
    num_workers = min(args.max_workers, os.cpu_count() or 1, len(todo))
    print(f"\nLaunching on {num_workers} worker(s)...\n")

    done, failures = [], []
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(_continue_point, d, extra, cfg): (name, d)
            for name, d, extra in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extending"):
            name, d = futures[fut]
            try:
                done.append(fut.result())
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- log & continue, like extend_points
                failures.append(d)
                print(f"\n  {name} ({d}) failed: {exc!r}")

    for d in sorted(done):
        try:
            plot_point_results(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  diagnostics for {d} failed: {exc!r}")

    print(f"\nExtended {len(done)}/{len(todo)} replica run(s).")
    if failures:
        print(f"{len(failures)} failed: {sorted(failures)}")
    print("Re-run the back-half trend check (--dry-run) before starting production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/low_press_npt_scan")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, launch nothing"
    )
    args = parser.parse_args()
    main()
