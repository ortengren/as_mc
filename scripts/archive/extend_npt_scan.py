"""One-off tiered resume of the NPT *equilibration* scan in results/npt_scan.

Equilibration only has to remove the initial-condition bias -- the monotonic
relaxation away from the rho*=1.4 columnar start -- NOT to flatten fluctuations.
The 625k production run that follows samples the fluctuations and forms the
averages. So points are gated on whether a *persistent one-directional trend*
remains over the back half of the trajectory, not on a flat tail.

Tiers (chosen from the back-half trend-vs-fluctuation analysis; see the table in
the project notes). Everything not listed is considered equilibrated and goes
straight to production -- including the hot/transition points (145/1e-5, 160/*,
175/*), whose large volume swings are physical near-transition *fluctuation*,
not relaxation: more equilibration won't tighten them, longer production +
replicas will.

    +250k -- clear, steep relaxation still underway
    +125k -- mild residual densification
    (none) -- equilibrated -> production

The 145/0.0001 and 145/0.0003 points trend *upward* with trend ~ fluctuation;
they are placed in +250k as "likely still relaxing" but are borderline -- move
them to PRODUCTION (delete from EXTEND_250) if you'd rather treat them with
replicas instead.

Reuses the tested ``_continue_point`` worker + ``plot_point_results`` from
asmcmc.utils.npt_equilibration. All tiers share ONE spawn pool, longest job first, so
the +250k points (the long pole) start at t=0 and the workers stay busy.

    python extend_npt_scan.py --dry-run     # print the plan, run nothing
    python extend_npt_scan.py               # launch the extension
    python extend_npt_scan.py --max-workers 8 --out-dir results/npt_scan
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import ase.db
import numpy as np
from tqdm import tqdm

from asmcmc.utils.npt_equilibration import _continue_point, plot_point_results

# Point dir names (T{temp}_P{pressure}) -> extra equilibration steps.
EXTEND_250 = {
    "T85_P0.0003",
    "T115_P0.0001",
    "T115_P1e-05",
    "T130_P1e-05",
    "T145_P0.0001",
    "T145_P0.0003",
}
EXTEND_125 = {
    "T40_P1e-05",
    "T70_P0.0001",
    "T85_P1e-05",
    "T100_P0.0003",
    "T115_P0.0003",
    "T130_P0.0001",
}


def extra_steps_for(point_name):
    """Extra steps for a point, or None if it's equilibrated (-> production)."""
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
    """Return {point_name: seed_dir} for every resumable point under out_dir."""
    points = {}
    for db_path in glob.glob(os.path.join(out_dir, "T*_P*", "*", "equilibration.db")):
        seed_dir = os.path.dirname(db_path)
        name = os.path.basename(os.path.dirname(seed_dir))
        points[name] = seed_dir
    return points


def main():
    points = discover_points(args.out_dir)
    if not points:
        print(f"No points found under {args.out_dir}")
        return

    # Sanity check: every named point must actually exist on disk.
    missing = (EXTEND_250 | EXTEND_125) - points.keys()
    if missing:
        print(f"WARNING: named points not found on disk (skipped): {sorted(missing)}")

    todo = []  # (point_name, seed_dir, extra_steps)
    print(f"\nNPT equilibration resume plan ({args.out_dir}):\n")
    print(f"  {'point':16s} {'tier':>8}   {'trend%':>7} {'fluct%':>7}")
    for name in sorted(points):
        seed_dir = points[name]
        extra = extra_steps_for(name)
        trend, fluct = back_half_trend(os.path.join(seed_dir, "equilibration.db"))
        tier = "prod" if extra is None else f"+{extra // 1000}k"
        print(f"  {name:16s} {tier:>8}   {trend:+7.2f} {fluct:7.2f}")
        if extra is not None:
            todo.append((name, seed_dir, extra))

    todo.sort(key=lambda t: -t[2])  # longest first: long pole starts at t=0
    n_skip = len(points) - len(todo)
    total_units = sum(extra for _, _, extra in todo)
    print(
        f"\n  {n_skip} point(s) -> production (no extra equil); "
        f"{len(todo)} to extend, {total_units:,} total step-units."
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
            pool.submit(_continue_point, d, extra, cfg): name for name, d, extra in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extending"):
            name = futures[fut]
            try:
                done.append(fut.result())
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- log & continue, like extend_points
                failures.append(name)
                print(f"\n  point {name} failed: {exc!r}")

    for d in sorted(done):
        try:
            plot_point_results(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  diagnostics for {d} failed: {exc!r}")

    print(f"\nExtended {len(done)}/{len(todo)} point(s).")
    if failures:
        print(f"{len(failures)} failed: {sorted(failures)}")
    print("Re-run the back-half trend check before starting production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/npt_scan")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, launch nothing"
    )
    args = parser.parse_args()
    main()
