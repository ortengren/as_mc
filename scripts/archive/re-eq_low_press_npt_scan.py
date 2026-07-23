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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import ase.db
import numpy as np
from tqdm import tqdm

from asmcmc.npt_equilibration import _continue_point, plot_point_results

# Point dir names (T{temp}_P{pressure}); both replica seeds under each are
# extended by the tier's step budget.
EXTEND_500 = {"T10_P1.5e-06", "T10_P5e-07"}
EXTEND_375 = set()
EXTEND_250 = set()
EXTEND_125 = {"T200_P5e-07"}


def extra_steps_for(point_name):
    """Extra steps for a point, or None if it's equilibrated (-> production)."""
    if point_name in EXTEND_500:
        return 500_000
    if point_name in EXTEND_375:
        return 375_000
    if point_name in EXTEND_250:
        return 250_000
    if point_name in EXTEND_125:
        return 125_000
    return None


def discover_points(in_dir, out_dir):
    """Return {point_name: [seed_dir, ...]} for every resumable point under
    in_dir -- ALL replica seed dirs per point (this scan has two)."""
    points = {}
    for db_path in glob.glob(os.path.join(in_dir, "T*_P*", "*", "simulation.db")):
        seed_dir = os.path.dirname(db_path)
        name = os.path.basename(os.path.dirname(seed_dir))
        points.setdefault(name, []).append(seed_dir)
        print(db_path)
    return {name: sorted(seeds) for name, seeds in points.items()}


def main():
    discover_points("results/low_press_npt_scan")


if __name__ == "__main__":
    main()
