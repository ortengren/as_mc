"""Regenerate the UMA reference energies for the 197 Cacelli et al. dimers.

Rebuilds each row of ``data/cacelli_2004_dimers`` as the 24-atom dimer it
describes, evaluates it with the UMA MLIP, and writes what the dimer benchmark
and ``notebooks/uma_vs_cacelli.ipynb`` read, so neither needs fairchem:

    dimer_energies.csv  per row: geometry, family, MP2, UMA and GBQIII energies
    family_curves.csv   dense cofacial / parallel-displaced / T-shaped scans
    dimers.xyz          the 197 rebuilt frames, for viewing (not tracked)

It also prints UMA and GBQIII scored against the supplement's MP2 energies.
Needs the ``uma`` extra and a Hugging Face login for the gated checkpoint.

    python scripts/uma_cacelli_dimers.py [--device cuda]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import write

from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.units import EV_TO_KCAL
from asmcmc.delta_learning.uma import DEFAULT_UMA_MODEL, load_uma_calculator
from asmcmc.delta_learning.dimer_benchmark import (
    EULER_SEQ,
    atomistic_pair_energies,
    atomistic_scan,
    cacelli_dimer_frames,
    cg_scan,
    family_labels,
    family_scan_geometry,
    load_cacelli_dimers,
    score_energies,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_UMA_MODEL)
    parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    parser.add_argument(
        "--euler-seq",
        default=EULER_SEQ,
        help="scipy Euler sequence for the 53 angle-carrying rows.",
    )
    parser.add_argument(
        "--scan-points",
        type=int,
        default=121,
        help="Points per dense family scan; 0 skips them (wells then come "
        "from the ab initio rows themselves).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data/uma_dimers"
    )
    return parser.parse_args()


def write_rows(path, data, labels, e_uma, e_gbq):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            "row_index x y z alpha beta gamma com_sep family "
            "e_mp2_kcal e_uma_kcal e_gbq_kcal".split()
        )
        for k in range(len(data)):
            w.writerow(
                [
                    k,
                    *(f"{v:.4f}" for v in data.r[k]),
                    *(f"{v:.2f}" for v in data.euler_deg[k]),
                    f"{np.linalg.norm(data.r[k]):.4f}",
                    labels[k],
                    f"{data.energy_kcal[k]:.6f}",
                    f"{e_uma[k]:.6f}",
                    f"{e_gbq[k]:.6f}",
                ]
            )


def write_curves(path, curves):
    """Dense family scans, with the offset vector each point was taken at.

    ``parallel_displaced`` is a two-parameter family (stack height + lateral
    slip), so ``|r|`` alone does not identify a geometry -- plotting it
    against ``|r|`` scatters the family instead of drawing a curve. Writing
    the components lets a reader pick the abscissa that actually varies.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "source", "r", "dx", "dy", "dz", "energy_kcal"])
        for family, series in curves.items():
            for source, (energies, r_values, offsets) in series.items():
                for offset, r, e in zip(offsets, r_values, energies):
                    w.writerow(
                        [family, source, f"{r:.4f}", *(f"{v:.4f}" for v in offset),
                         f"{e:.6f}"]
                    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_cacelli_dimers()
    frames = cacelli_dimer_frames(data, euler_seq=args.euler_seq)
    print(f"{len(data)} ab initio rows -> {len(frames)} atomistic dimers")

    calculator = load_uma_calculator(args.model, device=args.device)
    e_uma = atomistic_pair_energies(frames, calculator)
    e_gbq = CACELLI_POTENTIAL.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL

    # Record each family's dense scan on the way past, so the well search and
    # the plotted curve are the same evaluations rather than two passes.
    curves = {}

    def recording(source, scan, n_points):
        def wrapped(family, data_, i_min):
            energies, r_values = scan(family, data_, i_min)
            # Use the same call as the scan so the offsets line up point for
            # point, since the two sources scan at different resolutions.
            offsets = family_scan_geometry(family, data_, i_min, n_points)[1]
            curves.setdefault(family, {})[source] = (energies, r_values, offsets)
            return energies, r_values

        return wrapped

    uma_scan = (
        recording(
            "uma",
            atomistic_scan(calculator, None, args.euler_seq, args.scan_points),
            args.scan_points,
        )
        if args.scan_points
        else None
    )
    bench_uma = score_energies(
        e_uma, data, name=f"UMA {args.model}", scan_fn=uma_scan
    )
    bench_gbq = score_energies(
        e_gbq,
        data,
        name=CACELLI_POTENTIAL.name,
        scan_fn=recording("gbq", cg_scan(CACELLI_POTENTIAL), 601),
    )

    print()
    print(bench_uma.summary())
    print()
    print(bench_gbq.summary())

    labels = family_labels(data)
    write_rows(args.out_dir / "dimer_energies.csv", data, labels, e_uma, e_gbq)
    write_curves(args.out_dir / "family_curves.csv", curves)

    for k, frame in enumerate(frames):
        frame.info.update(
            {"uma_kcal": float(e_uma[k]), "gbq_kcal": float(e_gbq[k]),
             "family": labels[k], "uma_model": args.model}
        )
    write(args.out_dir / "dimers.xyz", frames)

    print(f"\nWrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
