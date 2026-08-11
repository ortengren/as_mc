"""Score the UMA MLIP against the Cacelli et al. ab initio benzene dimers.

The 197 MP2/6-31G* interaction energies in `data/new_data/3648_1_supplements`
are the repo's only near-ground-truth pair data, and they are the reference
`asmcmc/validation.py` scores every candidate coarse-grained potential
against. They are also few, orientation-sparse, and of modest quantum-chemistry
quality. If UMA reproduces them, it can label orders of magnitude more dimer
geometries than 197 -- which is what a Delta-learning or AniSOAP pair model
needs.

This rebuilds each ab initio row as the 24-atom dimer it describes
(`validation.cacelli_dimer_frames`), evaluates it with UMA, and reports the
result through the same `DimerBenchmark` the coarse-grained benchmark uses --
so MP2, UMA and the GBQIII pair potential land on one set of axes.

    python scripts/uma_cacelli_dimers.py [--device cuda]

Everything is cached to `--out-dir` so the notebook that plots this never has
to import fairchem:

    dimer_energies.csv  per row: geometry, family, MP2, UMA, GBQIII
    dimers.xyz          the 197 rebuilt frames, energies in info
    family_curves.csv   dense cofacial / PD / T-shaped scans
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import write

from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.utils.uma import DEFAULT_UMA_MODEL, load_uma_calculator
from asmcmc.utils.validation import (
    EULER_SEQ,
    EV_TO_KCAL,
    atomistic_pair_energies,
    atomistic_scan,
    cacelli_dimer_frames,
    cg_scan,
    family_labels,
    family_scan_geometry,
    load_cacelli_dimers,
    score_model_energies,
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
        "--out-dir", type=Path, default=REPO_ROOT / "results/validation/uma_cacelli"
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
            # Same call the scan used, so the offsets line up point for point
            # -- the two sources scan at different resolutions.
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
    bench_uma = score_model_energies(
        e_uma, data, name=f"UMA {args.model}", scan_fn=uma_scan
    )
    bench_gbq = score_model_energies(
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
