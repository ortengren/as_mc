"""The hyperparameter sweep: find which AniSOAP settings give the best Delta-model.

This module does all of the fit's file handling. :mod:`asmcmc.delta_learning.descriptors`
and :mod:`asmcmc.delta_learning.model` only compute and never write files, so
moving the bookkeeping elsewhere (to signac or a job queue, say) would only mean
changing this file. ``Hypers.to_dict`` already has the form of a signac state
point for that reason.

The sweep varies ``max_angular``, ``max_radial`` and ``cutoff_radius``. The first
two set the resolution of the representation (the number of features runs from
64 to 490 over the default grid), and the third sets how far the correction can
see. The ellipsoid semiaxes and the radial Gaussian width stay at the ``Hypers``
defaults. They describe the particle rather than the basis, and the AniSOAP paper
tuned them without supervision (against atomistic SOAP), which is a different
goal from predicting Delta.

Three things are built once, before the loop:

* the :class:`~asmcmc.delta_learning.descriptors.TrainingSet`, parsed in the
  parent process so the spawned workers read a cached ``.npz`` instead of each
  re-parsing 32 MB of extxyz at the same time;
* the train/test split, because every point has to be scored on the same
  held-out frames, otherwise the sweep compares splits as much as
  representations;
* the dimer reference data, which
  :func:`~asmcmc.delta_learning.dimer_benchmark.load_reference_dimers` would
  otherwise re-read on every call.

Every point is also run through the dimer benchmark against the reference dimers
(UMA by default), and its ``improves_on_baseline`` verdict is saved next to its
RMSE. Nothing is filtered out on it, but it matters for ranking, since a good
RMSE can hide a model that makes the wells worse. Rank on
``improves_on_baseline`` and the well errors, not on ``stacking_bound``, which
only checks that the cofacial stack is bound at all.

``--regate`` re-scores a finished sweep against a different reference by
re-running the benchmark on each point's saved ``model.npz``. The fit doesn't
depend on the benchmark's reference (the target is always ``E_UMA - E_GBQ``), so
nothing needs to be refitted.
"""

from __future__ import annotations

import os

# Each worker runs its own BLAS. Without this they oversubscribe the machine and
# the pool runs slower than a single process. It has to be set before numpy is
# imported, as in dataset.py.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from tqdm import tqdm

from asmcmc.delta_learning.dataset import load_dataset_config
from asmcmc.delta_learning.descriptors import (
    Hypers,
    campaign_descriptors,
    load_training_set,
)
from asmcmc.delta_learning.model import (
    AniSOAPDeltaPotential,
    DeltaModel,
    fit_delta,
    train_test_split,
)
from asmcmc.delta_learning.dimer_benchmark import (
    DEFAULT_REFERENCE,
    REFERENCES,
    dimer_benchmark,
    load_reference_dimers,
)

DEFAULT_CAMPAIGN = "results/cluster_train"
DEFAULT_OUT = "results/anisoap_fit"

# The representation grid. The angular/radial corners span the 64-to-490 feature
# range the model docstring quotes. The cutoffs run from the first coordination
# shell (the herringbone runs use a 6.8 A neighbour list) out to 12 A, asking
# whether the correction needs to see past it.
DEFAULT_ANGULAR = (3, 5, 7, 9)
DEFAULT_RADIAL = (3, 4, 5, 6)
DEFAULT_CUTOFF = (6.0, 7.5, 9.0, 12.0)

# Four workers by default, because each point holds its full descriptor matrix
# and the campaign geometry in memory, so memory limits the pool as much as the
# number of cores does. Use --workers to raise it on a machine with more memory.
DEFAULT_WORKERS = 4

METRICS_NAME = "metrics.json"
HYPERS_NAME = "hypers.json"
MODEL_NAME = "model.npz"
POINTS_DIRNAME = "points"
COMPARISON_NAME = "comparison.csv"

COMPARISON_FIELDS = [
    "key",
    "max_angular",
    "max_radial",
    "cutoff_radius",
    "n_features",
    "test_skill",
    "test_well_skill",
    "test_rmse",
    "test_well_rmse",
    "test_r2",
    "n_zero_test",
    "reference",
    "improves_on_baseline",
    "well_rmse_gain_kcal",
    "well_rmse_kcal",
    "baseline_well_rmse_kcal",
    "stacking_bound",
    "stacking_energy_kcal",
    "full_pearson_r",
    "alpha",
    "descriptors_s",
    "fit_s",
]


def build_grid(angular=DEFAULT_ANGULAR, radial=DEFAULT_RADIAL, cutoffs=DEFAULT_CUTOFF, base=None):
    """The sweep points, most expensive first.

    The cost grows along all three axes, so the slowest points are submitted
    first and the quick ones fill in at the end. Otherwise the pool would finish
    the cheap points early and then wait on one slow one. The order only affects
    scheduling: each point is identified by its :attr:`Hypers.key`, so its output
    directory doesn't depend on its place in the queue.
    """
    base = base or Hypers()
    grid = [
        base.replace(max_angular=int(l), max_radial=int(n), cutoff_radius=float(rc))
        for rc, l, n in itertools.product(cutoffs, angular, radial)
    ]
    return sorted(
        grid,
        key=lambda h: (h.cutoff_radius, h.max_angular, h.max_radial),
        reverse=True,
    )


def _gate_to_dict(bench):
    """Flatten a :class:`DimerBenchmark` to JSON primitives.

    It has no ``to_dict`` of its own and its ``wells`` hold dataclasses, so the
    three families are unpacked by hand rather than through ``asdict``.
    """
    out = {
        "name": bench.name,
        "reference": bench.reference,
        "stacking_bound": bool(bench.stacking_bound),
        "stacking_energy_kcal": float(bench.stacking_energy_kcal),
        "full_pearson_r": float(bench.full_pearson_r),
        "full_rmse_kcal": float(bench.full_rmse_kcal),
        "well_pearson_r": float(bench.well_pearson_r),
        "well_rmse_kcal": float(bench.well_rmse_kcal),
        "baseline_name": bench.baseline_name,
        "baseline_well_rmse_kcal": float(bench.baseline_well_rmse_kcal),
        "well_rmse_gain_kcal": float(bench.well_rmse_gain_kcal),
        "improves_on_baseline": bool(bench.improves_on_baseline),
    }
    for family, well in bench.wells.items():
        out[family] = {
            "ab_depth": float(well.ab_depth),
            "ab_r": float(well.ab_r),
            "model_at_ab_min": float(well.model_at_ab_min),
            "model_depth": float(well.model_depth),
            "model_r": float(well.model_r),
        }
    return out


def save_model(point_dir, model):
    """Persist a :class:`DeltaModel` so it can be rebuilt without refitting."""
    point_dir = Path(point_dir)
    point_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        point_dir / MODEL_NAME,
        coef=model.coef,
        scale=np.array(model.scale, dtype=float),
        alpha=np.array(model.alpha, dtype=float),
        # Save the hypers with the coefficients, since the model can't be used
        # without them.
        hypers=np.array(json.dumps(model.hypers.to_dict())),
    )


def load_model(point_dir):
    """Rebuild the :class:`DeltaModel` a point fitted.

    This is how a finished sweep point becomes a potential that can be
    benchmarked or run in the sampler, through
    :class:`~asmcmc.delta_learning.model.AniSOAPDeltaPotential`.
    """
    with np.load(Path(point_dir) / MODEL_NAME) as handle:
        return DeltaModel(
            hypers=Hypers.from_dict(json.loads(str(handle["hypers"]))),
            coef=handle["coef"],
            scale=float(handle["scale"]),
            alpha=float(handle["alpha"]),
        )


def _row_from_metrics(record):
    """The ``comparison.csv`` row for one point, from its metrics payload.

    A ``metrics.json`` without ``timing`` or ``gate`` entries leaves those cells
    blank rather than failing the whole comparison.
    """
    hypers, test, gate = record["hypers"], record["test"], record.get("gate") or {}
    timing = record.get("timing") or {}
    return {
        "key": record["key"],
        "max_angular": hypers["max_angular"],
        "max_radial": hypers["max_radial"],
        "cutoff_radius": hypers["cutoff_radius"],
        "n_features": record["extras"]["n_features"],
        "test_skill": test["skill"],
        "test_well_skill": test["well_skill"],
        "test_rmse": test["rmse"],
        "test_well_rmse": test["well_rmse"],
        "test_r2": test["r2"],
        "n_zero_test": record["n_zero_test"],
        # Blank when the stored benchmark result doesn't say which reference it used.
        "reference": gate.get("reference", ""),
        "improves_on_baseline": gate.get("improves_on_baseline", ""),
        "well_rmse_gain_kcal": gate.get("well_rmse_gain_kcal", ""),
        "well_rmse_kcal": gate.get("well_rmse_kcal", ""),
        "baseline_well_rmse_kcal": gate.get("baseline_well_rmse_kcal", ""),
        "stacking_bound": gate.get("stacking_bound", ""),
        "stacking_energy_kcal": gate.get("stacking_energy_kcal", ""),
        "full_pearson_r": gate.get("full_pearson_r", ""),
        "alpha": record["alpha"],
        "descriptors_s": timing.get("descriptors_s", ""),
        "fit_s": timing.get("fit_s", ""),
    }


def fit_and_score_point(hypers_dict, cfg):
    """Fit and score one hyperparameter point, and write its three output files.

    It's a module-level function that takes plain dicts, so its arguments can be
    pickled for a spawned worker. It returns the metrics record, which is the
    same as what it writes to disk.

    If the point's ``metrics.json`` already exists, it's loaded and returned
    instead of refitting, so re-running an interrupted sweep resumes it. The
    ``timing`` in a reloaded record is from the original fit, not from the
    resumed run.

    With ``regate``, a finished point still has the benchmark re-run on its
    ``model.npz``. That's how a finished sweep is re-scored against a different
    reference without refitting.
    """
    hypers = Hypers.from_dict(hypers_dict)
    point_dir = Path(cfg["out_dir"]) / POINTS_DIRNAME / hypers.key
    metrics_path = point_dir / METRICS_NAME

    if metrics_path.exists():
        record = json.loads(metrics_path.read_text())
        # regate is optional, so a cfg written without it still works.
        if cfg.get("regate"):
            # Re-run the benchmark on the saved model instead of refitting. Only
            # the benchmark's reference can have changed: ``test`` and ``train``
            # measure Delta against UMA whatever the benchmark scores against.
            record["gate"] = _gate_to_dict(
                dimer_benchmark(
                    AniSOAPDeltaPotential(load_model(point_dir)), data=cfg["dimers"]
                )
            )
            record.setdefault("meta", {})["reference"] = cfg["meta"]["reference"]
            metrics_path.write_text(json.dumps(record, indent=2))
        record["skipped"] = True
        return record

    # Only the two stages whose cost depends on the hyperparameters are timed:
    # building the descriptors and fitting. Loading the geometry reads a cached
    # npz (~4 ms, the same for every point), and the benchmark's cost doesn't
    # depend on the representation. perf_counter is monotonic, so unlike time()
    # it won't jump if the system clock is adjusted mid-point.
    timing = {}
    training_set = load_training_set(cfg["campaign"], cache_dir=cfg["cache_dir"])

    mark = time.perf_counter()
    X = campaign_descriptors(training_set, hypers)
    timing["descriptors_s"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    result = fit_delta(
        X,
        training_set.delta,
        np.asarray(cfg["train_idx"], dtype=int),
        np.asarray(cfg["test_idx"], dtype=int),
        min_pair_r=training_set.min_pair_r,
        hypers=hypers,
        cv_seed=cfg["cv_seed"],
    )
    timing["fit_s"] = round(time.perf_counter() - mark, 3)

    gate = None
    if cfg["gate"]:
        # Kept separate so a benchmark error doesn't throw away a fit that took minutes.
        try:
            gate = _gate_to_dict(
                dimer_benchmark(AniSOAPDeltaPotential(result.model), data=cfg["dimers"])
            )
        except Exception as exc:
            gate = {"gate_error": f"{type(exc).__name__}: {exc}"}

    record = {
        "key": hypers.key,
        "hypers": hypers.to_dict(),
        "test": result.test.to_dict(),
        "train": result.train.to_dict(),
        "extras": result.extras,
        "n_zero_train": result.n_zero_train,
        "n_zero_test": result.n_zero_test,
        "alpha": result.model.alpha,
        "gate": gate,
        "timing": timing,
        "meta": cfg["meta"],
    }

    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / HYPERS_NAME).write_text(json.dumps(hypers.to_dict(), indent=2))
    save_model(point_dir, result.model)
    # Written last: the resume path takes this file to mean the point is finished,
    # so it mustn't exist before the other files do.
    metrics_path.write_text(json.dumps(record, indent=2))

    record["skipped"] = False
    return record


def write_comparison(path, records):
    """One row per point, ranked by held-out skill."""
    rows = [_row_from_metrics(r) for r in records]
    rows.sort(key=lambda r: (-(r["test_skill"] or 0.0), r["key"]))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return rows


def run_sweep(
    campaign=DEFAULT_CAMPAIGN,
    out_dir=DEFAULT_OUT,
    angular=DEFAULT_ANGULAR,
    radial=DEFAULT_RADIAL,
    cutoffs=DEFAULT_CUTOFF,
    test_frac=0.2,
    split_seed=0,
    cv_seed=0,
    workers=DEFAULT_WORKERS,
    gate=True,
    reference=DEFAULT_REFERENCE,
    regate=False,
    refresh=False,
    progress=True,
    cache_dir=None,
):
    """Run the grid and write ``comparison.csv``.

    Returns the list of metrics records, one per point, in completion order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = out_dir / "cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Parse the campaign here, once, so the workers inherit a warm cache.
    training_set = load_training_set(campaign, cache_dir=cache_dir, refresh=refresh)
    train_idx, test_idx = train_test_split(len(training_set), test_frac, split_seed)

    grid = build_grid(angular, radial, cutoffs)
    cfg = {
        "campaign": str(campaign),
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "cv_seed": cv_seed,
        "gate": gate,
        "regate": regate and gate,
        "dimers": load_reference_dimers(reference) if gate else None,
        "meta": {
            "campaign": str(campaign),
            "reference": reference,
            "n_frames": len(training_set),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_frac": test_frac,
            "split_seed": split_seed,
            "cv_seed": cv_seed,
            "campaign_config": load_dataset_config(campaign),
        },
    }

    print(
        f"{len(grid)} points over {len(training_set)} frames "
        f"({len(train_idx)} train / {len(test_idx)} test), "
        f"gate={'regate ' + reference if cfg['regate'] else 'on' if gate else 'off'}"
    )

    records, failures = [], []
    if workers == 1 or len(grid) == 1:
        for hypers in tqdm(grid, desc="sweep", disable=not progress):
            try:
                records.append(fit_and_score_point(hypers.to_dict(), cfg))
            except Exception as exc:
                failures.append((hypers.key, exc))
                print(f"\n  {hypers.key} failed: {exc!r}")
    else:
        cpu_limit = max(1, (os.cpu_count() or 2) // 2)
        num_workers = min(workers, cpu_limit, len(grid))
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(fit_and_score_point, hypers.to_dict(), cfg): hypers.key
                for hypers in grid
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="sweep", disable=not progress
            ):
                key = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:  # one bad point cannot sink the sweep
                    failures.append((key, exc))
                    print(f"\n  {key} failed: {exc!r}")

    skipped = sum(1 for r in records if r.get("skipped"))
    print(f"\n{len(records)}/{len(grid)} points ({skipped} already done, {len(failures)} failed)")
    for key, exc in failures:
        print(f"  FAILED {key}: {exc!r}")

    if gate and not regate:
        # Finished points return their stored benchmark result as it is, so
        # without this check a changed --reference would quietly produce a
        # comparison.csv that mixes references.
        stale = [
            r["key"]
            for r in records
            if r.get("skipped") and (r.get("gate") or {}).get("reference") != reference
        ]
        if stale:
            print(
                f"WARNING: {len(stale)} finished points were scored against a "
                f"different reference, so comparison.csv mixes references. Re-score "
                f"them with --regate."
            )

    if records:
        write_comparison(out_dir / COMPARISON_NAME, records)
        verdicts = [(r.get("gate") or {}).get("improves_on_baseline") for r in records]
        scored = [v for v in verdicts if v is not None]
        if scored:
            print(
                f"{sum(scored)}/{len(scored)} points improve on GBQIII "
                f"(scored against {reference})"
            )
    return records


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m asmcmc.delta_learning.sweep",
        description="AniSOAP Delta-learning hyperparameter sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN, help="UMA-labelled cluster campaign directory.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Where points/ and comparison.csv are written.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Parsed-campaign cache. Defaults to <out-dir>/cache; point several sweeps "
        "at one shared dir to skip the re-parse.",
    )

    grid = parser.add_argument_group("grid axes")
    grid.add_argument("--angular", type=int, nargs="+", default=list(DEFAULT_ANGULAR))
    grid.add_argument("--radial", type=int, nargs="+", default=list(DEFAULT_RADIAL))
    grid.add_argument("--cutoff", type=float, nargs="+", default=list(DEFAULT_CUTOFF), dest="cutoffs")

    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the held-out split. Keep it fixed across sweeps you want to compare.",
    )
    parser.add_argument("--cv-seed", type=int, default=0, help="RidgeCV fold assignment.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--no-gate",
        dest="gate",
        action="store_false",
        help="Skip the dimer benchmark (~12 s per point).",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        choices=list(REFERENCES),
        help="Reference energies for the dimer benchmark. 'mp2' is only a diagnostic, "
        "because GBQIII was fitted to those rows.",
    )
    parser.add_argument(
        "--regate",
        action="store_true",
        help="Re-run the dimer benchmark on finished points using their saved "
        "models, instead of reusing the stored result. Use this to re-score a "
        "sweep against a new --reference; nothing is refitted (~12 s per point).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-parse the campaign, ignoring the cache.")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    return run_sweep(
        campaign=args.campaign,
        out_dir=args.out_dir,
        angular=tuple(args.angular),
        radial=tuple(args.radial),
        cutoffs=tuple(args.cutoffs),
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        cv_seed=args.cv_seed,
        workers=args.workers,
        gate=args.gate,
        reference=args.reference,
        regate=args.regate,
        refresh=args.refresh,
        progress=args.progress,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    cli()
