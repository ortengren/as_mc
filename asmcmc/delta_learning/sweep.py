"""The hyperparameter sweep: which AniSOAP representation buys Delta-skill.

This module owns **all** persistence for the fit. :mod:`asmcmc.delta_learning.descriptors`
and :mod:`asmcmc.delta_learning.model` are pure computation with no notion of an
output directory, so moving the bookkeeping elsewhere (signac, a queue) touches
this file and nothing else -- ``Hypers.to_dict`` is already shaped as a valid
state point for exactly that.

**What is swept, and what is not.** ``max_angular`` x ``max_radial`` x
``cutoff_radius``. The first two set the representation's resolution (feature
count runs 64 to 490 over the default grid); the third sets how much of the far
field the correction can see at all. The ellipsoid semiaxes and the radial
gaussian width are held at the ``Hypers`` defaults: they describe the *particle*
rather than the basis, and the legacy GFRE study
(``data/anisoap_data/benzenes/hyperparameter_tuning/gfre.py``) tuned them
unsupervised against atomistic SOAP, which is a different objective from
Delta-skill and worth keeping separate from it.

**Three things are built once, before the loop**, because the alternative
silently changes what is being measured:

* the :class:`~asmcmc.delta_learning.descriptors.Geometry` -- parsed in the *parent*
  so spawned workers hit a warm ``.npz`` instead of each re-parsing 32 MB of
  extxyz concurrently;
* the train/test split -- every point must be scored on identical held-out
  frames or the sweep compares splits as much as representations;
* the dimer probe table -- :func:`~asmcmc.delta_learning.dimer_benchmark.load_reference_dimers`
  re-reads the file on every call otherwise.

**The physics probe is recorded, not enforced.** Every point is scored against
the reference dimers (UMA by default) and its ``improves_on_baseline`` verdict
is written next to its RMSE. Nothing is filtered on it here: the GB+Q refit
postmortem is precisely that a model can post good fit parity while being
repulsive at the cofacial stack, so the numbers belong side by side where
ranking happens, not collapsed into a pass/fail that hides which points were
sound. ``improves_on_baseline``, not ``stacking_bound``, is the verdict that
means something -- the latter passed 64/64 points of a sweep in which none
improved on doing nothing.

A finished sweep is re-scored against a different reference with ``--regate``,
which recomputes each point's gate from its saved ``model.npz``. The fit is
reference-independent -- the target is always ``E_UMA - E_GBQ`` -- so only the
probe needs redoing, and refitting to change it would be pure waste.
"""

from __future__ import annotations

import os

# Each worker runs its own BLAS; without this they oversubscribe the machine and
# the pool runs slower than serial. Set before numpy arrives, as the sampler's
# parallel drivers do.
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
    load_geometry,
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

# The representation grid. The angular/radial corners span exactly the 64-to-490
# feature range the fit's docstring quotes. The cutoffs bracket what matters
# physically: 6.8 A is the sampler's own neighbour-list radius, and the campaign
# samples centres out to 15 A, so this asks whether the correction needs to see
# past the first shell.
DEFAULT_ANGULAR = (3, 5, 7, 9)
DEFAULT_RADIAL = (3, 4, 5, 6)
DEFAULT_CUTOFF = (6.0, 7.5, 9.0, 12.0)

# Four cores, deliberately: a point holds the full descriptor matrix alongside
# the campaign geometry, so the pool is bounded by what the machine can hold as
# much as by core count. Raise it with --workers if the box has room.
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

    Cost grows with all three axes, so the longest points are submitted at t=0
    and the short ones backfill the tail -- otherwise the pool finishes its
    cheap work early and then waits on one straggler. Ordering affects
    scheduling only: a point's identity is its :attr:`Hypers.key`, so its output
    directory does not depend on where it landed in the queue.
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
        # The hypers travel with the coefficients: a descriptor matrix is
        # meaningless without them, and deployment needs both.
        hypers=np.array(json.dumps(model.hypers.to_dict())),
    )


def load_model(point_dir):
    """Rebuild the :class:`DeltaModel` a point fitted.

    The consumer-facing entry point: this is what turns a finished sweep into a
    potential the sampler can run, via
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

    Reads ``timing`` tolerantly: a ``metrics.json`` written before the per-stage
    timings existed leaves those cells blank rather than raising a ``KeyError``
    that would sink the whole comparison write.
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
        # Blank on a gate written before the reference was recorded, which is
        # exactly the signal that the row predates the UMA switch.
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


def evaluate_point(hypers_dict, cfg):
    """Fit and score one hyperparameter point, writing its three artifacts.

    Module-level and taking plain dicts so the payload pickles into a spawned
    worker. Returns the metrics record, which is also what landed on disk.

    Idempotent: a point whose ``metrics.json`` already exists is loaded and
    returned rather than refitted, so an interrupted sweep resumes by
    re-running -- the same finished-marker discipline the (T, P) grid uses.
    A skipped point therefore carries the timing of the computation that
    originally produced it, not of the resume: ``timing`` describes the fit,
    never the invocation that read it back.

    Under ``regate`` a skipped point still has its gate recomputed from
    ``model.npz``, which is how a finished sweep is re-scored against a
    different reference without refitting.
    """
    hypers = Hypers.from_dict(hypers_dict)
    point_dir = Path(cfg["out_dir"]) / POINTS_DIRNAME / hypers.key
    metrics_path = point_dir / METRICS_NAME

    if metrics_path.exists():
        record = json.loads(metrics_path.read_text())
        # .get, not []: regate is optional, so a cfg built before it existed
        # (or by hand) still drives the skip path.
        if cfg.get("regate"):
            # Re-gate from the saved model instead of refitting: the probe's
            # reference is the only thing that changed, and ``test``/``train``
            # are already Delta-against-UMA whatever the probe scores against.
            record["gate"] = _gate_to_dict(
                dimer_benchmark(
                    AniSOAPDeltaPotential(load_model(point_dir)), data=cfg["dimers"]
                )
            )
            record.setdefault("meta", {})["reference"] = cfg["meta"]["reference"]
            metrics_path.write_text(json.dumps(record, indent=2))
        record["skipped"] = True
        return record

    # Only the two stages that are hyperparameter-dependent are timed: the
    # descriptor build and the fit. The geometry load is a warm-npz read (~4 ms,
    # constant across points) and the gate does not depend on the representation
    # in any way the sweep is asking about. perf_counter, not time(): a monotonic
    # clock that cannot step under an NTP adjustment mid-point.
    timing = {}
    geometry = load_geometry(cfg["campaign"], cache_dir=cfg["cache_dir"])

    mark = time.perf_counter()
    X = campaign_descriptors(geometry, hypers)
    timing["descriptors_s"] = round(time.perf_counter() - mark, 3)

    mark = time.perf_counter()
    result = fit_delta(
        X,
        geometry.delta,
        np.asarray(cfg["train_idx"], dtype=int),
        np.asarray(cfg["test_idx"], dtype=int),
        min_pair_r=geometry.min_pair_r,
        hypers=hypers,
        cv_seed=cfg["cv_seed"],
    )
    timing["fit_s"] = round(time.perf_counter() - mark, 3)

    gate = None
    if cfg["gate"]:
        # Isolated: a gate failure must not discard a fit that took minutes.
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
    # Written last: it is the finished marker the resume path checks, so it must
    # not appear before the artifacts it claims are present.
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


def main(
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
    geometry = load_geometry(campaign, cache_dir=cache_dir, refresh=refresh)
    train_idx, test_idx = train_test_split(len(geometry), test_frac, split_seed)

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
            "n_frames": len(geometry),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_frac": test_frac,
            "split_seed": split_seed,
            "cv_seed": cv_seed,
            "campaign_config": load_dataset_config(campaign),
        },
    }

    print(
        f"{len(grid)} points over {len(geometry)} frames "
        f"({len(train_idx)} train / {len(test_idx)} test), "
        f"gate={'regate ' + reference if cfg['regate'] else 'on' if gate else 'off'}"
    )

    records, failures = [], []
    if workers == 1 or len(grid) == 1:
        for hypers in tqdm(grid, desc="sweep", disable=not progress):
            try:
                records.append(evaluate_point(hypers.to_dict(), cfg))
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
                pool.submit(evaluate_point, hypers.to_dict(), cfg): hypers.key
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
        # The skip path returns a stored gate verbatim, so without this a
        # changed --reference would silently produce a comparison.csv that
        # mixes references and looks clean.
        stale = [
            r["key"]
            for r in records
            if r.get("skipped") and (r.get("gate") or {}).get("reference") != reference
        ]
        if stale:
            print(
                f"WARNING: {len(stale)} skipped points were gated against another "
                f"reference; comparison.csv mixes references. Refresh with --regate."
            )

    if records:
        write_comparison(out_dir / COMPARISON_NAME, records)
        verdicts = [(r.get("gate") or {}).get("improves_on_baseline") for r in records]
        scored = [v for v in verdicts if v is not None]
        if scored:
            print(f"{sum(scored)}/{len(scored)} points improve on the {reference} baseline")
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
    parser.add_argument("--split-seed", type=int, default=0, help="Held-out partition; pin it across sweeps.")
    parser.add_argument("--cv-seed", type=int, default=0, help="RidgeCV fold assignment.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--no-gate",
        dest="gate",
        action="store_false",
        help="Skip the dimer-well benchmark (~12 s/point).",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        choices=list(REFERENCES),
        help="Ground truth for the dimer probe. 'mp2' is a diagnostic only — "
        "GBQIII was fitted to those rows.",
    )
    parser.add_argument(
        "--regate",
        action="store_true",
        help="Recompute the gate of already-finished points from their saved "
        "model instead of returning the stored one. This is how a sweep is "
        "re-scored against a new --reference; it refits nothing (~12 s/point).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-parse the campaign, ignoring the cache.")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    return main(
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
