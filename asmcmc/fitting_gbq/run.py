"""Runner / CLI for the GBQ fit: data -> fit -> artifacts + plots.

Ties together :mod:`asmcmc.fitting_gbq.data` (dataset build), :mod:`asmcmc.fitting_gbq.fit`
(``differential_evolution`` global fit), :mod:`asmcmc.fitting_gbq.report` (metrics +
JSON + markdown) and :mod:`asmcmc.fitting_gbq.plots` (diagnostic PNGs).

Two weighting variants are fit on the *same* train/test split so they are
directly comparable: ``uniform`` (equal weight on every frame, the production
objective) and ``boltzmann`` (near-equilibrium-weighted, kept as a reference).
Each writes to its own subdirectory of the output root.

Run with ``python -m asmcmc.fitting_gbq.run``.
"""

import argparse
import os

import numpy as np

from asmcmc.fitting_gbq.data import build_dataset
from asmcmc.fitting_gbq.fit import (
    DEFAULT_ALPHA,
    boltzmann_weights,
    run_fit,
    train_test_split,
)
from asmcmc.fitting_gbq.report import write_artifacts
from asmcmc.fitting_gbq.plots import write_plots

DEFAULT_DATA = "data/xyz_files/ellipsoids_with_axes_and_energies.xyz"
# Lattice-sum cutoff (Angstrom); matches the MC neighbour radius (nl_radius=15).
DEFAULT_CUTOFF = 15.0
DEFAULT_OUT = "results/fit_gb"


def fit_variant(
    data,
    train_idx,
    test_idx,
    out_dir,
    weighting="boltzmann",
    alpha=DEFAULT_ALPHA,
    seed=0,
    workers=1,
    progress=True,
    dataset_path=None,
    **de_kwargs,
):
    """Fit one weighting variant on ``train_idx`` and write its artifacts + plots.

    ``weighting`` selects the per-frame weights: ``"boltzmann"`` uses
    :func:`boltzmann_weights` (near-equilibrium emphasis), ``"uniform"`` weights
    every frame equally. The fit only ever sees ``train_idx`` (``run_fit``
    renormalises the weights within it and ``default_bounds`` centres ``E_intra``
    on the train mean); ``test_idx`` is held out for the metrics. ``dataset_path``
    is recorded in the report provenance only. ``de_kwargs`` forward to
    ``differential_evolution`` (``maxiter``, ``popsize``, ...).

    Returns ``(theta, result, artifacts)``: the fitted parameter vector, the raw
    ``OptimizeResult``, and the in-memory ``{params, metrics, sanity}`` dicts.
    """
    if weighting == "boltzmann":
        weights = boltzmann_weights(data.target_per_mol, alpha)
    elif weighting == "uniform":
        weights = np.ones(data.n_frames)
    else:
        raise ValueError("unknown weighting {!r}".format(weighting))

    result = run_fit(
        data,
        weights=weights,
        idx=train_idx,
        seed=seed,
        workers=workers,
        progress=progress,
        **de_kwargs,
    )
    theta = result.x

    meta = {
        "dataset": dataset_path if dataset_path is not None else "?",
        "weighting": weighting,
        "cutoff_A": data.cutoff,
        "n_frames": data.n_frames,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "seed": seed,
        "objective": float(result.fun),
    }
    artifacts = write_artifacts(out_dir, theta, data, train_idx, test_idx, meta=meta)
    write_plots(out_dir, theta, data, train_idx, test_idx)
    return theta, result, artifacts


def main(
    dataset_path=DEFAULT_DATA,
    cutoff=DEFAULT_CUTOFF,
    out_dir=DEFAULT_OUT,
    weightings=("boltzmann", "uniform"),
    test_frac=0.2,
    split_seed=0,
    fit_seed=0,
    workers=1,
    progress=True,
    index=":",
    alpha=DEFAULT_ALPHA,
    cache_dir=None,
    **de_kwargs,
):
    """Fit every weighting variant on one shared split and write a comparison.

    Builds the dataset once (cached under ``cache_dir``, default ``out_dir/cache``;
    point several runs at one shared dir to skip the per-run neighbour-list
    rebuild -- the cache is keyed by (file, cutoff, mtime), independent of
    weighting/alpha/seed), makes a single
    ``train_test_split`` so all variants are compared on identical held-out
    frames, then fits each into ``out_dir/<weighting>/``. ``index`` is forwarded
    to :func:`build_dataset` (use a slice like ``":200"`` for a fast test run);
    ``alpha`` sets the Boltzmann weight scale (ignored by the ``uniform``
    variant); ``de_kwargs`` tune ``differential_evolution``. Writes a
    ``comparison.csv`` (one row per variant: test RMSE/MAE/R² + objective) and
    returns the ``{weighting: (theta, result, artifacts)}`` map.
    """
    if cache_dir is None:
        cache_dir = os.path.join(out_dir, "cache")
    data = build_dataset(dataset_path, cutoff, index=index, cache_dir=cache_dir)
    train_idx, test_idx = train_test_split(data.n_frames, test_frac, split_seed)

    results = {}
    for weighting in weightings:
        sub = os.path.join(out_dir, weighting)
        print("Fitting {!r} -> {}".format(weighting, sub))
        results[weighting] = fit_variant(
            data,
            train_idx,
            test_idx,
            sub,
            weighting=weighting,
            alpha=alpha,
            seed=fit_seed,
            workers=workers,
            progress=progress,
            dataset_path=dataset_path,
            **de_kwargs,
        )

    _write_comparison(os.path.join(out_dir, "comparison.csv"), results)
    return results


def _write_comparison(path, results):
    """Write one row per variant: test-set RMSE/MAE/R^2 + DE objective.

    Pulls the held-out ``test`` metrics from each variant's in-memory artifacts
    so the variants line up in a single CSV for a direct read-off. Falls back to
    the ``all`` partition if a variant was fit without a split.
    """
    import csv

    fields = ["weighting", "test_rmse", "test_mae", "test_r2", "test_n", "objective"]
    rows = []
    for weighting, (theta, result, artifacts) in results.items():
        metrics = artifacts["metrics"]
        m = metrics.get("test", metrics.get("all"))
        rows.append(
            {
                "weighting": weighting,
                "test_rmse": m["rmse"],
                "test_mae": m["mae"],
                "test_r2": m["r2"],
                "test_n": m["n"],
                "objective": float(result.fun),
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {}".format(path))


def build_parser():
    """Argparse front-end mapping CLI flags onto :func:`main`.

    Two flag groups: run/split settings (dataset, cutoff, out dir, weightings,
    seeds, ...) map straight to ``main`` kwargs; the DE knobs (``--maxiter``,
    ``--popsize``, ``--tol``, ``--workers``) are forwarded as ``de_kwargs``.
    ``--weighting`` is repeatable (``--weighting boltzmann --weighting uniform``)
    and defaults to both. ``-1`` workers uses every core; ``--no-progress``
    silences the tqdm bar (e.g. for non-interactive logs).
    """
    p = argparse.ArgumentParser(
        prog="python -m asmcmc.fitting_gbq.run",
        description="Fit the GB+quadrupole potential and write artifacts + plots.",
    )
    p.add_argument(
        "--dataset", default=DEFAULT_DATA, help="training xyz (default: %(default)s)"
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="lattice-sum cutoff in Angstrom (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT, help="output root (default: %(default)s)"
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="shared built-dataset cache dir (default: <out-dir>/cache). Point "
        "several runs at one dir to skip the per-run neighbour-list rebuild.",
    )
    p.add_argument(
        "--weighting",
        dest="weightings",
        action="append",
        choices=["boltzmann", "uniform"],
        help="weighting variant; repeatable (default: both)",
    )
    p.add_argument(
        "--index",
        default=":",
        help="ase.io.read frame slice, e.g. ':200' for a smoke run "
        "(default: all frames)",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="held-out fraction (default: %(default)s)",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="train/test split seed (default: %(default)s)",
    )
    p.add_argument(
        "--fit-seed",
        type=int,
        default=0,
        help="differential_evolution seed (default: %(default)s)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Boltzmann weight scale (1/eV); alpha = 1/(k_B*T), so "
        "~5.8/4.6/3.9/3.3/2.9 ~ 2000/2500/3000/3500/4000 K. "
        "Ignored by the uniform variant (the production fit) "
        "(default: %(default).4g ~ 4000 K, a reference sweep point)",
    )
    p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="silence the tqdm progress bar",
    )

    de = p.add_argument_group("differential_evolution knobs")
    de.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers; -1 uses every core (default: %(default)s)",
    )
    de.add_argument(
        "--maxiter", type=int, help="max DE generations (scipy default 1000)"
    )
    de.add_argument(
        "--popsize", type=int, help="population multiplier (scipy default 15)"
    )
    de.add_argument(
        "--tol", type=float, help="relative convergence tolerance (scipy default 0.01)"
    )
    return p


def cli(argv=None):
    """Parse ``argv`` (default ``sys.argv``) and dispatch to :func:`main`.

    Translates the parsed namespace into ``main`` kwargs: the DE knobs are
    collected into ``de_kwargs`` with unset (``None``) ones dropped so SciPy's
    own defaults apply, and an empty ``--weighting`` list falls back to both
    variants (``append`` defaults to ``None``, not the tuple). Returns whatever
    :func:`main` returns.
    """
    args = build_parser().parse_args(argv)

    weightings = tuple(args.weightings) if args.weightings else ("boltzmann", "uniform")
    de_kwargs = {
        k: getattr(args, k)
        for k in ("maxiter", "popsize", "tol")
        if getattr(args, k) is not None
    }

    return main(
        dataset_path=args.dataset,
        cutoff=args.cutoff,
        out_dir=args.out_dir,
        weightings=weightings,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        fit_seed=args.fit_seed,
        workers=args.workers,
        progress=args.progress,
        index=args.index,
        alpha=args.alpha,
        cache_dir=args.cache_dir,
        **de_kwargs,
    )


if __name__ == "__main__":
    cli()
