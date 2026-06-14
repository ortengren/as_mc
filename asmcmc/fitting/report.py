"""Evaluation metrics and artifacts for a fitted GBQ potential.

Fit quality is judged by *plain* (unweighted) regression metrics on a held-out
test split -- reported both overall and for the bound subset (target < 0), since
~32% of frames are repulsive and an unweighted error would otherwise be
dominated by the wall. Diagnostic plots build on these metrics (later slice).
"""

import json
import os

import numpy as np

from asmcmc.fitting.fit import predict_per_mol, PARAM_NAMES

# Unit annotation per parameter, so the serialised values are self-describing.
# Q enters the potential only as Q^2, so its natural unit is (energy*length^5)^1/2.
PARAM_UNITS = {
    "sigma0": "Angstrom",
    "eps0": "eV",
    "kappa": "dimensionless",
    "kappa_prime": "dimensionless",
    "mu": "dimensionless",
    "nu": "dimensionless",
    "Q": "(eV*Angstrom^5)^0.5",
    "E_intra": "eV/molecule",
}

# Benzene sublimation enthalpy ~ 44 kJ/mol -> eV/molecule; a rough magnitude the
# deepest per-molecule lattice energy should be in the ballpark of.
KJ_PER_MOL_IN_EV = 0.01036410
BENZENE_SUBLIMATION_EV_PER_MOL = -44.0 * KJ_PER_MOL_IN_EV  # ~ -0.456


def regression_metrics(pred, target):
    """RMSE / MAE / max|err| / R^2 / count for one (pred, target) pair (eV/mol).

    All scalars are plain Python floats so the result is JSON-serialisable. R^2
    is NaN when the target variance is zero or the subset is empty, since it is
    then undefined rather than 0 or 1.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    n = int(target.size)
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "max_abs_err": float("nan"), "r2": float("nan")}

    err = pred - target
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def evaluate_fit(theta, data, train_idx=None, test_idx=None, pred=None):
    """Overall regression metrics for a fitted ``theta``, keyed by partition.

    Predictions are sliced per partition: ``train``/``test`` when a split is
    supplied, otherwise a single ``all`` key. Pass ``pred`` (per-frame
    predictions, length ``n_frames``) to reuse predictions a caller already has
    -- e.g. a plot routine -- and skip recomputing ``predict_per_mol``; in that
    case ``theta`` is unused and may be ``None``.
    """
    if pred is None:
        pred = predict_per_mol(theta, data)
    target = data.target_per_mol

    if train_idx is None and test_idx is None:
        partitions = {"all": np.arange(data.n_frames)}
    else:
        partitions = {}
        if train_idx is not None:
            partitions["train"] = np.asarray(train_idx)
        if test_idx is not None:
            partitions["test"] = np.asarray(test_idx)

    return {k: regression_metrics(pred[i], target[i]) for k, i in partitions.items()}


def params_to_dict(theta):
    """Map a fitted ``theta`` to ``{name: {'value': float, 'unit': str}}``.

    Pairs each value with its unit (from ``PARAM_UNITS``) so ``params.json`` is
    self-describing; values are cast to plain floats for JSON serialisation.
    Order follows ``PARAM_NAMES``.
    """
    return {
        name: {"value": float(v), "unit": PARAM_UNITS[name]}
        for name, v in zip(PARAM_NAMES, theta)
    }


def sanity_checks(theta, data, pred=None):
    """Physical reasonableness checks for a fitted ``theta`` (eV/molecule).

    - ``E_intra``: the fitted offset, an estimate of the isolated-molecule
      reference; should land near the mean target (the ~-1601 eV pedestal).
    - ``min_lattice_energy``: the deepest per-molecule lattice energy,
      ``min(pred - E_intra)`` -- compare with benzene's sublimation enthalpy.

    ``pred`` is reused if supplied (else recomputed from ``theta``).
    """
    if pred is None:
        pred = predict_per_mol(theta, data)
    e_intra = float(theta[PARAM_NAMES.index("E_intra")])
    lattice = pred - e_intra
    return {
        "E_intra_eV_per_mol": e_intra,
        "mean_target_eV_per_mol": float(data.target_per_mol.mean()),
        "min_lattice_energy_eV_per_mol": float(lattice.min()),
        "benzene_sublimation_ref_eV_per_mol": BENZENE_SUBLIMATION_EV_PER_MOL,
    }


def format_report(params, metrics, sanity, meta=None):
    """Render ``params`` + ``metrics`` + ``sanity`` as a Markdown report string.

    ``params`` is from :func:`params_to_dict`, ``metrics`` from
    :func:`evaluate_fit` (one row per partition), ``sanity`` from
    :func:`sanity_checks`. ``meta`` is an optional ``{label: value}`` dict for
    run provenance (seed, dataset, cutoff, DE settings, ...).
    """
    lines = ["# GBQ fit report", ""]

    if meta:
        lines.append("## Run")
        lines += ["- **{}**: {}".format(k, v) for k, v in meta.items()]
        lines.append("")

    lines.append("## Parameters")
    lines.append("| parameter | value | unit |")
    lines.append("| --- | --- | --- |")
    for name in PARAM_NAMES:
        p = params[name]
        lines.append("| {} | {:.6g} | {} |".format(name, p["value"], p["unit"]))
    lines.append("")

    cols = ("n", "rmse", "mae", "max_abs_err", "r2")
    lines.append("## Metrics (eV/molecule)")
    lines.append("| partition | " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * (len(cols) + 1)) + " |")
    for partition, m in metrics.items():
        cells = [str(m["n"])] + ["{:.4g}".format(m[c]) for c in cols[1:]]
        lines.append("| {} | ".format(partition) + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Sanity checks (eV/molecule)")
    lines.append("- inferred E_intra (isolated-molecule ref): "
                 "{:.4f}".format(sanity["E_intra_eV_per_mol"]))
    lines.append("- mean target: {:.4f}".format(sanity["mean_target_eV_per_mol"]))
    lines.append("- min lattice energy (pred - E_intra): {:.4f} "
                 "(benzene sublimation ref ~ {:.3f})".format(
                     sanity["min_lattice_energy_eV_per_mol"],
                     sanity["benzene_sublimation_ref_eV_per_mol"]))
    lines.append("")
    return "\n".join(lines)


def write_artifacts(out_dir, theta, data, train_idx=None, test_idx=None, meta=None):
    """Write ``params.json``, ``metrics.json`` and ``fit_report.md`` to ``out_dir``.

    Predicts once and threads the predictions through metrics + sanity. The
    ``metrics.json`` carries both the per-partition metrics and the ``sanity``
    block. Returns the in-memory ``{'params', 'metrics', 'sanity'}`` so callers
    can use them without re-reading the files.
    """
    os.makedirs(out_dir, exist_ok=True)
    pred = predict_per_mol(theta, data)

    params = params_to_dict(theta)
    metrics = evaluate_fit(theta, data, train_idx, test_idx, pred=pred)
    sanity = sanity_checks(theta, data, pred=pred)

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "sanity": sanity}, f, indent=2)
    with open(os.path.join(out_dir, "fit_report.md"), "w") as f:
        f.write(format_report(params, metrics, sanity, meta=meta))

    return {"params": params, "metrics": metrics, "sanity": sanity}
