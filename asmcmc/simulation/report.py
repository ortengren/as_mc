"""Observables, run provenance, and artifacts for one MC (T, P) state point.

The analogue of :mod:`asmcmc.fitting.report` for the sampler: it reads a finished
``simulation.db`` back through the :mod:`asmcmc.measurements` framework and emits
a self-describing artifact set -- ``run_config.json`` (what was run),
``observables.json`` (scalar results) + ``observables.npz`` (the RDF/OCF arrays),
and a human-readable ``report.md``.
"""

import json
import os

import numpy as np

from asmcmc.measurements import (
    AverageEnergy,
    HeatCapacity,
    NematicOrderParameter,
    OrientationalCorrelationFunction,
    RadialDistributionFunction,
    TrajectoryAnalyzer,
)

BOLTZCONST = 8.617e-5  # eV / K

# Unit annotation per scalar observable so observables.json is self-describing.
OBSERVABLE_UNITS = {
    "avg_energy": "eV",
    "var_energy": "eV^2",
    "energy_per_particle": "eV",
    "heat_capacity": "eV/K",
    "heat_capacity_per_kB": "dimensionless",
    "nematic_S": "dimensionless",
    "nematic_S_std": "dimensionless",
    "nematic_S_lab": "dimensionless",
    "mean_volume": "Angstrom^3",
    "volume_per_particle": "Angstrom^3",
    "number_density": "1/Angstrom^3",
}


def _safe_mean(decisions):
    """Mean of a 0/1 decision list, or NaN when it is empty."""
    return float(np.mean(decisions)) if len(decisions) else float("nan")


def compute_observables(
    db_path,
    temp,
    n_particles,
    nl_radius,
    potential,
    recompute=True,
    rdf_r_max=15.0,
    rdf_bins=100,
    progress=True,
):
    """Run the measurement framework over ``db_path`` and collect the results.

    Wires the production trajectory through ``AverageEnergy`` (``recompute`` re-
    evaluates each frame's energy from scratch to dodge incremental-tracker
    drift), ``HeatCapacity``, ``NematicOrderParameter`` and the RDF/OCF pair
    functions. Returns a flat dict mixing scalar observables with the RDF/OCF
    arrays (``rdf_r``/``rdf_g``/``ocf_r``/``ocf_s2``); :func:`write_artifacts`
    splits the scalars (JSON) from the arrays (NPZ).
    """
    analyzer = TrajectoryAnalyzer(db_path)
    analyzer.add_measurement(
        "energy",
        AverageEnergy(recompute=recompute, nl_radius=nl_radius, potential=potential),
    )
    analyzer.add_measurement("heat_capacity", HeatCapacity(temp, n_particles))
    analyzer.add_measurement("nematic", NematicOrderParameter())
    analyzer.add_measurement("rdf", RadialDistributionFunction(rdf_r_max, rdf_bins))
    analyzer.add_measurement(
        "ocf", OrientationalCorrelationFunction(rdf_r_max, rdf_bins)
    )
    results = analyzer.run_analysis(progress=progress)

    avg_e, var_e = results["energy"]
    cv = results["heat_capacity"]
    nem = results["nematic"]
    rdf = results["rdf"]
    ocf = results["ocf"]

    return {
        "avg_energy": float(avg_e),
        "var_energy": float(var_e),
        "energy_per_particle": float(avg_e / n_particles),
        "heat_capacity": float(cv),
        "heat_capacity_per_kB": float(cv / BOLTZCONST),
        "nematic_S": float(nem["S"]),
        "nematic_S_std": float(nem["S_std"]),
        "nematic_S_lab": float(nem["S_lab"]),
        "director": [float(x) for x in nem["director"]],
        "rdf_r": np.asarray(rdf["r"]),
        "rdf_g": np.asarray(rdf["g_r"]),
        "ocf_r": np.asarray(ocf["r"]),
        "ocf_s2": np.asarray(ocf["s2_r"]),
    }


def _trajectory_volume(db_path):
    """Mean cell volume over the stored frames (Angstrom^3), or NaN if empty."""
    from ase.db import connect

    vols = []
    with connect(db_path) as db:
        for row in db.select():
            vols.append(row.key_value_pairs.get("vol", float("nan")))
    return float(np.mean(vols)) if vols else float("nan")


def run_config(metro, db_path, meta=None):
    """Provenance + configuration dict for a finished run.

    Pulls the static settings off the ``MetropolisCalculator`` (ensemble,
    neighbour-list, potential name, final adapted trial-move deltas) and the
    production acceptance rates off its ``*_decisions`` lists; ``meta`` supplies
    what the object does not retain (temperatures/pressures as given, step
    budgets, seed, initial deltas). Volume/density are taken from the trajectory.
    """
    meta = dict(meta or {})
    n_particles = len(metro.current_frame)
    temp = 1.0 / (metro.beta * BOLTZCONST)
    mean_vol = _trajectory_volume(db_path)

    config = {
        "temp": float(temp),
        "pressure": float(metro.pressure),
        "ensemble": "npt" if metro.npt_ensemble else "nvt",
        "n_particles": int(n_particles),
        "potential": metro.potential.name,
        "nl_radius": float(metro.nl_cutoffs[0]),
        "nl_skin": float(metro.nl_skin),
        "acceptance": {
            "position": _safe_mean(metro.pos_decisions),
            "orientation": _safe_mean(metro.or_decisions),
            "volume": _safe_mean(metro.vol_decisions),
        },
        "final_deltas": {
            "pos": float(metro.pos_delt),
            "or": float(metro.or_delt),
            "vol": float(metro.vol_delt),
        },
        "mean_volume": mean_vol,
        "number_density": float(n_particles / mean_vol) if mean_vol > 0 else float("nan"),
    }
    config.update(meta)
    return config


def format_report(config, observables):
    """Render ``config`` + ``observables`` as a Markdown report string.

    Mirrors :func:`asmcmc.fitting.report.format_report`: a Run section (the
    config), an Acceptance section, and an Observables table with units.
    """
    lines = ["# MC run report", "", "## Run"]
    scalar_keys = [
        "temp", "pressure", "ensemble", "n_particles", "potential",
        "n_steps", "block_size", "num_eq_steps", "seed",
        "nl_radius", "nl_skin", "mean_volume", "number_density",
    ]
    for k in scalar_keys:
        if k in config:
            lines.append("- **{}**: {}".format(k, config[k]))
    lines.append("")

    acc = config.get("acceptance", {})
    deltas = config.get("final_deltas", {})
    lines.append("## Acceptance (production)")
    lines.append("| move | acc. rate | final delta |")
    lines.append("| --- | --- | --- |")
    for move, dkey in (("position", "pos"), ("orientation", "or"), ("volume", "vol")):
        lines.append(
            "| {} | {:.3f} | {:.4g} |".format(
                move, acc.get(move, float("nan")), deltas.get(dkey, float("nan"))
            )
        )
    lines.append("")

    lines.append("## Observables")
    lines.append("| observable | value | unit |")
    lines.append("| --- | --- | --- |")
    for name, unit in OBSERVABLE_UNITS.items():
        if name in observables:
            lines.append("| {} | {:.6g} | {} |".format(name, observables[name], unit))
    lines.append("")
    return "\n".join(lines)


def write_artifacts(out_dir, metro, db_path, meta=None, progress=True):
    """Write ``run_config.json``, ``observables.json``/``.npz`` and ``report.md``.

    The MC analogue of :func:`asmcmc.fitting.report.write_artifacts`: computes the
    observables once, splits scalars (JSON, with units) from the RDF/OCF arrays
    (NPZ), and renders the Markdown report. Returns the in-memory
    ``{'config', 'observables'}`` so callers (and plots) can reuse them without
    re-reading the files.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta or {})

    config = run_config(metro, db_path, meta=meta)
    observables = compute_observables(
        db_path,
        temp=config["temp"],
        n_particles=config["n_particles"],
        nl_radius=config["nl_radius"],
        potential=metro.potential,
        recompute=meta.get("recompute_energy", True),
        rdf_r_max=meta.get("rdf_r_max", 15.0),
        rdf_bins=meta.get("rdf_bins", 100),
        progress=progress,
    )

    array_keys = ("rdf_r", "rdf_g", "ocf_r", "ocf_s2")
    scalar_obs = {
        k: v for k, v in observables.items() if k not in array_keys
    }

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "observables.json"), "w") as f:
        json.dump(scalar_obs, f, indent=2)
    np.savez(
        os.path.join(out_dir, "observables.npz"),
        **{k: observables[k] for k in array_keys},
    )
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(format_report(config, scalar_obs))

    return {"config": config, "observables": observables}


def aggregate_observables(rep_observables):
    """Mean/std/sem across replicas for each scalar observable.

    ``rep_observables`` is a list of per-replica observables dicts (the flat
    dicts returned by :func:`compute_observables`). Only the scalar keys in
    :data:`OBSERVABLE_UNITS` are aggregated, and only over their finite values.
    Returns ``{key: {mean, std, sem, n, unit}}`` -- ``std`` is the sample
    standard deviation (``ddof=1``) and ``sem = std / sqrt(n)``, the spread that
    quantifies the run-to-run uncertainty at this state point.
    """
    agg = {}
    for key in OBSERVABLE_UNITS:
        vals = [
            o[key]
            for o in rep_observables
            if key in o and np.isfinite(o[key])
        ]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        agg[key] = {
            "mean": float(np.mean(arr)),
            "std": std,
            "sem": std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "unit": OBSERVABLE_UNITS[key],
        }
    return agg


def format_summary(temp, pressure, agg, n_replicas):
    """Render the replica aggregate as a Markdown summary string."""
    lines = [
        "# MC replica summary",
        "",
        "- **temp**: {}".format(temp),
        "- **pressure**: {}".format(pressure),
        "- **n_replicas**: {}".format(n_replicas),
        "",
        "## Observables (mean +/- std over replicas)",
        "| observable | mean | std | sem | n | unit |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, s in agg.items():
        lines.append(
            "| {} | {:.6g} | {:.3g} | {:.3g} | {} | {} |".format(
                key, s["mean"], s["std"], s["sem"], s["n"], s["unit"]
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_summary(out_dir, temp, pressure, rep_results):
    """Aggregate replica artifacts and write ``summary.json`` + ``summary.md``.

    ``rep_results`` is the list of ``{'config', 'observables'}`` dicts returned
    by :func:`write_artifacts` for each replica of one (T, P) point. Writes the
    cross-replica mean/std/sem to ``out_dir`` (the point dir, above the per-
    replica ``rep*/`` dirs) and returns the in-memory summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate_observables([r["observables"] for r in rep_results])
    summary = {
        "temp": float(temp),
        "pressure": float(pressure),
        "n_replicas": len(rep_results),
        "seeds": [r["config"].get("seed") for r in rep_results],
        "observables": agg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(format_summary(temp, pressure, agg, len(rep_results)))
    return summary
