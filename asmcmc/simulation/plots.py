"""Diagnostic plots for one finished MC (T, P) state point.

The sampler analogue of :mod:`asmcmc.fitting.plots`: a ``write_plots`` aggregator
that writes a fixed set of PNGs with the Agg backend (batch reporting, never an
interactive window). Trajectory traces are read straight from the run's
``simulation.db``; the RDF/OCF curves come from the ``observables`` dict that
:func:`asmcmc.simulation.report.compute_observables` already produced.
"""

import os

import matplotlib

matplotlib.use("Agg")  # batch reporting: write figures to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect

from asmcmc.measurements import nematic_q_tensor


def _read_trajectory(db_path):
    """Read per-block traces out of ``db_path`` in stored order.

    Returns ``step`` plus the per-block ``total_energy`` and pos/or/vol
    acceptance rates (scalar data), and the per-frame nematic order parameter
    ``S`` (largest eigenvalue of each frame's Q-tensor, from the stored
    ``or_vec``). Everything is a NumPy array of equal length.
    """
    step, energy, vol = [], [], []
    pos_acc, or_acc, vol_acc, s_vals = [], [], [], []
    with connect(db_path) as db:
        for row in db.select():
            kv = row.key_value_pairs
            step.append(kv.get("step", np.nan))
            energy.append(kv.get("total_energy", np.nan))
            vol.append(kv.get("vol", np.nan))
            pos_acc.append(kv.get("pos_acc_rate", np.nan))
            or_acc.append(kv.get("or_acc_rate", np.nan))
            vol_acc.append(kv.get("vol_acc_rate", np.nan))
            q = nematic_q_tensor(np.asarray(row.data["or_vec"]))
            s_vals.append(np.linalg.eigvalsh(q)[-1])
    return {
        "step": np.asarray(step, dtype=float),
        "energy": np.asarray(energy, dtype=float),
        "vol": np.asarray(vol, dtype=float),
        "pos_acc": np.asarray(pos_acc, dtype=float),
        "or_acc": np.asarray(or_acc, dtype=float),
        "vol_acc": np.asarray(vol_acc, dtype=float),
        "S": np.asarray(s_vals, dtype=float),
    }


def energy_trace(traj, path=None):
    """Total energy vs MC step over the production trajectory."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(traj["step"], traj["energy"], marker=".", lw=1)
    ax.set_xlabel("MC step")
    ax.set_ylabel("total energy (eV)")
    ax.set_title("Energy trace")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def acceptance_trace(traj, path=None):
    """Per-block acceptance rate vs MC step for each move type."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(traj["step"], traj["pos_acc"], marker=".", lw=1, label="position")
    ax.plot(traj["step"], traj["or_acc"], marker=".", lw=1, label="orientation")
    if np.any(np.isfinite(traj["vol_acc"])):
        ax.plot(traj["step"], traj["vol_acc"], marker=".", lw=1, label="volume")
    ax.axhline(0.275, color="k", ls="--", lw=1, label="target")
    ax.set_ylim(0, 1)
    ax.set_xlabel("MC step")
    ax.set_ylabel("acceptance rate")
    ax.set_title("Acceptance trace")
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def nematic_trace(traj, path=None):
    """Per-frame nematic order parameter S vs MC step."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(traj["step"], traj["S"], marker=".", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("MC step")
    ax.set_ylabel("nematic order parameter S")
    ax.set_title("Nematic order trace")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def volume_trace(traj, path=None):
    """Cell volume vs MC step -- the equilibration plateau check for NPT runs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(traj["step"], traj["vol"], marker=".", lw=1)
    ax.set_xlabel("MC step")
    ax.set_ylabel("cell volume (Angstrom^3)")
    ax.set_title("Volume trace")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def pair_function(r, y, ylabel, title, path=None):
    """Generic r-vs-y line plot (used for the RDF g(r) and the OCF s2(r))."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r, y, lw=1)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def _trace_specs(traj):
    """The per-block trajectory traces (energy/volume/acceptance/nematic).

    Shared by :func:`write_plots` (full report) and :func:`write_trace_plots`
    (equilibration check) so both render identical traces from a db.
    """
    return {
        "energy_trace": energy_trace(traj),
        "volume_trace": volume_trace(traj),
        "acceptance_trace": acceptance_trace(traj),
        "nematic_order": nematic_trace(traj),
    }


def _save_specs(out_dir, specs):
    """Save each ``{name: Figure}`` to ``out_dir/name.png`` and close it."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, fig in specs.items():
        path = os.path.join(out_dir, name + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = path
    return paths


def write_trace_plots(out_dir, db_path):
    """Write only the trajectory-trace PNGs from any db (no ``observables`` needed).

    Unlike :func:`write_plots`, this skips the RDF/OCF pair functions, so it works
    on an ``equilibration.db`` -- render its ``volume_trace.png`` to confirm the
    cell volume has plateaued before committing to a production run. Returns
    ``{name: path}`` for the files written.
    """
    return _save_specs(out_dir, _trace_specs(_read_trajectory(db_path)))


def write_plots(out_dir, db_path, observables):
    """Write all diagnostic PNGs for one state point to ``out_dir``.

    The plotting analogue of :func:`asmcmc.simulation.report.write_artifacts`:
    reads the db traces once, reuses the ``observables`` arrays for the pair
    functions, and closes each Figure after saving so a batch run does not
    accumulate open figures. Returns ``{name: path}`` for the files written.
    """
    traj = _read_trajectory(db_path)
    specs = _trace_specs(traj)
    specs["rdf"] = pair_function(
        observables["rdf_r"], observables["rdf_g"], "g(r)", "Radial distribution"
    )
    specs["orientational_correlation"] = pair_function(
        observables["ocf_r"],
        observables["ocf_s2"],
        "s2(r)",
        "Orientational correlation",
    )
    return _save_specs(out_dir, specs)


def _cli(argv=None):
    """``python -m asmcmc.simulation.plots <db> [--out-dir DIR]`` -> trace PNGs.

    Renders the trajectory traces (energy/volume/acceptance/nematic) for any db,
    e.g. an ``equilibration.db``, so the volume plateau can be eyeballed before
    starting production. Defaults to writing alongside the db.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m asmcmc.simulation.plots",
        description="Render trajectory-trace PNGs (energy/volume/acceptance/nematic) from a db.",
    )
    p.add_argument("db_path", help="path to an equilibration.db or simulation.db")
    p.add_argument(
        "--out-dir",
        default=None,
        help="output dir for the PNGs (default: the db's directory)",
    )
    args = p.parse_args(argv)
    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.db_path)))
    paths = write_trace_plots(out_dir, args.db_path)
    for name, path in paths.items():
        print("wrote {}".format(path))
    return paths


if __name__ == "__main__":
    _cli()
