"""Diagnostic figures for one run directory, drawn from its db.

:func:`render` writes four figures:

    structure.png    g(r) and orientational correlation at the end of the run (which phase?)
    phase.png        nematic order S and density against cycle (when did it settle?)
    acceptance.png   position, orientation and volume acceptance against cycle
    energy.png       energy per particle against cycle (has it converged?)

File names start with the db's name (``equilibration_energy.png``,
``simulation_energy.png``), so production figures never overwrite the
equilibration ones. ``scripts/plot_run.py`` and ``scripts/export_xyz.py`` are the
command-line wrappers.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import ase.io
from ase.db import connect
import matplotlib

matplotlib.use("Agg")  # write figures to files; never open a window
import matplotlib.pyplot as plt

from asmcmc.mc.measurements import (
    OrientationalCorrelationFunction,
    RadialDistributionFunction,
    nematic_q_tensor,
)
from asmcmc.mc.metropolis import TARGET_ACC_RATE

# RDF and OCF are averaged over the last TAIL_FRACTION of the run, so the curves
# describe where it ended up. Each frame costs a full distance matrix (~0.2 s at
# N = 400), so at most TAIL_MAX_FRAMES frames, spread evenly over the tail, are used.
TAIL_FRACTION = 0.1
TAIL_MAX_FRAMES = 40

R_MAX = 12  # stays below half the (NPT-fluctuating) box; RDF/OCF skip wider bins
NUM_BINS = 120

# One colour per move type, from a colour-blind-safe categorical palette.
SERIES = {"pos": "#2a78d6", "or": "#eb6834", "vol": "#1baf7a"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"


@dataclass
class RunTrace:
    """One run directory's recorded blocks, reduced to arrays that can be plotted.

    The ``or_vec`` arrays aren't kept. A long equilibration has one (N, 3) array
    per recorded block, which would take hundreds of MB, and the plots only need
    one number per frame. :func:`load_run` reduces each frame to its nematic S as
    it reads, then drops the array.
    """

    run_dir: str
    db_name: str
    n_particles: int
    cycles: np.ndarray
    steps: np.ndarray
    energy: np.ndarray  # total, eV
    volume: np.ndarray  # A^3
    density: np.ndarray  # N / V, A^-3
    nematic_s: np.ndarray
    pos_acc: np.ndarray
    or_acc: np.ndarray
    vol_acc: np.ndarray
    tail_frames: int
    rdf: dict = field(default_factory=dict)  # {"r", "g_r"}
    ocf: dict = field(default_factory=dict)  # {"r", "s2_r"}

    @property
    def label(self):
        return f"{os.path.relpath(self.run_dir)}  [{self.db_name}]"


def load_run(run_dir, db_name="equilibration.db", r_max=R_MAX, num_bins=NUM_BINS):
    """Read a run directory's db once and return a :class:`RunTrace`.

    The db is read only once because a long equilibration db is slow to read, and
    otherwise each of the four plots would read it again. Per-frame scalars are
    collected for every block, and the frames in the last ``TAIL_FRACTION`` of the
    run are also passed to the RDF and OCF accumulators through their per-frame
    ``compute()`` method. ``TrajectoryAnalyzer`` isn't used because it always
    reads the whole db, whereas the tail is meant to show where the run ended up.

    Rows are read in the order they were written, which is step order because a
    resumed run appends to the db. If the steps aren't increasing, an error is
    raised rather than the rows being silently re-sorted.
    """
    path = os.path.join(run_dir, db_name)
    with connect(path) as db:
        total = db.count()
        if total == 0:
            raise ValueError(f"{path} has no recorded frames")

        window = max(1, math.ceil(TAIL_FRACTION * total))
        tail_start = total - window
        # Evenly spaced samples across the tail window, capped for cost.
        sampled = set(
            np.unique(
                np.linspace(tail_start, total - 1, min(window, TAIL_MAX_FRAMES)).astype(int)
            ).tolist()
        )
        tail_frames = len(sampled)

        rdf = RadialDistributionFunction(r_max, num_bins)
        ocf = OrientationalCorrelationFunction(r_max, num_bins)

        steps, energy, volume, npart = [], [], [], []
        s_vals, pos_acc, or_acc, vol_acc = [], [], [], []

        for i, row in enumerate(db.select()):
            steps.append(row.step)
            energy.append(row.total_energy)
            volume.append(row.vol)
            npart.append(row.num_particles)
            pos_acc.append(row.pos_acc_rate)
            or_acc.append(row.or_acc_rate)
            vol_acc.append(row.vol_acc_rate)

            or_vec = np.asarray(row.data["or_vec"])
            s_vals.append(float(np.linalg.eigvalsh(nematic_q_tensor(or_vec))[-1]))

            if i in sampled:
                # Both measurements share one toatoms() call, because each of them
                # also builds its own distance matrix. toatoms() doesn't carry
                # or_vec, so the OCF reads it from array_data, which is row.data.
                frame = row.toatoms()
                rdf.compute(frame, row.key_value_pairs, row.data)
                ocf.compute(frame, row.key_value_pairs, row.data)

    steps = np.asarray(steps, dtype=float)
    if np.any(np.diff(steps) < 0):
        raise ValueError(f"{path} step axis is not monotonic; db may be corrupt")

    n_particles = int(npart[-1])
    volume = np.asarray(volume, dtype=float)
    return RunTrace(
        run_dir=run_dir,
        db_name=db_name,
        n_particles=n_particles,
        steps=steps,
        cycles=steps / max(n_particles, 1),
        energy=np.asarray(energy, dtype=float),
        volume=volume,
        density=np.asarray(npart, dtype=float) / volume,
        nematic_s=np.asarray(s_vals, dtype=float),
        pos_acc=np.asarray(pos_acc, dtype=float),
        or_acc=np.asarray(or_acc, dtype=float),
        vol_acc=np.asarray(vol_acc, dtype=float),
        tail_frames=tail_frames,
        rdf=rdf.finalize(),
        ocf=ocf.finalize(),
    )


def _chrome(ax):
    """Light grid and axes, so the data stands out."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _finish(fig, trace, path):
    fig.suptitle(trace.label, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_structure(trace, path):
    """g(r) and orientational correlation averaged over the end of the run.

    These identify the phase. A crystal has sharp, well-separated g(r) peaks and
    a structured s2(r). A liquid has one broad first shell, after which g(r)
    decays to 1 and s2(r) to 0. Reference lines mark both of those limits.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    axs[0].axhline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    axs[0].plot(trace.rdf["r"], trace.rdf["g_r"], lw=2, color=SERIES["pos"], zorder=3)
    axs[0].set_ylabel("g(r)")

    axs[1].axhline(0.0, color=MUTED, lw=1, ls=":", zorder=1)
    axs[1].plot(trace.ocf["r"], trace.ocf["s2_r"], lw=2, color=SERIES["or"], zorder=3)
    axs[1].set_ylabel(r"$\langle P_2(\cos\theta)\rangle$")

    for ax in axs:
        ax.set_xlabel("r  (Å)")
        _chrome(ax)
    axs[0].set_title(
        f"averaged over {trace.tail_frames} frames sampled across the last "
        f"{TAIL_FRACTION:.0%} of the run",
        fontsize=9, color=MUTED, loc="left",
    )
    return _finish(fig, trace, path)


def plot_phase(trace, path):
    """Nematic order and density against cycle, showing when the run settled.

    The two panels share a cycle axis because an ordered phase has both a high S
    and a higher density, so the two curves should change together at a
    transition.
    """
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axs[0].plot(trace.cycles, trace.nematic_s, lw=2, color=SERIES["or"])
    axs[0].set_ylabel("nematic order  S")
    axs[0].set_ylim(bottom=0)

    axs[1].plot(trace.cycles, trace.density, lw=2, color=SERIES["vol"])
    axs[1].set_ylabel(r"density  N/V  (Å$^{-3}$)")

    axs[-1].set_xlabel("cycle  (N attempted moves)")
    for ax in axs:
        _chrome(ax)
    return _finish(fig, trace, path)


def plot_acceptance(trace, path):
    """Acceptance rate of each move type against cycle, with the tuner's target.

    This shows whether the move widths are right. The widths are tuned during
    equilibration and then fixed, so in production these curves should stay flat
    near the target. A curve that drifts away means the configuration has changed
    since the width was tuned.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # The target goes in the legend, not as an annotation on the line: these
    # traces are dense enough that in-plot text lands under the data.
    ax.axhline(TARGET_ACC_RATE, color=MUTED, lw=1, ls="--", zorder=1,
               label=f"target {TARGET_ACC_RATE:.1%}")
    for key, label in (("pos", "position"), ("or", "orientation"), ("vol", "volume")):
        ax.plot(trace.cycles, getattr(trace, f"{key}_acc"), lw=2,
                color=SERIES[key], label=label, zorder=3)

    ax.set_xlabel("cycle  (N attempted moves)")
    ax.set_ylabel("acceptance")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=4, loc="upper right")
    _chrome(ax)
    return _finish(fig, trace, path)


def plot_energy(trace, path):
    """Energy per particle against cycle, to check convergence."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trace.cycles, trace.energy / trace.n_particles, lw=2, color=SERIES["pos"])
    ax.set_xlabel("cycle  (N attempted moves)")
    ax.set_ylabel("U / N  (eV)")
    _chrome(ax)
    return _finish(fig, trace, path)


PLOTS = {
    "structure": plot_structure,
    "phase": plot_phase,
    "acceptance": plot_acceptance,
    "energy": plot_energy,
}


def render(run_dir, which=None, db_name="equilibration.db", out_dir=None):
    """Render the selected diagnostics for ``run_dir``; return ``{name: png_path}``.

    ``which`` defaults to every plot in :data:`PLOTS`. The db is loaded once and
    shared by all of them. File names start with the db's name
    (``equilibration_structure.png``, ``simulation_structure.png``), so plotting a
    production run never overwrites the equilibration figures.
    """
    which = list(PLOTS) if which is None else list(which)
    unknown = [w for w in which if w not in PLOTS]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown}; choose from {sorted(PLOTS)}")

    trace = load_run(run_dir, db_name=db_name)
    out_dir = run_dir if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(db_name)[0]
    written = {}
    for name in which:
        written[name] = PLOTS[name](trace, os.path.join(out_dir, f"{stem}_{name}.png"))
    return written


# Ellipsoid semiaxes (A) written as a per-particle ``shape`` array, for viewers.
SHAPE = (2.5, 2.5, 1.0)


def export_xyz(run_dir, db_name="simulation.db"):
    """Write ``<db stem>.xyz`` (extended XYZ) next to the db, for OVITO or ASE's GUI.

    Each frame carries the per-particle ``c_q``, ``or_vec`` and ``shape`` arrays
    and its ``total_energy``. Returns the path written.
    """
    xyz_path = os.path.join(run_dir, os.path.splitext(db_name)[0] + ".xyz")
    frames = []
    with connect(os.path.join(run_dir, db_name)) as db:
        for row in db.select():
            atoms = row.toatoms()
            atoms.new_array("c_q", np.asarray(row.data["c_q"]))
            atoms.new_array("or_vec", np.asarray(row.data["or_vec"]))
            atoms.new_array("shape", np.tile(SHAPE, (len(atoms), 1)))
            atoms.info["total_energy"] = row["total_energy"]
            frames.append(atoms)
    ase.io.write(xyz_path, frames, format="extxyz")
    return xyz_path
