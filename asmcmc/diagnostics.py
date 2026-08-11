"""
diagnostics.py — Read one run directory and render what it says about the run.

``state_point.plot_point_results`` answers *has this converged* (energy and volume
vs step). This module answers *what phase did it settle into, and was it sampled
properly* — the question a validation run is actually asking. Four figures, each
grouped around one question:

    structure.png    RDF g(r) + orientational correlation, tail-averaged
    phase.png        nematic order S + density vs cycle
    acceptance.png   position / orientation / volume acceptance vs cycle
    energy.png       total energy vs cycle

Nothing here needs the sampler to record anything new: acceptance rates,
``num_particles`` and ``vol`` are already block scalars, ``or_vec`` is already in
``array_data``, and the RDF/OCF accumulators already exist in
:mod:`asmcmc.measurements`. This is assembly and plotting only.

    from asmcmc.diagnostics import render
    render("results/validation/150.0_6.324209e-07/phase_check")

The CLI wrapper is ``scripts/plot_run.py``.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
from ase.db import connect
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from asmcmc.measurements import (
    OrientationalCorrelationFunction,
    RadialDistributionFunction,
    nematic_q_tensor,
)
from asmcmc.metropolis import TARGET_ACC_RATE

# Window of recorded frames averaged into the RDF/OCF. A single frame's g(r) is
# too noisy at N~500 to read a phase off, so a tail is averaged rather than one
# frame; the window is the last TAIL_FRACTION of the run so the curves still
# describe where it *ended up*, not its whole history.
#
# TAIL_MAX_FRAMES caps the cost. RDF and OCF each build a full mic distance matrix
# per frame — measured at ~200 ms/frame for N=400 — so a literal 10% tail of a
# 25,500-frame equilibration would be 2,550 frames and over eight minutes. Frames
# are instead sampled *evenly across* the window: same tail semantics, bounded
# cost, and the samples are less correlated than consecutive blocks would be.
TAIL_FRACTION = 0.1
TAIL_MAX_FRAMES = 40

R_MAX = 12  # stays below half the (NPT-fluctuating) box; RDF/OCF skip wider bins
NUM_BINS = 120

# Validated 3-colour categorical palette (dataviz reference theme, slots 1-3),
# assigned to move types in fixed order and never cycled. Checked against the
# light chart surface: lightness band, chroma floor, CVD separation (worst
# adjacent dE 9.2 deutan), normal-vision floor (27.6). The aqua's contrast warning
# is discharged by the legend, which is mandatory for a 3-series chart anyway.
SERIES = {"pos": "#2a78d6", "or": "#eb6834", "vol": "#1baf7a"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"


@dataclass
class RunTrace:
    """One run directory's recorded blocks, reduced to plottable arrays.

    ``or_vec`` is *not* retained: a long equilibration holds one (N, 3) array per
    recorded block, so keeping them would cost hundreds of MB for a quantity that
    reduces to a single scalar. :func:`load_run` collapses each frame to its
    nematic S during the read and discards the array.
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
    """Stream a run dir's db once and return a :class:`RunTrace`.

    One pass, because a long equilibration db is expensive to re-read and the four
    plots would otherwise each pay for it. Per-frame scalars are collected for
    every block; the last ``TAIL_FRACTION`` of frames are additionally fed to the
    RDF/OCF accumulators via their per-frame ``compute()`` API. (``TrajectoryAnalyzer``
    is not used here: it always consumes the whole db, and the point of the tail is
    to describe where the run ended up.)

    Rows are read in write order, which is step order — a resumed run appends. The
    step axis is asserted monotonic rather than sorted, so a db that violates that
    is reported instead of being silently reordered.
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
                # One toatoms() shared by both measurements -- it is not free, and
                # each of them additionally builds its own mic distance matrix.
                # toatoms() does not carry or_vec; OCF reads it out of array_data,
                # which is exactly what row.data is.
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
    """Recessive grid and axes so the data carries the ink."""
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
    """Tail-averaged g(r) and orientational correlation — the phase read.

    A crystal shows sharp, well-separated g(r) peaks and structured s2(r); a liquid
    shows one broad first shell decaying to g=1 and s2 to 0. Reference lines mark
    both uncorrelated limits so the eye has an anchor.
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
    """Nematic order and density vs cycle — when the run picked its basin.

    Stacked on a shared cycle axis because they answer the same question from two
    sides: an ordered phase holds S up *and* sits at the denser volume, and the two
    should move together at a transition.
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
    """Acceptance per move type vs cycle, against the tuner's target.

    The panel that says whether the move widths are right — the whole point of the
    calibrate-then-fix protocol. Under fixed widths these should sit flat near the
    target; a curve drifting away means the configuration moved out from under the
    width it was calibrated at.
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
    """Total energy vs cycle — the convergence check, per molecule as well."""
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
    shared across them. Output names are prefixed with the db stem
    (``equilibration_structure.png``, ``simulation_structure.png``) so rendering a
    production trajectory never clobbers the equilibration figures.
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
