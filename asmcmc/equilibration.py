"""
equilibration.py — Single-point NPT equilibration primitives.

The building blocks for equilibrating one (temperature, pressure) NPT state
point, in *physical* units (Kelvin, eV/Å³), into a resumable run directory —
plus the tools to discover, resume, and eyeball those directories::

    equilibrate_point   run one point into a fresh run dir
    continue_point      resume a finished point in place (re-entrant equilibrate)
    find_point_dirs     discover resumable T*_P*/{seed}/ dirs under a scan tree
    plot_point_results  per-point energy/volume-vs-step convergence diagnostics
    point_dirname       filesystem-safe T{temp}_P{pressure} directory name

This is the serial, single-point layer. The parallel grid scan that drives many
of these at once lives in ``npt_equilibration`` (and re-exports them for backward
compatibility); longer serial protocols built on top of them (e.g. a staged
pressure ramp) also belong here.
"""

import os
import glob
import random
from collections.abc import Sequence

import numpy as np
from ase.db import connect
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from asmcmc.initialize import FrameInitializer
from asmcmc.metropolis import MetropolisCalculator


def point_dirname(temp, pressure):
    """Filesystem-safe directory name for one (T, P) point: ``T{temp}_P{pressure}``.

    ``:g`` keeps it compact and lossless for the typical values (e.g. ``T300_P0``,
    ``T300_P0.001``, ``T300_P1e-06``) without trailing-zero noise.
    """
    return f"T{temp:g}_P{pressure:g}"


def equilibrate_point(
    temp,
    pressure,
    num_steps,
    initializer,
    block_size,
    output_dir,
    seed=None,
    buffer_size=100,
    dynamic_delta=True,
    potential=None,
    progress=False,
):
    """Equilibrate one (T, P) NPT point into ``output_dir`` and return that path.

    Builds an NPT sampler from ``initializer`` at (``temp`` [K], ``pressure``
    [eV/Å³]) and runs ``equilibrate`` (adaptive move widths), which writes
    ``equilibration.db`` + a write-once ``run_config.json`` into ``output_dir``.

    ``potential`` (default ``None`` ⇒ ``DEFAULT_POTENTIAL``) selects the pair
    potential — pass a ``GBQPotential`` to run a non-default fit. ``progress``
    (default ``False``) shows a tqdm bar for the run — left off by default so
    parallel-scan workers don't interleave bars.

    Reseeding the global ``random``/``np.random`` streams (with ``seed``, or the
    initializer's own seed) makes the point reproducible and independent of how
    many other points ran first in the same worker — the safety property the
    parallel scan relies on. The initial lattice is already pinned by the
    initializer's seed.
    """
    if seed is None:
        seed = initializer.seed
    random.seed(seed)
    np.random.seed(seed)

    # equilibrate() *appends* to equilibration.db, so refuse to run on top of an
    # existing one (that would interleave a fresh trajectory into old frames).
    # Resuming is a separate path: MetropolisCalculator.from_equilibration.
    if os.path.exists(os.path.join(output_dir, "equilibration.db")):
        raise FileExistsError(
            f"{output_dir} already has an equilibration.db; "
            "resume it with MetropolisCalculator.from_equilibration instead."
        )

    metro = MetropolisCalculator(
        temp=temp,
        pressure=pressure,
        initializer=initializer,
        potential=potential,
        output_dir=output_dir,
    )
    metro.equilibrate(
        num_steps=num_steps,
        block_size=block_size,
        buffer_size=buffer_size,
        dynamic_delta=dynamic_delta,
        progress=progress,
    )
    return output_dir


def find_point_dirs(out_dir):
    """Every resumable point dir under ``out_dir`` — those holding both a
    ``run_config.json`` and an ``equilibration.db`` at the ``T*_P*/{seed}/`` layout
    that ``main`` writes. Sorted for stable ordering. This is the set of gridpoints
    an ``extend_points`` run continues."""
    dirs = []
    for cfg_path in glob.glob(os.path.join(out_dir, "*", "*", "run_config.json")):
        d = os.path.dirname(cfg_path)
        if os.path.exists(os.path.join(d, "equilibration.db")):
            dirs.append(d)
    return sorted(dirs)


def continue_point(
    output_dir,
    extra_steps,
    block_size=None,
    buffer_size=100,
    dynamic_delta=True,
    vol_delt=None,
    progress=False,
):
    """Resume one finished point in place and equilibrate ``extra_steps`` further.

    Rebuilds the sampler from the point's ``run_config.json`` + last
    ``equilibration.db`` frame via ``from_equilibration`` (which restores
    ``step_count`` and points ``output_dir`` back at the same dir), then calls the
    re-entrant ``equilibrate`` with an *absolute* target of ``step_count +
    extra_steps`` so the trajectory is appended to the same db rather than
    restarted. ``block_size`` defaults to the particle count (matching ``main``'s
    one-frame-per-pass default).

    ``vol_delt`` (default ``None``) is forwarded to ``from_equilibration`` to
    optionally reset the carried volume move width before continuing.

    Reseeds the global RNG from the point's seed subdir (offset by the resumed
    step) so the extension is reproducible and independent of how many other
    points a worker continued first — the same parallel-safety property
    ``equilibrate_point`` relies on.
    """
    metro = MetropolisCalculator.from_equilibration(output_dir, vol_delt=vol_delt)

    seed_name = os.path.basename(os.path.normpath(output_dir))
    seed = int(seed_name) if seed_name.isdigit() else abs(hash(output_dir))
    random.seed(seed + metro.step_count)
    np.random.seed((seed + metro.step_count) % (2**32))

    if block_size is None:
        block_size = len(metro.current_frame)
    metro.equilibrate(
        num_steps=metro.step_count + extra_steps,
        block_size=block_size,
        buffer_size=buffer_size,
        dynamic_delta=dynamic_delta,
        progress=progress,
    )
    return output_dir


def plot_point_results(
    output_dir, db_name="equilibration.db", png_name="equilibration_diagnostics.png"
):
    """Per-point convergence diagnostics: total energy and volume versus step,
    read back from the point's ``db_name``.

    The visual equilibration check for one (T, P) point — flat energy and volume
    tails mean the point has relaxed and is equilibrated enough to resume /
    sample from. Writes ``png_name`` into ``output_dir`` and returns its path.
    (``db_name``/``png_name`` are parameterised so the same plot can render a
    production trajectory's ``simulation.db`` without clobbering the equilibration
    diagnostics.)
    """
    steps, energy, vol = [], [], []
    with connect(os.path.join(output_dir, db_name)) as db:  # type: ignore  # Pylance false positive
        for row in db.select():
            steps.append(row.step)
            energy.append(row.total_energy)
            vol.append(row.vol)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(steps, energy, marker=".")
    axs[0].set_ylabel("Total energy  (eV)")
    axs[1].plot(steps, vol, marker=".", color="tab:green")
    axs[1].set_ylabel("Volume  (Å³)")
    for ax in axs:
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    fig.suptitle(os.path.relpath(output_dir))
    fig.tight_layout()
    png = os.path.join(output_dir, png_name)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def pressure_ramp(
    temp,
    pressures: Sequence[float],
    num_steps: int | Sequence[int],
    initializer,
    output_dir,
    potential=None,
    seed=None,
    block_size=None,
    buffer_size=100,
    dynamic_delta=True,
    ascending=True,
    progress=False,
):
    """Equilibrate one system through a staged pressure ramp and return the stage dirs.

    Walks pressure across ``pressures``, equilibrating at each stage before stepping
    up. Slowly raising P keeps the system on the fluid/ordered branch as it densifies
    instead of over-driving it into a jammed glass.

    Each stage writes its own resumable run dir ``output_dir/stage{k}_P{p}/``
    (equilibration.db + write-once run_config.json + diagnostics PNG), exactly
    like a scan point, so any stage is independently resumable via
    ``MetropolisCalculator.from_equilibration``. Stage 0 starts from
    ``initializer``; each later stage starts from the *previous* stage's final
    frame (wrapped in a ``FrameInitializer``, which carries the orientation
    arrays), so the configuration flows continuously through the ramp.

    ``num_steps`` is the **per-stage** budget (each stage's db starts at step 0):
    pass one int to apply the same budget to every stage, or a sequence to size
    them individually — the dense final stage usually wants the largest. Note that
    move widths (incl. ``vol_delt``) are retuned from defaults each stage rather
    than carried across; a future refinement could thread the tuned widths forward.

    ``potential`` is forwarded to every stage (default ``None`` ⇒ DEFAULT_POTENTIAL).
    ``seed`` (default ``None`` ⇒ the initializer's own seed) fixes the MC stream;
    stage ``k`` uses ``seed + k`` so stages are reproducible yet independent.
    ``block_size`` defaults to the particle count (one recorded frame per pass).
    ``progress`` (default ``False``) prints a per-stage header and shows that
    stage's equilibration progress bar.
    """
    if isinstance(num_steps, int):
        num_steps = [num_steps] * len(pressures)

    if len(pressures) == 0:
        raise ValueError("pressures must be non-empty")

    if len(num_steps) != len(pressures):
        raise ValueError(
            f"steps and pressures must be the same length; "
            f"got {len(num_steps)} steps for {len(pressures)} pressures"
        )
    if ascending:
        sorted_pairs = sorted(zip(pressures, num_steps), key=lambda x: x[0])
    else:
        sorted_pairs = sorted(
            zip(pressures, num_steps), key=lambda x: x[0], reverse=True
        )

    os.makedirs(output_dir, exist_ok=True)
    base_seed = initializer.seed if seed is None else seed
    if block_size is None:
        block_size = initializer.n_particles

    stage_dirs = []
    for k, (pressure, n_steps) in enumerate(sorted_pairs):
        stage_dir = os.path.join(output_dir, f"stage{k:02d}_P{pressure:g}")

        if k == 0:
            stage_init = initializer
        else:
            # Chain: the previous stage's final frame (with its c_q/or_vec
            # orientation arrays, reconstructed by from_equilibration) is the
            # starting config for this stage at the next pressure.
            prev = MetropolisCalculator.from_equilibration(stage_dirs[-1])
            stage_init = FrameInitializer(prev.current_frame)

        stage_seed = None if base_seed is None else base_seed + k
        if progress:
            print(
                f"[pressure_ramp] stage {k + 1}/{len(sorted_pairs)}  "
                f"P={pressure:g} eV/Å³  ({n_steps} steps)"
            )
        equilibrate_point(
            temp,
            pressure,
            num_steps=n_steps,
            initializer=stage_init,
            block_size=block_size,
            output_dir=stage_dir,
            seed=stage_seed,
            buffer_size=buffer_size,
            dynamic_delta=dynamic_delta,
            potential=potential,
            progress=progress,
        )
        plot_point_results(stage_dir)
        stage_dirs.append(stage_dir)

    return stage_dirs
