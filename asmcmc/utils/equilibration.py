"""Resume a finished equilibration in place."""

import os
import random

import numpy as np

from asmcmc.base.metropolis import MetropolisCalculator


def continue_point(
    output_dir,
    extra_steps,
    block_size=None,
    buffer_size=100,
    dynamic_delta=True,
    vol_delt=None,
    max_or_delt=None,
    progress=False,
):
    """Resume one finished point in place and equilibrate ``extra_steps`` further.

    Rebuilds the sampler from the point's ``run_config.json`` + last
    ``equilibration.db`` frame via ``from_equilibration`` (which restores
    ``step_count`` and points ``output_dir`` back at the same dir), then calls the
    re-entrant ``equilibrate`` with an *absolute* target of ``step_count +
    extra_steps`` so the trajectory is appended to the same db rather than
    restarted. ``block_size`` defaults to the particle count.

    ``vol_delt`` (default ``None``) is forwarded to ``from_equilibration`` to
    optionally reset the carried volume move width before continuing.
    ``max_or_delt`` is forwarded to ``equilibrate`` to cap the adapted rotation
    width (a resumed run re-tunes its deltas, so an uncapped continuation could
    otherwise walk or_delt back up).

    Reseeds the global RNG from the point's seed subdir (offset by the resumed
    step) so the extension is reproducible and independent of how many other
    points a worker continued first.
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
        max_or_delt=max_or_delt,
        progress=progress,
    )
    return output_dir
