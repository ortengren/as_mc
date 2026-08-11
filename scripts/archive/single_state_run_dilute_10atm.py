"""Bracket the (T, P) state point from the *dilute* side.

Starts from a disordered, low-density random config and slowly pressurizes it up
to the target pressure with a staged ``pressure_ramp``. Raising P gradually lets
the box densify onto the fluid/ordered branch instead of collapsing straight into
a glass (the failure mode of dropping the dilute start at the target pressure).

Paired with ``single_state_run_dense.py``, which approaches the same target from
an ordered, over-dense columnar start — agreement between the two is the check
that the state point is actually equilibrated rather than hysteretic.
"""

import numpy as np

from asmcmc.base.initialize import RandomLatticeInitializer
from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.utils.equilibration import pressure_ramp, continue_point
from asmcmc.base.metropolis import MetropolisCalculator

T = 100.0  # K
P_TARGET = 6.324209e-6  # eV / Å^3 = 10 atm

N_PARTICLES = 125
DENSITY = 0.6  # rho* of the dilute, disordered start
SEED = 42

# Geometric (log-spaced) pressure schedule ending exactly at the target. Starting
# well below it and squeezing over several stages spreads the compression out so
# the system can relax at each density instead of jamming. Tune N_STAGES / the
# starting fraction to trade wall-clock for how gently it densifies.
N_STAGES = 6
PRESSURES = np.geomspace(P_TARGET / 100, P_TARGET, N_STAGES).tolist()

# Per-stage step budget: modest at each intermediate pressure, largest at the
# final (target) stage — that's the one we want fully converged and samplable.
STEPS = [300_000] * (N_STAGES - 1) + [1_500_000]

OUTPUT_DIR = f"../results/validation/{T}_{P_TARGET}/dilute"


def build_initializer():
    return RandomLatticeInitializer(
        n_particles=N_PARTICLES,
        density=DENSITY,
        seed=SEED,
        potential=CACELLI_POTENTIAL,
    )


def run_ramp():
    """Run the full pressure ramp; returns the per-stage run dirs (ascending P).

    Each stage writes its own resumable dir under OUTPUT_DIR; the last is the
    target-pressure state, resumable via ``MetropolisCalculator.from_equilibration``
    (to equilibrate further) or ``calculate_trajectory`` (to collect observables).
    """
    return pressure_ramp(
        T,
        PRESSURES,
        STEPS,
        initializer=build_initializer(),
        output_dir=OUTPUT_DIR,
        potential=CACELLI_POTENTIAL,
        seed=SEED,
        block_size=N_PARTICLES,
        progress=True,  # per-stage header + tqdm bar
    )


def resume_equilibration():
    """Resume the last stage of the ramp and equilibrate further."""

    outdir = f"{OUTPUT_DIR}/stage05_P6.32421e-06"
    continue_point(outdir, extra_steps=2_000_000, block_size=N_PARTICLES, progress=True)


def main():
    resume_equilibration()


if __name__ == "__main__":
    main()
