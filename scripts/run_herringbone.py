"""Run one herringbone state point at 1 atm under Cacelli et al.'s GBQIII potential.

This is the validation protocol that reproduced Cacelli et al.'s crystal at
100 K (see docs/findings.md). Each temperature has a fixed seed and run
directory, ``results/validation/{T}_{P}/{run name}/``:

    python scripts/run_herringbone.py --temp 150                # produce, then measure
    python scripts/run_herringbone.py --temp 150 equilibrate    # 1e6 steps from the crystal
    python scripts/run_herringbone.py --temp 150 resume         # 9.2e6 more equilibration steps
    python scripts/run_herringbone.py --temp 150 produce        # 1.5e7 production steps
    python scripts/run_herringbone.py --temp 150 measure        # -> measurements.pkl

The seed fixes the initial jitter only; the sampler's own random stream is not
seeded, so repeat runs differ.
"""

import argparse
import json
import pickle

from asmcmc.mc.initialize import HerringboneLatticeInitializer
from asmcmc.mc.measurements import (
    AverageEnthalpy,
    HeatCapacity,
    NematicOrderParameter,
    OrientationalCorrelationFunction,
    RadialDistributionFunction,
    TrajectoryAnalyzer,
)
from asmcmc.mc.metropolis import MetropolisSampler, continue_equilibration
from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.paths import REPO_ROOT
from asmcmc.units import ATM_TO_EV_PER_A3

PRESSURE = 1.0 * ATM_TO_EV_PER_A3  # eV/Å^3

# temperature (K) -> (initial-configuration seed, run directory name)
STATE_POINTS = {
    100.0: (45, "herringbone_jittered_2"),
    150.0: (313, "herringbone_jittered_0"),
    200.0: (312, "herringbone_jittered_0"),
    273.15: (311, "herringbone_jittered_0"),
    300.0: (123219, "herringbone_jittered_2"),
}

N_PARTICLES = 400
NL_RADIUS = 6.8  # Å
NL_SKIN = 1.0  # Å
VOL_DELT = 0.025
POS_JITTER = 0.15  # Å
OR_JITTER = 0.15  # rad
# Cap on the adapted rotation width (~14 degrees, a libration). Without it the
# tuner grows or_delt to near-randomising rotations that melt the crystal while
# the box is still collapsing; see docs/findings.md.
MAX_OR_DELT = 0.25  # rad
BUFFER_SIZE = 500

EQUILIBRATION_STEPS = 1_000_000
RESUME_STEPS = 9_200_000
PRODUCTION_STEPS = 15_000_000

R_MAX = 12  # Å; below half the NPT-fluctuating box, the range RDF/OCF can resolve
NUM_BINS = 120


def run_dir(temp):
    """The run directory for ``temp``."""
    _, name = STATE_POINTS[temp]
    return REPO_ROOT / "results" / "validation" / f"{temp}_{PRESSURE}" / name


def build_sampler(temp):
    """A fresh sampler on the jittered herringbone crystal."""
    seed, _ = STATE_POINTS[temp]
    initializer = HerringboneLatticeInitializer(
        n_particles=N_PARTICLES,
        density=None,  # the cif's own cell
        pos_jitter=POS_JITTER,
        or_jitter=OR_JITTER,
        seed=seed,
        potential=CACELLI_POTENTIAL,
    )
    return MetropolisSampler(
        temp,
        PRESSURE,
        initializer=initializer,
        potential=CACELLI_POTENTIAL,
        output_dir=str(run_dir(temp)),
        nl_radius=NL_RADIUS,
        nl_skin=NL_SKIN,
        vol_delt=VOL_DELT,
    )


def equilibrate(temp):
    """Start a new run: equilibrate from the crystal."""
    if (run_dir(temp) / "equilibration.db").exists():
        raise SystemExit(f"{run_dir(temp)} already has an equilibration.db; use `resume`.")
    build_sampler(temp).equilibrate(
        EQUILIBRATION_STEPS,
        N_PARTICLES,
        max_or_delt=MAX_OR_DELT,
        buffer_size=BUFFER_SIZE,
        progress=True,
    )


def resume(temp, extra_steps=RESUME_STEPS):
    """Continue the equilibration in place."""
    continue_equilibration(
        str(run_dir(temp)),
        extra_steps=extra_steps,
        block_size=N_PARTICLES,
        max_or_delt=MAX_OR_DELT,
        progress=True,
        buffer_size=BUFFER_SIZE,
    )


def produce(temp):
    """Run production from the last equilibration frame, with the tuned move widths."""
    if (run_dir(temp) / "simulation.db").exists():
        raise SystemExit(f"{run_dir(temp)} already has a simulation.db.")
    sampler = MetropolisSampler.from_equilibration(str(run_dir(temp)))
    sampler.calculate_trajectory(
        num_steps=PRODUCTION_STEPS,
        block_size=N_PARTICLES,
        num_eq_steps=0,
        buffer_size=BUFFER_SIZE,
    )


def measure(temp):
    """Write ``measurements.pkl`` from ``simulation.db``, using the run's own T, P and N."""
    directory = run_dir(temp)
    config = json.loads((directory / "run_config.json").read_text())
    run_temp, pressure = config["temp"], config["pressure"]
    n_particles = config["init"]["init_n_particles"]
    if (run_temp, pressure, n_particles) != (temp, PRESSURE, N_PARTICLES):
        print(
            f"warning: {directory} was run at T={run_temp}, P={pressure}, N={n_particles}; "
            f"this script expects T={temp}, P={PRESSURE}, N={N_PARTICLES}"
        )

    analyzer = TrajectoryAnalyzer(str(directory / "simulation.db"))
    analyzer.add_measurement("rdf", RadialDistributionFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("ocf", OrientationalCorrelationFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("nematic", NematicOrderParameter())
    analyzer.add_measurement("enthalpy", AverageEnthalpy(pressure))
    analyzer.add_measurement(
        "heat_capacity", HeatCapacity(run_temp, n_particles, pressure=pressure)
    )
    with open(directory / "measurements.pkl", "wb") as f:
        pickle.dump(analyzer.run_analysis(), f)


STAGES = {"equilibrate": equilibrate, "resume": resume, "produce": produce, "measure": measure}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--temp", type=float, required=True, choices=sorted(STATE_POINTS))
    parser.add_argument(
        "stage", nargs="?", choices=sorted(STAGES), help="default: produce, then measure"
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=RESUME_STEPS,
        help="equilibration steps to add with `resume` (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.stage is None:
        produce(args.temp)
        measure(args.temp)
    elif args.stage == "resume":
        resume(args.temp, args.extra_steps)
    else:
        STAGES[args.stage](args.temp)


if __name__ == "__main__":
    main()
