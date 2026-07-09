from asmcmc.initialize import HerringboneLatticeInitializer
from asmcmc.metropolis import MetropolisCalculator
from asmcmc.potentials import CACELLI_POTENTIAL
from asmcmc.measurements import (
    TrajectoryAnalyzer,
    RadialDistributionFunction,
    OrientationalCorrelationFunction,
    HeatCapacity,
    NematicOrderParameter,
    AverageEnthalpy,
)

import pickle
import json

T = 100.0  # K
P = 6.324209e-7  # eV / Å^3 = 1 atm

SEED = 43
N_PARTICLES = 128

OUTPUT_DIR = f"../results/validation/{T}_{P}/herringbone"


def build_initializer():
    return HerringboneLatticeInitializer(
        n_particles=N_PARTICLES,
        density=None,
        pos_jitter=0.0,
        or_jitter=0.0,
        seed=SEED,
        potential=CACELLI_POTENTIAL,
    )


def build_calculator():
    return MetropolisCalculator(
        T,
        P,
        initializer=build_initializer(),
        potential=CACELLI_POTENTIAL,
        output_dir=OUTPUT_DIR,
        pos_delt=0.07,
        or_delt=0.02,
    )


def equilibrate():
    calculator = build_calculator()
    calculator.equilibrate(1_280_000, N_PARTICLES)
    return calculator


def run_simulation():
    metro = MetropolisCalculator.from_equilibration(OUTPUT_DIR)
    metro.calculate_trajectory(
        num_steps=2_000_000, block_size=N_PARTICLES, num_eq_steps=0, buffer_size=100
    )
    return metro


def take_measurements():
    RUN_DIR = "../results/validation/100.0_6.324209e-07/herringbone"

    # Pull the state point straight from the run's write-once config so the
    # measurements stay consistent with how the trajectory was generated.
    with open(f"{RUN_DIR}/run_config.json") as f:
        config = json.load(f)

    temp = config["temp"]
    pressure = config["pressure"]
    n_particles = config["init"]["init_n_particles"]

    print(f"config temp:     {temp}")
    print(f"hardcoded temp:  {T}")
    print(f"config pressure:     {pressure}")
    print(f"hardcoded pressure:  {P}")
    print(f"config num particles:     {n_particles}")
    print(f"hardcoded num particles:  {N_PARTICLES}")

    # r_max stays below half the (NPT-fluctuating) box: the RDF/OCF only fill bins
    # inside each frame's L/2
    R_MAX = 8.5
    NUM_BINS = 85

    analyzer = TrajectoryAnalyzer(f"{RUN_DIR}/simulation.db")
    analyzer.add_measurement("rdf", RadialDistributionFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("ocf", OrientationalCorrelationFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("nematic", NematicOrderParameter())
    analyzer.add_measurement("enthalpy", AverageEnthalpy(pressure))
    analyzer.add_measurement(
        "heat_capacity", HeatCapacity(temp, n_particles, pressure=pressure)
    )
    results = analyzer.run_analysis()

    RESULTS_PATH = f"{RUN_DIR}/measurements.pkl"

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)


def main():
    equilibrate()


if __name__ == "__main__":
    main()
