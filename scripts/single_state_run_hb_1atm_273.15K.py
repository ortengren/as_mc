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
from asmcmc.equilibration import continue_point

import pickle
import json
import matplotlib.pyplot as plt
from ase.db import connect

T = 273.15  # K
P = 6.324209e-7  # eV / Å^3 = 1 atm

SEED = 311
N_PARTICLES = 400
NL_RADIUS = 6.8
NL_SKIN = 1.0

# Cap the adaptive rotation width to a physical libration (~14 deg). The
# herringbone's site-level orientational landscape is ~1 kT under the corrected
# GBQIII potential, so an uncapped tuner walks or_delt to near-randomizing
# rotations (~1.2 rad) that orientationally melt the crystal during the box
# collapse -> dense glass (runs 2 and 3 of this validation).
MAX_OR_DELT = 0.25  # rad

OUTPUT_DIR = f"../results/validation/{T}_{P}/herringbone_jittered_0"


def build_initializer():
    return HerringboneLatticeInitializer(
        n_particles=N_PARTICLES,
        density=None,
        pos_jitter=0.15,
        or_jitter=0.15,
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
        nl_radius=NL_RADIUS,
        nl_skin=NL_SKIN,
        vol_delt=0.025,
    )


def equilibrate():
    calculator = build_calculator()
    calculator.equilibrate(
        1_000_000, N_PARTICLES, max_or_delt=MAX_OR_DELT, buffer_size=500, progress=True
    )
    return calculator


def resume_equilibration():
    """Resume the last stage of the ramp and equilibrate further."""

    continue_point(
        OUTPUT_DIR,
        extra_steps=9_200_000,
        block_size=N_PARTICLES,
        max_or_delt=MAX_OR_DELT,
        progress=True,
        buffer_size=500,
    )


def run_simulation():
    metro = MetropolisCalculator.from_equilibration(OUTPUT_DIR)
    metro.calculate_trajectory(
        num_steps=15_000_000, block_size=N_PARTICLES, num_eq_steps=0, buffer_size=500
    )
    return metro


def take_measurements():

    # Pull the state point straight from the run's write-once config so the
    # measurements stay consistent with how the trajectory was generated.
    with open(f"{OUTPUT_DIR}/run_config.json") as f:
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
    R_MAX = 12
    NUM_BINS = 120

    analyzer = TrajectoryAnalyzer(f"{OUTPUT_DIR}/simulation.db")
    analyzer.add_measurement("rdf", RadialDistributionFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("ocf", OrientationalCorrelationFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("nematic", NematicOrderParameter())
    analyzer.add_measurement("enthalpy", AverageEnthalpy(pressure))
    analyzer.add_measurement(
        "heat_capacity", HeatCapacity(temp, n_particles, pressure=pressure)
    )
    results = analyzer.run_analysis()

    RESULTS_PATH = f"{OUTPUT_DIR}/measurements.pkl"

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)


def equilibration_diagnostics():
    steps, energy, vol = [], [], []
    with connect(f"{OUTPUT_DIR}/simulation.db") as db:
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

    fig.suptitle("Equilibration Diagnostics")
    fig.tight_layout()

    fig.savefig(
        f"{OUTPUT_DIR}/equilibration_diagnostics.png",
        dpi=150,
    )


def main():
    run_simulation()
    take_measurements()


if __name__ == "__main__":
    main()
