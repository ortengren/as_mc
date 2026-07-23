from asmcmc.initialize import (
    RandomLatticeInitializer,
    ColumnarLatticeInitializer,
    FrameInitializer,
)
from asmcmc.metropolis import MetropolisCalculator
from asmcmc.potentials import GBQPotential, CACELLI_POTENTIAL

T = 100.0  # K
P = 6.324209e-6  # eV / Å^3 = 10 atm

N_PARTICLES = 125
DENSITY = 1.5
SEED = 43

OUTPUT_DIR = f"../results/validation/{T}_{P}/dense"


def build_initializer():
    return ColumnarLatticeInitializer(
        n_particles=N_PARTICLES, density=DENSITY, seed=SEED, potential=CACELLI_POTENTIAL
    )


def build_calculator():
    return MetropolisCalculator(
        T,
        P,
        initializer=build_initializer(),
        potential=CACELLI_POTENTIAL,
        output_dir=OUTPUT_DIR,
    )


def equilibrate():
    calculator = build_calculator()
    calculator.equilibrate(2_000_000, 125)
    return calculator


def resume_equilibration():
    metro = MetropolisCalculator.from_equilibration(OUTPUT_DIR)
    metro.equilibrate(3_000_000, 125)
    return metro


def run_simulation():
    metro = MetropolisCalculator.from_equilibration(OUTPUT_DIR)
    metro.calculate_trajectory(
        num_steps=2_000_000,
        block_size=125,
        num_eq_steps=0,
    )
    return metro


def main():
    metro = run_simulation()


if __name__ == "__main__":
    main()
