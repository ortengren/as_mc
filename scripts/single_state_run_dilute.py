from asmcmc.initialize import (
    RandomLatticeInitializer,
    ColumnarLatticeInitializer,
    FrameInitializer,
)
from asmcmc.metropolis import MetropolisCalculator
from asmcmc.potentials import GBQPotential, CACELLI_POTENTIAL

T = 100.0  # K
P = 6.324209e-6  # eV / Å^3 = 1 atm

N_PARTICLES = 125
DENSITY = 0.6
SEED = 42

OUTPUT_DIR = f"../results/validation/{T}_{P}/dilute"


def build_initializer():
    return RandomLatticeInitializer(
        n_particles=N_PARTICLES,
        density=DENSITY,
        seed=SEED,
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
    calculator.equilibrate(2_500_000, 125, vol_max_scale=1.05)
    return calculator


def main():
    metro = equilibrate()


if __name__ == "__main__":
    main()
