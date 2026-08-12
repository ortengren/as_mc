from asmcmc.base.initialize import HerringboneLatticeInitializer
from asmcmc.base.metropolis import MetropolisCalculator
from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.utils.measurements import (
    TrajectoryAnalyzer,
    RadialDistributionFunction,
    OrientationalCorrelationFunction,
    HeatCapacity,
    NematicOrderParameter,
    AverageEnthalpy,
)
from asmcmc.utils.equilibration import continue_point

import pickle
import json

from flow import FlowProject

class Project(FlowProject):
    pass

@Project.label
def equilibriation_finished(job):
    return job.isfile("equilibration.db")

@Project.label
def production_finished(job):
    return job.isfile("simulation.db")

@Project.label
def measurements_finished(job):
    return job.isfile("measurements.pkl")


def build_initializer(job):
    if job.sp["initial_config"] == 'herringbone':
        return HerringboneLatticeInitializer(
                n_particles=job.sp["n_particles"],
                density=None,
                pos_jitter=0.15,
                or_jitter=0.15,
                seed=job.sp["seed"],
                potential=CACELLI_POTENTIAL,
            )
def build_calculator(job):
    potential_dict = {"gbq":CACELLI_POTENTIAL, "anisoap":None}   # fill in anisoap later...

    initializer = build_initializer(job)
    potential = potential_dict[job.sp['potential']]
    output_dir = job.fn("")     # job.fn("fname") returns  path/to/workspace/fname, so empty quotes just returns path/to/workspace
    return MetropolisCalculator(
            job.sp["T"],
            job.sp["P"],
            initializer=initializer,
            potential=potential,
            output_dir=output_dir,
            nl_radius=job.sp["nl_radius"],
            nl_skin=job.sp["nl_skin"],
            vol_delt=job.sp["vol_delta"],
        )

@Project.post(equilibriation_finished)
@Project.operation
def equilibriate(job):
    calculator = build_calculator(job)
    calculator.equilibrate(
            1_00_000, job.sp["n_particles"], max_or_delt=job.sp["max_or_delt"], buffer_size=500, progress=True
        )
    return calculator

@Project.pre(equilibriation_finished)
@Project.post(production_finished)
@Project.operation
def run_simulation(job):
    output_dir = job.fn("")
    metro = MetropolisCalculator.from_equilibration(output_dir)
    metro.calculate_trajectory(
        num_steps=150_000, block_size=job.sp["n_particles"], num_eq_steps=0, buffer_size=500
    )
    return metro

@Project.pre(production_finished)
@Project.post(measurements_finished)
@Project.operation
def take_measurements(job):
    # RUN_DIR = "../results/validation/100.0_6.324209e-07/herringbone_jittered_2"
    RUN_DIR = job.fn("")

    # Pull the state point straight from the run's write-once config so the
    # measurements stay consistent with how the trajectory was generated.
    # with open(f"{RUN_DIR}/run_config.json") as f:
    with open(job.fn("run_config.json")) as f:
        config = json.load(f)

    temp = config["temp"]
    pressure = config["pressure"]
    n_particles = config["init"]["init_n_particles"]

    print(f"config temp:     {temp}")
    print(f"hardcoded temp:  {job.sp["T"]}")
    print(f"config pressure:     {pressure}")
    print(f"hardcoded pressure:  {job.sp["P"]}")
    print(f"config num particles:     {n_particles}")
    print(f"hardcoded num particles:  {job.sp["n_particles"]}")

    # r_max stays below half the (NPT-fluctuating) box: the RDF/OCF only fill bins
    # inside each frame's L/2
    R_MAX = 12
    NUM_BINS = 120

    # analyzer = TrajectoryAnalyzer(f"{RUN_DIR}/simulation.db")
    analyzer = TrajectoryAnalyzer(job.fn("simulation.db"))
    analyzer.add_measurement("rdf", RadialDistributionFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("ocf", OrientationalCorrelationFunction(R_MAX, NUM_BINS))
    analyzer.add_measurement("nematic", NematicOrderParameter())
    analyzer.add_measurement("enthalpy", AverageEnthalpy(pressure))
    analyzer.add_measurement(
        "heat_capacity", HeatCapacity(temp, n_particles, pressure=pressure)
    )
    results = analyzer.run_analysis()

    # RESULTS_PATH = f"{RUN_DIR}/measurements.pkl"
    RESULTS_PATH = job.fn("measurements.pkl")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    Project().main()