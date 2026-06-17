import os
import sys
import csv
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

# Cap each worker's BLAS/threadpool to one thread BEFORE numpy loads: the grid
# is parallelised across processes, so letting each also spin up N math threads
# would interfere with the parallelisation.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg")  # batch run: write figures to file, never open a window
import matplotlib.pyplot as plt

from asmcmc.initialize import RandomLatticeInitializer
from asmcmc.metropolis import MetropolisCalculator, BOLTZCONST
from asmcmc.potentials import GB_PARAMS
from asmcmc.measurements import TrajectoryAnalyzer, AverageEnergy, NematicOrderParameter


def equilibrate_point(
    temp,
    pressure,
    num_eq_steps,
    initializer,
    block_size,
    seed,
    scratch_dir,
    buffer_size=50,
):
    random.seed(seed)
    np.random.seed(seed)

    metro = MetropolisCalculator(
        temp=temp,
        pressure=pressure,
        initializer=initializer,
        output_dir=scratch_dir,
    )
