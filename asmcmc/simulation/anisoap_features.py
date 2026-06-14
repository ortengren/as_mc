"""AniSOAP feature generation for MC trajectories -- preserved, not wired in.

This is the AniSOAP power-spectrum code that used to live in ``asmcmc/driver.py``.
It is kept here for later coarse-grained-representation work but is intentionally
*not* part of the standard simulation run path (:mod:`asmcmc.simulation.run`).
Import and call it explicitly when you need power-spectrum features for a run.
``asmcmc/generate_cg_reps.py`` is the related, separate rep-generation entry point.
"""

import numpy as np
import matplotlib.pyplot as plt
import metatensor
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
from skmatter.preprocessing import StandardFlexibleScaler


DEFAULT_HPARAMS = {
    "max_angular": 9,
    "max_radial": 6,
    "radial_basis_name": "gto",
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "subtract_center_contribution": True,
    "radial_gaussian_width": 1.5,
    "cutoff_radius": 8.0,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-4,
}


def generate_ps(features, simulation_id=None):
    """Build a scaled power spectrum from AniSOAP density-expansion features.

    Combines the nu=1 features with themselves via Clebsch-Gordan coefficients
    (lcut=0), averages over centers, and standard-scales the result. When
    ``simulation_id`` is given, saves the array to
    ``results/simulations/{simulation_id}/power_spectrum.npy``.
    """
    print("combining Clebsch Gordan coefficients...")
    mycg = ClebschGordanReal(DEFAULT_HPARAMS["max_angular"])
    aniso_nu1 = standardize_keys(features)
    aniso_nu2 = cg_combine(
        aniso_nu1,
        aniso_nu1,
        clebsch_gordan=mycg,
        lcut=0,  # TODO: justify this cutoff
        other_keys_match=["types_center"],
    )
    rep = metatensor.operations.mean_over_samples(aniso_nu2, sample_names="center")
    x_raw = rep.block().values.squeeze()
    print("scaling result")
    x_scaler = StandardFlexibleScaler(column_wise=False).fit(x_raw)
    x = x_scaler.transform(x_raw)
    if simulation_id is not None:
        np.save(f"results/simulations/{simulation_id}/power_spectrum.npy", x)
    return x


def plot_power_spectrum(x, simulation_id):
    """Save a quick line plot of the power spectrum for a given simulation id."""
    plt.plot(x)
    plt.savefig(f"results/simulations/{simulation_id}/power_spectrum.png")
