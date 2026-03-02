import numpy as np
import ase.io
from metropolis import MetropolisCalculator
import ase
import pickle
import datetime
import os
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
import metatensor
from skmatter.preprocessing import StandardFlexibleScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


BOLTZCONST = 8.617E-5 # eV / K

TARGET_ACC_RATE = 0.275

DEFAULT_HPARAMS = {
    "max_angular": 9,
    "max_radial": 6,
    "radial_basis_name": "gto",
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "subtract_center_contribution": True,
    "radial_gaussian_width": 1.5,
    "cutoff_radius": 8.,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-4,
}


def generate_simulation_id(method="datetime"):
    if method == "datetime":
        dt = datetime.datetime.today().isoformat(timespec="minutes")
        if not os.path.exists(f"simulations/{dt}"):
            os.makedirs(f"simulations/{dt}")
        return dt
    else:
        return NotImplementedError


def generate_ps(features, simulation_id=None):
    print("combining Clebsch Gordan coefficients...")
    mycg = ClebschGordanReal(DEFAULT_HPARAMS["max_angular"])
    aniso_nu1 = standardize_keys(features)
    aniso_nu2 = cg_combine(
        aniso_nu1,
        aniso_nu1,
        clebsch_gordan=mycg,
        lcut=0, # TODO: justify this cutoff
        other_keys_match=["types_center"]
    )
    rep = metatensor.operations.mean_over_samples(aniso_nu2, sample_names="center")
    x_raw = rep.block().values.squeeze()
    print("scaling result")
    x_scaler = StandardFlexibleScaler(column_wise=False).fit(x_raw)
    x = x_scaler.transform(x_raw)
    if simulation_id is not None:
        np.save(f"simulations/{simulation_id}/power_spectrum.npy", x)
    return x


def plot_power_spectrum(x, simulation_id):
    plt.plot(x)
    plt.savefig(f"simulations/{simulation_id}/power_spectrum.png")


def main():
    # create directory for simulation
    sim_id_base = "multi_temp_trial"
    temps = [50., 100., 150., 200.]
    frame = ase.io.read("xyz_files/duped_ellipsoids_with_axes.xyz")
    for temp in temps:
        metro = MetropolisCalculator(
            frame,
            energy_func="GB",
            output_dir=f"{sim_id_base}/{temp}"
        )
        metro.calculate_trajectory(
            100_000,
            temp,
            block_size=400,
            num_eq_steps=100_000,
            buffer_size=50
        )


if __name__ == "__main__":
    main()
