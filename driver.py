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


def run_simulation(simulation_id=None):
    frames = ase.io.read("xyz_files/two_benzenes.xyz", ":")
    ell_frames = ase.io.read("xyz_files/two_ells_with_axes.xyz", ":")
    print("frames length: ", len(frames))
    print("ell_frames length: ", len(ell_frames))

    for ell_frame, frame in zip(ell_frames, frames):
        ell_frame.info["energy"] = frame.get_total_energy()

    metro = MetropolisCalculator(ell_frames[0], energy_func="GB")
    metro.calculate_trajectory(10_000)
    # save MetropolisCalculator object to file
    if simulation_id is not None:
        with open(f"simulations/{simulation_id}/MetropolisCalculator.pkl", "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(metro, f)
    print("Simulation Completed")
    return metro


def analyze_trajectory(metro, simulation_id=None):
    for frame in metro.frames:
        axes = np.array(frame.arrays["axes"])
        frame.arrays[r"c_diameter[1]"] = axes[:, 0]
        frame.arrays[r"c_diameter[2]"] = axes[:, 1]
        frame.arrays[r"c_diameter[3]"] = axes[:, 2]

    calculator = EllipsoidalDensityProjection(**DEFAULT_HPARAMS)
    print("calculator initiated")
    print("calculating rep_raw...")
    rep_raw = calculator.transform(metro.frames, show_progress=True)
    print("calculated rep_raw")

    if simulation_id is not None:
        metatensor.save(f"simulations/{simulation_id}/feature_tm.npz", rep_raw)
    return rep_raw


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


def calculate_all(simulation_id):
    metro = run_simulation(simulation_id)
    density_proj = analyze_trajectory(metro, simulation_id)
    X = generate_ps(density_proj, simulation_id)
    return metro, density_proj, X


def main():
    sim_id = generate_simulation_id()
    metro, density_proj, X = calculate_all(sim_id)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[1000:, 0], X_pca[1000:, 1])
    plt.savefig(f"simulations/{sim_id}/power_spectrum_pca.png")


if __name__ == "__main__":
    main()
