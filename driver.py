import ase.io
from metropolis import MetropolisCalculator
import ase
import pickle
import datetime


DEFAULT_HPARAMS = {
    "max_angular": 9,
    "max_radial": 6,
    "radial_basis_name": "gto",
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "subtract_center_contribution": True,
    "radial_gaussian_width": 1.5,
    "cutoff_radius": 8,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-4,
}


def run_simulation():
    frames = ase.io.read("two_benzenes.xyz", ":")
    ell_frames = ase.io.read("two_ells_with_axes.xyz", ":")

    for ell_frame, frame in zip(ell_frames, frames):
        ell_frame.info["energy"] = frame.get_total_energy()

    metro = MetropolisCalculator(ell_frames[0], ell_frames[0].info["energy"], energy_func="GB")
    metro.calculate_trajectory(10_000)
    # save MetropolisCalculator object to file
    dt = datetime.datetime.today().isoformat(timespec="minutes")
    with open(f"metro_{dt}.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(metro, f)
    print("done")


def analyze_trajectory(filename):
    with open(filename, "rb") as f:
        metro = pickle.load(f)
    traj_frames: MetropolisCalculator = metro.frames



def main():
    pass

if __name__ == "__main__":
    main()
