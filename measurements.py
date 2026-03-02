import ase
from ase.db import connect
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from potentials import calc_total_energy


BOLTZCONST = 8.617E-5 # eV / K


class Measurement(ABC):
    def __init__(self):
        self.results = []

    @abstractmethod
    def compute(self, frame, scalar_data, array_data):
        """Calculate metric for a single frame"""
        pass

    @abstractmethod
    def finalize(self):
        """Aggregate data across frames"""
        pass


class AverageEnergy(Measurement):
    def __init__(self):
        super().__init__()
        self.energies = []

    def compute(self, frame, scalar_data, array_data):
        energy = scalar_data["total_energy"]
        self.energies.append(energy)

    def finalize(self):
        avg_e = np.mean(self.energies)
        var_e = np.var(self.energies)
        return avg_e, var_e


class RadialDistributionFunction(Measurement):
    def __init__(self, r_max, num_bins):
        super().__init__()
        self.r_max = r_max
        self.num_bins = num_bins

        self.bin_edges = np.linspace(0, r_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.

        self.hist_counts = np.zeros(num_bins)
        self.num_frames = 0
        self.total_volume = 0.
        self.total_particles = 0

    def compute(self, frame, scalar_data, array_data):
        # update running totals
        self.num_frames += 1
        self.total_particles += len(frame)
        self.total_volume += frame.get_volume()

        # get distances between unique pairs
        dist_matrix = frame.get_all_distances(mic=True)
        i, j = np.triu_indices(len(frame), k=1)
        unique_distances = dist_matrix[i, j]

        # update histogram
        counts, _ = np.histogram(unique_distances, bins=self.bin_edges)
        self.hist_counts += counts

    def finalize(self):
        avg_vol = self.total_volume / self.num_frames
        avg_particles = self.total_particles / self.num_frames
        density = avg_particles / avg_vol

        # get volumes of each spherical shell
        r_inner = self.bin_edges[:-1]
        r_outer = self.bin_edges[1:]
        shell_vols = (4. / 3.) * np.pi * (r_outer**3 - r_inner**3)

        # calculate expected counts for ideal gas
        ideal_counts = (avg_particles / 2.) * self.num_frames * density * shell_vols

        # normalize
        g_r = np.zeros_like(self.hist_counts)
        np.divide(self.hist_counts, ideal_counts, out=g_r, where=(ideal_counts > 0))

        return {
            "r": self.bin_centers,
            "g_r": g_r,
        }


class OrientationalCorrelationFunction(Measurement):
    def __init__(self, r_max, num_bins):
        super().__init__()
        self.r_max = r_max
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0, r_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.

        self.hist_counts = np.zeros(num_bins)
        self.sum_p2 = np.zeros(num_bins)

    def compute(self, frame, scalar_data, array_data):
        num_particles = len(frame)

        # get distances
        dist_matrix = frame.get_all_distances(mic=True)

        # get all dot products in a shape (N,N) array
        or_vecs = array_data["or_vec"]
        dot_matrix = or_vecs @ or_vecs.T

        # get only unique pairs
        i, j = np.triu_indices(num_particles, k=1)
        dists = dist_matrix[i, j]
        cos_thetas = dot_matrix[i, j]

        # calculate P2 (second Legendre polynomial) for all pairs
        p2_vals = 0.5 * (3. * cos_thetas**2 - 1.)

        # build histogram for distances
        counts, _ = np.histogram(dists, bins=self.bin_edges)
        self.hist_counts += counts

        # build histogram for P2 values
        p2_sums, _ = np.histogram(dists, bins=self.bin_edges, weights=p2_vals)
        self.sum_p2 += p2_sums

    def finalize(self):
        # calculate average P2 value for each distance bin
        avg_p2 = np.zeros_like(self.sum_p2)
        np.divide(self.sum_p2, self.hist_counts, out=avg_p2, where=(self.hist_counts > 0))

        return {
            "r": self.bin_centers,
            "s2_r": avg_p2,
        }


class HeatCapacity(Measurement):
    def __init__(self, temperature, num_particles):
        super().__init__()
        self.energies = []
        self.temp = temperature
        self.num_particles = num_particles

    def compute(self, frame, scalar_data, array_data):
        self.energies.append(scalar_data["total_energy"])

    def finalize(self):
        e_var = np.var(self.energies)
        cv_ideal = 3 * BOLTZCONST
        cv_excess = e_var / (BOLTZCONST * self.num_particles * self.temp**2)
        return cv_ideal + cv_excess


class TrajectoryAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.measurements = {}

    def add_measurement(self, name, measurement_obj):
        self.measurements[name] = measurement_obj

    def run_analysis(self):
        with connect(self.db_path) as db:
            total_frames = db.count()

            with tqdm(total=total_frames, desc="Analyzing Trajectory") as pbar:
                for row in db.select():
                    frame = row.toatoms()
                    scalar_data = row.key_value_pairs
                    array_data = row.data

                    for meas in self.measurements.values():
                        meas.compute(frame, scalar_data, array_data)

                    pbar.update(1)

            final_results = {}
            for name, meas in self.measurements.items():
                final_results[name] = meas.finalize()

        return final_results