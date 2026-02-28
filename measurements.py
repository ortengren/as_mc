import ase
import numpy as np
from abc import ABC, abstractmethod
from potentials import calc_total_energy


class Measurement(ABC):
    def __init__(self):
        self.results = []

    @abstractmethod
    def compute(self, frame):
        """Calculate metric for a single frame"""
        pass

    @abstractmethod
    def finalize(self):
        """Aggregate data across frames"""


class GayBerneQuadrupoleEnergy(Measurement):
    def __init__(self, nl_cutoff=15.):
        super().__init__()
        self.energies = []
        self.cutoff = nl_cutoff

    def compute(self, frame):
        energy = calc_total_energy(frame, self.cutoff)
        self.energies.append(energy)

    def finalize(self):
        avg_e = np.mean(self.energies)
        var_e = np.var(self.energies)
        return avg_e, var_e


class RadialDistributionFunction(Measurement):
    def __init__(self):
        super().__init__()

    def compute(self, frame):
        return NotImplementedError

    def finalize(self):
        return NotImplementedError


class OrientationalCorrelationFunction(Measurement):



def get_separation(frame, idx_1, idx_2, vector=False):
    return frame.get_distance(idx_1, idx_2, mic=True, vector=vector)  # units of Å


def get_relative_angle(frame, idx_1, idx_2):
    v1 = frame.arrays["or_vec"][idx_1]
    v2 = frame.arrays["or_vec"][idx_2]
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))