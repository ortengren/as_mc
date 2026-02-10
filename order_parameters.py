import ase
import numpy as np


def get_separation(frame, idx_1, idx_2, vector=False):
    return frame.get_distance(idx_1, idx_2, mic=True, vector=vector)  # units of Å


def get_relative_angle(frame, idx_1, idx_2):
    v1 = frame.arrays["or_vec"][idx_1]
    v2 = frame.arrays["or_vec"][idx_2]
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))