import numpy as np
import random as rand
from scipy.spatial.transform import Rotation


def calc_or_vec(quat):
    R = Rotation.from_quat(np.roll(quat, -1))
    or_vec = R.as_matrix() @ [[0], [0], [1]]
    return or_vec


def calculate_com_move(r, delta):
    # generate random displacement (x_1, x_2, x_3) such that |x_i| <= delta / 2
    displacement = [rand.uniform(-delta / 2, delta / 2) for _ in range(3)]
    return r + displacement


def quaternion_multiply(q1, q2):
    """
    Computes the Hamilton product of two quaternions.
    Assumes scalar-first convention: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def calculate_quat_move(quat, delta):
    # generate a random rotation angle uniformly in [-delta, delta]
    theta = rand.uniform(-delta, delta)
    half_theta = theta / 2.0
    sin_half_theta = np.sin(half_theta)
    # generate a uniformly distributed random 3D unit axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # construct the perturbation quaternion
    dq = np.array([
        np.cos(half_theta),
        axis[0] * sin_half_theta,
        axis[1] * sin_half_theta,
        axis[2] * sin_half_theta
    ])
    # apply the rotation via quaternion multiplication
    new_quat = quaternion_multiply(dq, quat)
    # re-normalize to prevent floating-point drift
    new_quat /= np.linalg.norm(new_quat)
    return new_quat


def calculate_vol_move(cell, curr_vol, delta):
    # calculate random volume scaling factor uniformly in [1 - delta, 1 + delta]
    s_v = 1 + rand.uniform(-delta, delta)
    s_v = max(s_v, 1e-8)  # prevent complex cube root if vol_delt exceeds 1
    # calculate amount to scale cell axes by
    s_l = s_v**(1/3)
    new_cell = s_l * cell
    new_vol = curr_vol * s_v
    return new_cell, new_vol
