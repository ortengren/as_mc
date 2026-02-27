import numpy as np
import random as rand
from scipy.spatial.transform import Rotation


def calc_or_vec(quat):
    R = Rotation.from_quat(np.roll(quat, -1))
    or_vec = R.as_matrix() @ [[0], [0], [1]]
    return or_vec


def nudge_com(frame, particle_idx, delta, copy=True):
    if copy:
        nframe = frame.copy()
    else:
        nframe = frame
    # generate random displacement (x_1, x_2, x_3) such that |x_i| <= delta / 2
    displacement = [rand.uniform(-delta/2, delta/2) for _ in range(3)]
    # add displacement to particle position
    nframe[particle_idx].position += displacement
    nframe.wrap()
    return nframe


def calculate_com_move(r, delta):
    # generate random displacement (x_1, x_2, x_3) such that |x_i| <= delta / 2
    displacement = [rand.uniform(-delta / 2, delta / 2) for _ in range(3)]
    return r + displacement


def nudge_orientation(frame, particle_idx, delta, copy=True, quat_key="c_q"):
    if copy:
        nframe = frame.copy()
    else:
        nframe = frame
    # generate random orientational displacement
    displacement = get_rand_unit_quat()
    # add displacement to particle orientation and normalize the result
    new_orientation = nframe.arrays[quat_key][particle_idx] + (displacement * delta)
    new_mag = np.linalg.norm(new_orientation)
    new_orientation /= new_mag
    nframe.arrays[quat_key][particle_idx] = new_orientation
    nframe.arrays["or_vec"][particle_idx] = calc_or_vec(nframe, particle_idx, quat_key).flatten()
    nframe.wrap()
    return nframe


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


def get_rand_unit_quat():
    """
    Method due to Marsaglia, Ann. Math. Stat 43 (2) (1972), 645
    """
    v1 = -1
    v2 = -1
    while v1**2 + v2**2 >= 1:
        v1 = rand.uniform(-1, 1)
        v2 = rand.uniform(-1, 1)
    s1 = v1**2 + v2**2

    v3 = -1
    v4 = -1
    while v3**2 + v4**2 >= 1:
        v3 = rand.uniform(-1,1)
        v4 = rand.uniform(-1,1)
    s2 = v3**2 + v4**2

    factor = np.sqrt((1-s1)/s2)
    quat = [v1, v2, v3*factor, v4*factor]
    return np.array(quat)


def simultaneous_move(frame, idx, pos_delt, or_delt):
    nframe = nudge_com(frame, idx, pos_delt, copy=True)
    nframe = nudge_orientation(nframe, idx, or_delt, copy=False)
    return nframe

