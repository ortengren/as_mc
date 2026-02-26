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


def calculate_quat_move(quat, delta):
    # generate random orientational displacement
    displacement = get_rand_unit_quat()
    # add displacement to particle orientation and normalize the result
    new_orientation = quat + (displacement * delta)
    new_mag = np.linalg.norm(new_orientation)
    new_orientation /= new_mag
    return new_orientation


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

