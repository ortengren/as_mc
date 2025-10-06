import numpy as np
import random as rand


def pick_particle(frame):
    num_particles = len(frame)
    rand_idx = rand.randint(0, num_particles - 1)
    return rand_idx


def nudge_com(frame, particle_idx, delta, copy=True):
    if copy:
        nframe = frame.copy()
    else:
        nframe = frame
    displacement = [rand.uniform(-delta/2, delta/2) for _ in range(3)]
    nframe[particle_idx].position += displacement
    nframe.wrap()
    return nframe


def nudge_orientation(frame, particle_idx, delta, copy=True, quat_key="c_q"):
    if copy:
        nframe = frame.copy()
    else:
        nframe = frame
    displacement = get_rand_unit_quat()
    new_orientation = nframe.arrays[quat_key][particle_idx] + (displacement * delta)
    new_mag = np.linalg.norm(new_orientation)
    new_orientation /= new_mag
    nframe.arrays[quat_key][particle_idx] = new_orientation
    nframe.wrap()
    return nframe


def get_rand_unit_quat():
    """
    Method due to Marsaglia, Ann. Math. Stat 43 (2) (1972), 645
    """
    v1 = -1
    v2 = -1
    while v1**2 + v2**2 >= 1:
        v1 = rand.uniform(-1,1)
        v2 = rand.uniform(-1,1)
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


def simultaneous_move(frame, pos_delt, or_delt):
    rand_idx = pick_particle(frame)
    nframe = nudge_com(frame, rand_idx, pos_delt, copy=True)
    nframe = nudge_orientation(nframe, rand_idx, or_delt, copy=False)
    return nframe

