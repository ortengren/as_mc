"""Trial moves: particle translations and rotations, and volume moves.

Widths differ in meaning: ``pos_delt`` is a full width, while ``or_delt`` and
``vol_delt`` are half-widths.
"""

import numpy as np
import random as rand
from scipy.spatial.transform import Rotation


def quat_to_or_vec(quats):
    """Disc normals (the body z-axis) of scalar-first ``(w, x, y, z)`` quaternions.

    Takes one quaternion ``(4,)`` or a stack ``(n, 4)``; returns ``(3,)`` or ``(n, 3)``.
    """
    quats = np.asarray(quats, dtype=float)
    # scipy wants scalar-last (x, y, z, w); column 2 of R is R @ z-hat.
    return Rotation.from_quat(np.roll(quats, -1, axis=-1)).as_matrix()[..., :, 2]


def calculate_com_move(r, delta):
    """``r`` displaced by a uniform draw from the cube [-delta/2, delta/2]^3."""
    displacement = [rand.uniform(-delta / 2, delta / 2) for _ in range(3)]
    return r + displacement


def quaternion_multiply(q1, q2):
    """
    Computes the Hamilton product of two quaternions.
    Assumes scalar-first convention: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def calculate_quat_move(quat, delta):
    """``quat`` rotated by an angle drawn from [-delta, delta] (radians) about a
    uniformly random axis."""
    theta = rand.uniform(-delta, delta)
    half_theta = theta / 2.0
    sin_half_theta = np.sin(half_theta)
    # generate a uniformly distributed random 3D unit axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # construct the perturbation quaternion
    dq = np.array(
        [
            np.cos(half_theta),
            axis[0] * sin_half_theta,
            axis[1] * sin_half_theta,
            axis[2] * sin_half_theta,
        ]
    )
    # apply the rotation via quaternion multiplication
    new_quat = quaternion_multiply(dq, quat)
    # re-normalize to prevent floating-point drift
    new_quat /= np.linalg.norm(new_quat)
    return new_quat


def calculate_vol_move(cell, curr_vol, delta):
    """Isotropic volume move: every axis scaled by s_v**(1/3), ln(s_v) ~ U(-delta, delta)."""
    # log-uniform volume scaling: ln(s_v) ~ U(-delta, delta), so the proposal is
    # symmetric in ln(V). This matches the (N+1)*ln(V'/V) term in npt_decide_accept
    # (which is derived for ln-volume sampling) and keeps s_v > 0 for any delta.
    s_v = np.exp(rand.uniform(-delta, delta))
    # calculate amount to scale cell axes by
    s_l = s_v ** (1 / 3)
    new_cell = s_l * cell
    new_vol = curr_vol * s_v
    return new_cell, new_vol


def calculate_aniso_vol_move(cell, curr_vol, delta):
    """Anisotropic log-uniform volume move: rescale a single, randomly chosen
    lattice vector by s = exp(U(-delta, delta)), leaving the other two axes fixed.

    Where ``calculate_vol_move`` scales all three axes by the same s_v**(1/3) (an
    isotropic *size* change), this changes one box length at a time, so the box can
    relax its *aspect ratio* over many moves — needed to reach the ordered
    anisotropic (columnar / nematic) phases these oblate particles form, which an
    isotropic move can never reach from a differently-shaped start.

    Only one axis scales, so V'/V = s exactly, and the (N+1)*ln(V'/V) term in
    ``npt_decide_accept`` is therefore unchanged — it depends only on the total
    volume ratio, not on which or how many axes moved. Sampling ln(s) ~
    U(-delta, delta) keeps the proposal symmetric in ln(V) (the detailed-balance
    requirement for that criterion) and s > 0 for any delta.
    """
    s = np.exp(rand.uniform(-delta, delta))
    axis = rand.randrange(3)
    # np.array(..., dtype=float) copies (so the caller's cell is untouched) and
    # coerces an ASE Cell to a plain 3x3 ndarray; scaling row ``axis`` stretches
    # that lattice vector, keeping an orthorhombic cell orthorhombic.
    new_cell = np.array(cell, dtype=float)
    new_cell[axis] *= s
    return new_cell, curr_vol * s
