import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from asmcmc.trial_moves import (
    calc_or_vec,
    calculate_com_move,
    calculate_quat_move,
    calculate_vol_move,
    quaternion_multiply,
)


# --- calc_or_vec ---

def test_calc_or_vec_identity(identity_quat):
    """Identity quaternion should leave the z-axis unchanged."""
    result = calc_or_vec(identity_quat).squeeze()
    np.testing.assert_allclose(result, [0., 0., 1.], atol=1e-10)


def test_calc_or_vec_unit_length():
    """Output orientation vector should be a unit vector."""
    quat = np.array([0.5, 0.5, 0.5, 0.5])
    result = calc_or_vec(quat).squeeze()
    assert abs(np.linalg.norm(result) - 1.0) < 1e-10


# --- calculate_com_move ---

def test_com_move_bounded():
    """Displacement in each dimension must be within [-delta/2, delta/2]."""
    r = np.array([5., 5., 5.])
    delta = 0.4
    for _ in range(200):
        new_r = calculate_com_move(r, delta)
        diff = new_r - r
        assert np.all(np.abs(diff) <= delta / 2 + 1e-12)


def test_com_move_changes_position():
    """Returned position should differ from input (with delta > 0)."""
    r = np.array([0., 0., 0.])
    new_r = calculate_com_move(r, delta=1.0)
    assert not np.allclose(r, new_r)


# --- quaternion_multiply ---

def test_quaternion_multiply_identity(identity_quat):
    """q * identity = q for any quaternion."""
    q = np.array([0.5, 0.5, 0.5, 0.5])
    result = quaternion_multiply(q, identity_quat)
    np.testing.assert_allclose(result, q, atol=1e-12)


def test_quaternion_multiply_inverse(identity_quat):
    """q * conj(q) should equal the identity quaternion."""
    q = np.array([0.5, 0.5, 0.5, 0.5])  # already unit
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quaternion_multiply(q, q_conj)
    np.testing.assert_allclose(result, identity_quat, atol=1e-12)


def test_quaternion_multiply_known():
    """i * j = k in Hamilton product (scalar-first: [w,x,y,z])."""
    i = np.array([0., 1., 0., 0.])
    j = np.array([0., 0., 1., 0.])
    k = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(quaternion_multiply(i, j), k, atol=1e-12)


# --- calculate_quat_move ---

def test_quat_move_unit_norm(identity_quat):
    """Output quaternion must have unit norm."""
    for _ in range(200):
        result = calculate_quat_move(identity_quat, delta=0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10


def test_quat_move_zero_delta(identity_quat):
    """delta=0 means zero rotation angle; output should equal input."""
    result = calculate_quat_move(identity_quat, delta=0.0)
    np.testing.assert_allclose(np.abs(result), np.abs(identity_quat), atol=1e-10)


# --- calculate_vol_move ---

def test_vol_move_scale_factor_bounded():
    """s_v = det(new_cell)/det(old_cell) must be in [1-delta, 1+delta]."""
    cell = np.diag([10., 10., 10.])
    old_vol = np.linalg.det(cell)
    delta = 0.1
    for _ in range(200):
        new_cell, _ = calculate_vol_move(cell, old_vol, delta)
        s_v = np.linalg.det(new_cell) / old_vol
        assert np.exp(-delta) - 1e-12 <= s_v <= np.exp(delta) + 1e-12


def test_vol_move_log_uniform_and_symmetric():
    """ln(V'/V) is uniform on [-delta, delta] and symmetric about 0.

    Symmetry in ln(V) is the detailed-balance requirement for the (N+1)*ln(V'/V)
    acceptance criterion in npt_decide_accept; an asymmetric proposal (e.g. the old
    uniform-in-V move) would need a Hastings correction and biases the volume.
    """
    cell = np.diag([10., 10., 10.])
    old_vol = np.linalg.det(cell)
    delta = 0.3
    log_ratios = np.array([
        np.log(calculate_vol_move(cell, old_vol, delta)[1] / old_vol)
        for _ in range(20000)
    ])
    assert np.all(np.abs(log_ratios) <= delta + 1e-12)
    assert abs(log_ratios.mean()) < 0.02          # symmetric about 0
    assert log_ratios.min() < -0.9 * delta        # spans the full window
    assert log_ratios.max() > 0.9 * delta


def test_vol_move_volume_consistent():
    """Returned new_vol must equal det(new_cell)."""
    cell = np.diag([10., 10., 10.])
    old_vol = np.linalg.det(cell)
    new_cell, new_vol = calculate_vol_move(cell, old_vol, delta=0.1)
    np.testing.assert_allclose(new_vol, np.linalg.det(new_cell), rtol=1e-10)


def test_vol_move_positive_scale_large_delta():
    """s_v must be positive even when delta >= 1 (vol_delt can grow this large)."""
    cell = np.diag([10., 10., 10.])
    old_vol = np.linalg.det(cell)
    for _ in range(500):
        new_cell, new_vol = calculate_vol_move(cell, old_vol, delta=1.5)
        assert new_vol > 0, f"new_vol={new_vol} is not positive"


def test_vol_move_real_cell_large_delta():
    """Cell entries must be real-valued even when delta >= 1."""
    cell = np.diag([10., 10., 10.])
    old_vol = np.linalg.det(cell)
    for _ in range(500):
        new_cell, _ = calculate_vol_move(cell, old_vol, delta=1.5)
        assert np.isrealobj(new_cell), "Cell has complex entries"
        assert np.all(np.isfinite(new_cell)), "Cell has non-finite entries"
