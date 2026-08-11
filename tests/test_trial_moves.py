import numpy as np
import pytest
from asmcmc.base.trial_moves import (
    calc_or_vec,
    calculate_com_move,
    calculate_quat_move,
    calculate_vol_move,
    calculate_aniso_vol_move,
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


# --- calculate_aniso_vol_move ---

def test_aniso_vol_move_scale_factor_bounded():
    """V'/V = s must lie in [exp(-delta), exp(delta)] (a single-axis log scale)."""
    cell = np.diag([10., 12., 15.])  # non-cubic, so axes are distinguishable
    old_vol = np.linalg.det(cell)
    delta = 0.1
    for _ in range(300):
        _, new_vol = calculate_aniso_vol_move(cell, old_vol, delta)
        s_v = new_vol / old_vol
        assert np.exp(-delta) - 1e-12 <= s_v <= np.exp(delta) + 1e-12


def test_aniso_vol_move_changes_one_axis_only():
    """Exactly one lattice vector is rescaled (by V'/V, since only it moves); the
    other two are left untouched."""
    cell = np.diag([10., 12., 15.])
    old_vol = np.linalg.det(cell)
    for _ in range(300):
        new_cell, new_vol = calculate_aniso_vol_move(cell, old_vol, 0.3)
        changed = np.any(new_cell != np.asarray(cell), axis=1)  # per lattice vector
        assert changed.sum() == 1, "more than one axis moved"
        axis = int(np.argmax(changed))
        s = new_vol / old_vol  # one axis ⇒ the volume ratio *is* that axis's scale
        np.testing.assert_allclose(new_cell[axis], np.asarray(cell)[axis] * s, rtol=1e-10)
        for other in range(3):
            if other != axis:
                np.testing.assert_array_equal(new_cell[other], np.asarray(cell)[other])


def test_aniso_vol_move_stays_orthorhombic():
    """A diagonal (orthorhombic) cell stays diagonal — a box length changes but no
    shear is introduced."""
    cell = np.diag([10., 12., 15.])
    old_vol = np.linalg.det(cell)
    for _ in range(200):
        new_cell, _ = calculate_aniso_vol_move(cell, old_vol, 0.5)
        off_diag = new_cell[~np.eye(3, dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)


def test_aniso_vol_move_log_uniform_and_symmetric():
    """ln(V'/V) is uniform on [-delta, delta] and symmetric about 0 — the same
    detailed-balance requirement the (N+1)*ln(V'/V) criterion places on the
    isotropic move (see test_vol_move_log_uniform_and_symmetric)."""
    cell = np.diag([10., 12., 15.])
    old_vol = np.linalg.det(cell)
    delta = 0.3
    log_ratios = np.array([
        np.log(calculate_aniso_vol_move(cell, old_vol, delta)[1] / old_vol)
        for _ in range(20000)
    ])
    assert np.all(np.abs(log_ratios) <= delta + 1e-12)
    assert abs(log_ratios.mean()) < 0.02          # symmetric about 0
    assert log_ratios.min() < -0.9 * delta        # spans the full window
    assert log_ratios.max() > 0.9 * delta


def test_aniso_vol_move_volume_consistent():
    """Returned new_vol must equal det(new_cell)."""
    cell = np.diag([10., 12., 15.])
    old_vol = np.linalg.det(cell)
    for _ in range(200):
        new_cell, new_vol = calculate_aniso_vol_move(cell, old_vol, 0.2)
        np.testing.assert_allclose(new_vol, np.linalg.det(new_cell), rtol=1e-10)


def test_aniso_vol_move_does_not_mutate_input():
    """The caller's cell array must be left unchanged (the move returns a copy)."""
    cell = np.diag([10., 12., 15.])
    cell_before = cell.copy()
    calculate_aniso_vol_move(cell, np.linalg.det(cell), 0.5)
    np.testing.assert_array_equal(cell, cell_before)


def test_aniso_vol_move_positive_real_large_delta():
    """s > 0 and the cell stays real/finite even when delta >= 1."""
    cell = np.diag([10., 12., 15.])
    old_vol = np.linalg.det(cell)
    for _ in range(300):
        new_cell, new_vol = calculate_aniso_vol_move(cell, old_vol, 1.5)
        assert new_vol > 0, f"new_vol={new_vol} is not positive"
        assert np.isrealobj(new_cell) and np.all(np.isfinite(new_cell))
