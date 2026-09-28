"""Rotation and radial-sampling primitives used by cluster construction.

No MLIP anywhere.
"""

import numpy as np
import pytest

from asmcmc.data_preparation.proposals import (
    orthonormal_basis,
    rotation_to_normal,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260811)


# --- rotations ---------------------------------------------------------------


def test_orthonormal_basis_is_orthonormal(rng):
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    e1, e2 = orthonormal_basis(u)

    for a, b in [(e1, e1), (e2, e2)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 1.0)
    for a, b in [(e1, e2), (e1, u), (e2, u)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 0.0, atol=1e-12)


def test_rotation_carries_the_reference_normal_onto_the_target(rng):
    targets = rng.normal(size=(300, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    spin = rng.uniform(0, 2 * np.pi, size=300)

    rotations = rotation_to_normal(targets, spin)
    mapped = np.einsum("nij,j->ni", rotations, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(mapped, targets, atol=1e-10)
    assert np.allclose(np.linalg.det(rotations), 1.0)


def test_rotation_handles_the_antipodal_target():
    """cos = -1 is where the Rodrigues 1/(1+cos) form blows up."""
    rotations = rotation_to_normal(np.array([[0.0, 0.0, -1.0]]), np.array([0.0]))
    mapped = rotations[0] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(mapped, [0.0, 0.0, -1.0], atol=1e-10)
    assert np.linalg.det(rotations[0]) == pytest.approx(1.0)


def test_spin_leaves_the_normal_alone_but_moves_the_molecule():
    """Spin is a real atomistic degree of freedom the CG coordinates cannot see."""
    from asmcmc.data_preparation.cluster_dataset import build_reference_benzene

    target = np.array([[0.0, 0.0, 1.0]])
    a = rotation_to_normal(target, np.array([0.0]))[0]
    b = rotation_to_normal(target, np.array([0.7]))[0]

    positions = build_reference_benzene().get_positions()
    assert not np.allclose(positions @ a.T, positions @ b.T)
    assert np.allclose(a @ np.array([0, 0, 1.0]), b @ np.array([0, 0, 1.0]), atol=1e-12)
