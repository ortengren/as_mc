import numpy as np
import ase
import pytest


@pytest.fixture
def two_particle_frame():
    """Two particles 10 Å apart along x, both oriented along z."""
    positions = np.array([[0., 0., 0.], [10., 0., 0.]])
    cell = np.diag([60., 60., 60.])
    frame = ase.Atoms(symbols="HH", positions=positions, cell=cell, pbc=True)
    # identity quaternions [w, x, y, z]
    frame.new_array("c_q",    np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
    frame.new_array("or_vec", np.array([[0., 0., 1.], [0., 0., 1.]]))
    return frame


@pytest.fixture
def identity_quat():
    return np.array([1., 0., 0., 0.])
