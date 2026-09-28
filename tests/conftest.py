import random

import numpy as np
import ase
import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make stochastic tests deterministic.

    The MC sampler draws from the global ``random`` and ``numpy.random``
    streams; without seeding these are initialised from OS entropy, which
    makes energy-tracking and acceptance-rate assertions intermittently flaky.
    """
    random.seed(0)
    np.random.seed(0)


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


class StubCalculator:
    """A cheap stand-in for UMA: energy = -(number of atoms)/10, zero forces.

    Deliberately *not* physical. These tests check the generator's plumbing --
    decomposition algebra, sharding, resume -- and a stub makes the expected
    numbers exact instead of approximate.
    """

    def __init__(self):
        self.n_calls = 0

    def get_potential_energy(self, atoms=None):
        self.n_calls += 1
        return -0.1 * len(atoms)

    def get_forces(self, atoms=None):
        return np.zeros((len(atoms), 3))

    # ASE calls into the calculator through these on an attached Atoms.
    def calculate(self, *a, **k):
        pass

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name == "energy":
            return self.get_potential_energy(atoms)
        if name == "forces":
            return self.get_forces(atoms)
        raise NotImplementedError(name)

    def check_state(self, atoms):
        return []

    def get_stress(self, atoms=None):
        raise NotImplementedError


@pytest.fixture
def make_stub_calculator():
    """Factory for fresh ``StubCalculator`` instances (each counts its own calls)."""
    return StubCalculator


@pytest.fixture
def stub_uma(monkeypatch):
    """Replace UMA with ``StubCalculator`` wherever the dataset generator loads it."""
    import asmcmc.delta_learning.dataset as dataset

    monkeypatch.setattr(dataset, "load_uma_calculator", lambda *a, **k: StubCalculator())
    return dataset
