import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from asmcmc.initialize import (
    generate_random_config,
    SIGMA0,
    DEFAULT_N_PARTICLES,
    DEFAULT_DENSITY,
    Initializer,
    RandomLatticeInitializer,
    FrameInitializer,
)
from asmcmc.metropolis import MetropolisCalculator


# --- helpers ---

def _min_pairwise_dist(frame):
    """Minimum pairwise distance across all pairs, using minimum-image convention."""
    pos = frame.positions
    L = frame.cell[0, 0]  # cubic box
    dists = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[i] - pos[j]
            delta -= np.round(delta / L) * L  # minimum image
            dists.append(np.linalg.norm(delta))
    return min(dists)


# --- array shapes and types ---

def test_positions_shape():
    f = generate_random_config(27, seed=0)
    assert f.positions.shape == (27, 3)


def test_c_q_shape():
    f = generate_random_config(27, seed=0)
    assert f.arrays["c_q"].shape == (27, 4)


def test_or_vec_shape():
    f = generate_random_config(27, seed=0)
    assert f.arrays["or_vec"].shape == (27, 3)


# --- physical validity ---

def test_quaternion_norms_are_unity():
    f = generate_random_config(27, seed=0)
    norms = np.linalg.norm(f.arrays["c_q"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_or_vec_norms_are_unity():
    f = generate_random_config(27, seed=0)
    norms = np.linalg.norm(f.arrays["or_vec"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_no_hard_core_overlaps():
    """Minimum pairwise distance must be >= sigma0 for any density <= max safe."""
    f = generate_random_config(27, density=0.3, seed=0)
    assert _min_pairwise_dist(f) >= SIGMA0


def test_positions_within_box():
    f = generate_random_config(27, seed=0)
    L = f.cell[0, 0]
    assert np.all(f.positions >= 0)
    assert np.all(f.positions < L)


def test_pbc_enabled():
    f = generate_random_config(8, seed=0)
    assert all(f.pbc)


# --- reproducibility ---

def test_seed_reproducibility():
    f1 = generate_random_config(27, density=0.3, seed=99)
    f2 = generate_random_config(27, density=0.3, seed=99)
    np.testing.assert_array_equal(f1.positions, f2.positions)
    np.testing.assert_array_equal(f1.arrays["c_q"], f2.arrays["c_q"])


def test_different_seeds_give_different_configs():
    f1 = generate_random_config(27, density=0.3, seed=1)
    f2 = generate_random_config(27, density=0.3, seed=2)
    assert not np.allclose(f1.positions, f2.positions)


# --- density and box geometry ---

def test_box_volume_matches_density():
    n, density = 27, 0.3
    f = generate_random_config(n, density=density, seed=0)
    volume = np.linalg.det(f.cell)
    expected_density = n * SIGMA0**3 / volume
    assert abs(expected_density - density) < 1e-10


def test_cell_is_cubic():
    f = generate_random_config(8, seed=0)
    assert np.allclose(f.cell, np.diag(np.diag(f.cell)))
    assert np.allclose(np.diag(f.cell), f.cell[0, 0])


# --- error handling ---

def test_raises_on_density_too_high():
    with pytest.raises(ValueError, match="too high"):
        generate_random_config(8, density=10.0)


def test_non_cube_particle_count():
    """N not a perfect cube should still work (SC lattice fills n_side^3, takes first N)."""
    f = generate_random_config(10, density=0.2, seed=0)
    assert f.positions.shape == (10, 3)


# --- Initializer classes ---

def test_initializer_is_abstract():
    """The Initializer base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Initializer()


def test_random_lattice_initializer_generates_valid_frame():
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=0)
    frame = init.generate()
    assert frame.positions.shape == (27, 3)
    assert frame.arrays["c_q"].shape == (27, 4)
    # volume is recorded onto the initializer after generate()
    assert init.volume == pytest.approx(frame.get_volume())


def test_random_lattice_initializer_defaults():
    init = RandomLatticeInitializer()
    assert init.n_particles == DEFAULT_N_PARTICLES
    assert init.density == DEFAULT_DENSITY
    assert init.seed is None


def test_random_lattice_initializer_matches_generate_random_config():
    """The initializer is a thin wrapper around generate_random_config."""
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=7)
    direct = generate_random_config(n_particles=27, density=0.3, seed=7)
    np.testing.assert_array_equal(init.generate().positions, direct.positions)


def test_random_lattice_initializer_provenance():
    prov = RandomLatticeInitializer(n_particles=27, density=0.3, seed=5).provenance()
    assert prov == {"init_n_particles": 27, "init_density": 0.3, "init_seed": 5}


def test_frame_initializer_wraps_supplied_frame():
    frame = generate_random_config(27, density=0.3, seed=0)
    init = FrameInitializer(frame)
    assert init.generate() is frame
    assert init.n_particles == 27
    assert init.volume == pytest.approx(frame.get_volume())
    assert init.density == pytest.approx(27 / frame.get_volume())


# --- MetropolisCalculator frame-source resolution ---

def test_calculator_defaults_to_random_lattice_initializer():
    mc = MetropolisCalculator(temp=300, pressure=0.0)
    assert isinstance(mc.initializer, RandomLatticeInitializer)


def test_calculator_wraps_init_frame_in_frame_initializer():
    frame = generate_random_config(27, density=0.3, seed=0)
    mc = MetropolisCalculator(temp=300, pressure=0.0, init_frame=frame)
    assert isinstance(mc.initializer, FrameInitializer)


def test_calculator_accepts_explicit_initializer():
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=0)
    mc = MetropolisCalculator(temp=300, pressure=0.0, initializer=init)
    assert mc.initializer is init


def test_calculator_rejects_both_init_frame_and_initializer():
    frame = generate_random_config(27, density=0.3, seed=0)
    init = RandomLatticeInitializer(n_particles=27, density=0.3, seed=0)
    with pytest.raises(ValueError, match="at most one"):
        MetropolisCalculator(
            temp=300, pressure=0.0, init_frame=frame, initializer=init
        )
