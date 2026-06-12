import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from asmcmc.initialize import generate_random_config, SIGMA0


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
