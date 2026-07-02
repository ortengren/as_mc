import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dataclasses

import numpy as np
import pytest
from asmcmc.potentials import DEFAULT_POTENTIAL
from asmcmc.initialize import (
    generate_random_config,
    generate_columnar_config,
    SIGMA0,
    KAPPA,
    DEFAULT_N_PARTICLES,
    DEFAULT_DENSITY,
    DEFAULT_COLUMNAR_DENSITY,
    Initializer,
    RandomLatticeInitializer,
    ColumnarLatticeInitializer,
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


def _min_center_dist(frame):
    """Minimum center-center distance for a (possibly orthorhombic) box, MIC."""
    pos = frame.positions
    L = np.diag(frame.cell)  # per-axis lengths
    best = np.inf
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[i] - pos[j]
            delta -= np.round(delta / L) * L
            best = min(best, np.linalg.norm(delta))
    return best


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
    assert prov == {
        "init_n_particles": 27,
        "init_density": 0.3,
        "init_seed": 5,
        "init_packing": "random",
        "init_sigma0": SIGMA0,
    }


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


# --- columnar config ---

def test_columnar_shapes():
    f = generate_columnar_config(50, density=1.0, seed=0)
    assert f.positions.shape == (50, 3)
    assert f.arrays["c_q"].shape == (50, 4)
    assert f.arrays["or_vec"].shape == (50, 3)


def test_columnar_quaternion_and_or_vec_norms_unity():
    f = generate_columnar_config(50, density=1.0, seed=0)
    np.testing.assert_allclose(np.linalg.norm(f.arrays["c_q"], axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(f.arrays["or_vec"], axis=1), 1.0, atol=1e-10
    )


def test_columnar_reaches_high_density():
    """The point of the columnar start: rho* > 1, which the SC start cannot do."""
    f = generate_columnar_config(50, density=DEFAULT_COLUMNAR_DENSITY, seed=0)
    realized = 50 * SIGMA0**3 / f.get_volume()
    assert realized == pytest.approx(DEFAULT_COLUMNAR_DENSITY, rel=1e-10)
    assert realized > 1.0


def test_columnar_no_overlaps():
    """No two centers closer than the tightest (face-to-face) contact kappa*sigma0."""
    f = generate_columnar_config(50, density=1.0, seed=0)
    assert _min_center_dist(f) >= KAPPA * SIGMA0


def test_columnar_default_density_is_usable():
    """At the (tight) default density the config places without overlaps and has
    finite energy — i.e. it is a usable starting frame, not a blown-up core."""
    from asmcmc.potentials import calc_total_energy

    f = generate_columnar_config(50, density=DEFAULT_COLUMNAR_DENSITY, seed=0)
    assert _min_center_dist(f) >= KAPPA * SIGMA0
    energy = calc_total_energy(f, nl_cutoff=14.0)
    assert np.isfinite(energy)


def test_columnar_is_orientationally_ordered():
    """Discs stack face-to-face: symmetry axes align with the column axis (z)."""
    f = generate_columnar_config(50, density=1.0, seed=0)
    z_alignment = np.mean(np.abs(f.arrays["or_vec"][:, 2]))
    assert z_alignment > 0.95  # near-perfect alignment, modulo the small tilt


def test_columnar_box_is_orthorhombic_and_near_cubic():
    """Cell is diagonal and roughly cubic on every axis (the grid is chosen so
    the column height and near-square in-plane tiling keep L_x ~ L_y ~ L_z)."""
    f = generate_columnar_config(50, density=1.0, seed=0)
    cell = np.array(f.cell)
    assert np.allclose(cell, np.diag(np.diag(cell)))
    edges = np.diag(cell)
    assert edges.max() / edges.min() < 2.0


def test_columnar_seed_reproducibility():
    f1 = generate_columnar_config(50, density=1.0, seed=42)
    f2 = generate_columnar_config(50, density=1.0, seed=42)
    np.testing.assert_array_equal(f1.positions, f2.positions)
    np.testing.assert_array_equal(f1.arrays["c_q"], f2.arrays["c_q"])


def test_columnar_different_seeds_give_independent_starts():
    f1 = generate_columnar_config(50, density=1.0, seed=1)
    f2 = generate_columnar_config(50, density=1.0, seed=2)
    assert not np.allclose(f1.positions, f2.positions)
    assert not np.allclose(f1.arrays["c_q"], f2.arrays["c_q"])


def test_columnar_raises_on_density_too_high():
    with pytest.raises(ValueError, match="too high"):
        generate_columnar_config(50, density=5.0)


def test_columnar_pbc_and_positions_within_box():
    f = generate_columnar_config(50, density=1.0, seed=0)
    assert all(f.pbc)
    L = np.diag(f.cell)
    assert np.all(f.positions >= 0)
    assert np.all(f.positions < L)


# --- ColumnarLatticeInitializer ---

def test_columnar_initializer_generates_valid_frame():
    init = ColumnarLatticeInitializer(n_particles=50, density=1.0, seed=0)
    frame = init.generate()
    assert frame.positions.shape == (50, 3)
    assert init.volume == pytest.approx(frame.get_volume())


def test_columnar_initializer_defaults():
    init = ColumnarLatticeInitializer()
    assert init.n_particles == DEFAULT_N_PARTICLES
    assert init.density == DEFAULT_COLUMNAR_DENSITY
    assert init.seed is None


def test_columnar_initializer_matches_generate_columnar_config():
    init = ColumnarLatticeInitializer(n_particles=50, density=1.0, seed=7)
    direct = generate_columnar_config(n_particles=50, density=1.0, seed=7)
    np.testing.assert_array_equal(init.generate().positions, direct.positions)


def test_columnar_initializer_provenance_records_packing():
    prov = ColumnarLatticeInitializer(n_particles=50, density=1.2, seed=5).provenance()
    assert prov["init_n_particles"] == 50
    assert prov["init_density"] == 1.2
    assert prov["init_seed"] == 5
    assert prov["init_packing"] == "columnar"
    assert prov["init_tilt"] == pytest.approx(0.15)


def test_calculator_accepts_columnar_initializer():
    init = ColumnarLatticeInitializer(n_particles=50, density=1.0, seed=0)
    mc = MetropolisCalculator(temp=300, pressure=0.0, initializer=init)
    assert mc.initializer is init


# --- geometry is built for the simulated potential's shape, not the default ---


def test_random_config_honors_custom_sigma0():
    """rho* and hard-core spacing scale with the passed sigma0, not the global."""
    sig = 4.0
    f = generate_random_config(64, density=0.5, seed=0, sigma0=sig)
    assert 64 * sig**3 / f.get_volume() == pytest.approx(0.5, rel=1e-10)
    assert _min_pairwise_dist(f) >= sig


def test_columnar_config_honors_custom_shape():
    """Box (via sigma0) and contact distances (via kappa) follow the passed shape."""
    sig, kap = 4.0, 0.5
    f = generate_columnar_config(64, density=1.2, seed=0, sigma0=sig, kappa=kap)
    assert 64 * sig**3 / f.get_volume() == pytest.approx(1.2, rel=1e-10)
    # tightest packing is the axial (face-to-face) contact kappa*sigma0
    assert _min_center_dist(f) >= kap * sig - 1e-9


def test_random_initializer_reads_shape_from_potential():
    pot = dataclasses.replace(DEFAULT_POTENTIAL, sigma0=4.0, kappa=0.5)
    init = RandomLatticeInitializer(n_particles=64, density=0.5, seed=0, potential=pot)
    assert init.sigma0 == 4.0
    f = init.generate()
    assert 64 * 4.0**3 / f.get_volume() == pytest.approx(0.5, rel=1e-10)
    assert init.provenance()["init_sigma0"] == 4.0


def test_columnar_initializer_reads_shape_from_potential():
    pot = dataclasses.replace(DEFAULT_POTENTIAL, sigma0=4.0, kappa=0.5)
    init = ColumnarLatticeInitializer(
        n_particles=64, density=1.2, seed=0, potential=pot
    )
    assert (init.sigma0, init.kappa) == (4.0, 0.5)
    f = init.generate()
    assert 64 * 4.0**3 / f.get_volume() == pytest.approx(1.2, rel=1e-10)
    prov = init.provenance()
    assert prov["init_sigma0"] == 4.0
    assert prov["init_kappa"] == 0.5


def test_calculator_propagates_potential_shape_to_initializer():
    """The footgun fix: an initializer built without a potential adopts the one
    passed to the calculator, so geometry can't silently use the default shape."""
    pot = dataclasses.replace(DEFAULT_POTENTIAL, sigma0=6.0, kappa=0.5)
    init = ColumnarLatticeInitializer(n_particles=64, density=1.0, seed=0)
    mc = MetropolisCalculator(
        temp=300, pressure=0.0, initializer=init, potential=pot, nl_radius=10.0
    )
    assert (init.sigma0, init.kappa) == (6.0, 0.5)
    assert 64 * 6.0**3 / mc.init_frame.get_volume() == pytest.approx(1.0, rel=1e-10)


def test_explicit_initializer_potential_wins_over_calculator():
    """An initializer given its own potential keeps that shape; the calculator's
    set_potential does not override an explicit choice."""
    pot_init = dataclasses.replace(DEFAULT_POTENTIAL, sigma0=6.0, kappa=0.5)
    pot_calc = dataclasses.replace(DEFAULT_POTENTIAL, sigma0=9.0, kappa=0.7)
    init = ColumnarLatticeInitializer(
        n_particles=64, density=1.0, seed=0, potential=pot_init
    )
    init.set_potential(pot_calc)
    assert (init.sigma0, init.kappa) == (6.0, 0.5)
