import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from asmcmc.potentials import gb, quadrupole, calc_total_energy, GB_PARAMS, QQ


# Helpers: unit-vector pairs and a displacement array shaped (1, 3)

def make_pair(u1, u2, r_vec):
    """Return (uhat1, uhat2, r) with batch dimension of 1."""
    u1 = np.array(u1, dtype=float)[np.newaxis]
    u2 = np.array(u2, dtype=float)[np.newaxis]
    r  = np.array(r_vec, dtype=float)[np.newaxis]
    return u1, u2, r


Z = [0., 0., 1.]
X = [1., 0., 0.]


# --- Gay-Berne ---

def test_gb_large_separation_near_zero():
    """GB energy vanishes at large separations."""
    u1, u2, r = make_pair(Z, Z, [1000., 0., 0.])
    energy = gb(u1, u2, r, **GB_PARAMS)
    assert abs(energy.item()) < 1e-10


def test_gb_repulsive_at_short_range():
    """Very close particles should have large positive energy."""
    u1, u2, r = make_pair(Z, Z, [0.01, 0., 0.])
    energy = gb(u1, u2, r, **GB_PARAMS).item()
    assert energy > 1e3


def test_gb_symmetry():
    """Swapping particles (u1↔u2, r→-r) leaves energy unchanged."""
    u1, u2, r = make_pair(Z, X, [8., 3., 0.])
    e_fwd = gb(u1, u2,  r, **GB_PARAMS).item()
    e_bwd = gb(u2, u1, -r, **GB_PARAMS).item()
    np.testing.assert_allclose(e_fwd, e_bwd, rtol=1e-10)


def test_gb_side_vs_end():
    """Side-by-side and end-to-end configurations at the same distance differ."""
    # side-by-side: both oriented along z, separated along x
    u1, u2, r_side = make_pair(Z, Z, [10., 0., 0.])
    # end-to-end: both oriented along x, separated along x
    u1e, u2e, r_end = make_pair(X, X, [10., 0., 0.])
    e_side = gb(u1,  u2,  r_side, **GB_PARAMS).item()
    e_end  = gb(u1e, u2e, r_end,  **GB_PARAMS).item()
    assert not np.isclose(e_side, e_end)


# --- Quadrupole ---

def test_quadrupole_large_separation_near_zero():
    """Quadrupole energy scales as r⁻⁵ and should be negligible at 1000 Å."""
    u1, u2, r = make_pair(Z, Z, [1000., 0., 0.])
    energy = np.squeeze(quadrupole(u1, u2, r, QQ)).item()
    assert abs(energy) < 1e-10


def test_quadrupole_symmetry():
    """Swapping particles (u1↔u2, r→-r) leaves quadrupole energy unchanged."""
    u1, u2, r = make_pair(Z, X, [8., 3., 0.])
    e_fwd = np.squeeze(quadrupole(u1, u2,  r, QQ)).item()
    e_bwd = np.squeeze(quadrupole(u2, u1, -r, QQ)).item()
    np.testing.assert_allclose(e_fwd, e_bwd, rtol=1e-10)


# --- calc_total_energy (integration) ---

def test_calc_total_energy_two_particles(two_particle_frame):
    """Total energy for 2-particle frame matches manual pairwise calculation."""
    frame = two_particle_frame
    nl_cutoff = [15.] * len(frame)

    total = calc_total_energy(frame, nl_cutoff, method="GB")

    # Manual: single pair, r = 10 Å along x, both oriented along z
    u1 = np.array([[0., 0., 1.]])
    u2 = np.array([[0., 0., 1.]])
    r  = np.array([[10., 0., 0.]])
    expected = gb(u1, u2, r, **GB_PARAMS).item() + np.squeeze(quadrupole(u1, u2, r, QQ)).item()

    np.testing.assert_allclose(total, expected, rtol=1e-8)
