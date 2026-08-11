import itertools
import json

import ase
import numpy as np
import pytest
from asmcmc.potentials import (
    gb,
    quadrupole,
    calc_total_energy,
    GBQPotential,
    GB_PARAMS,
    QQ,
)


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

    total = calc_total_energy(frame, nl_cutoff)

    # Manual: single pair, r = 10 Å along x, both oriented along z
    u1 = np.array([[0., 0., 1.]])
    u2 = np.array([[0., 0., 1.]])
    r  = np.array([[10., 0., 0.]])
    expected = gb(u1, u2, r, **GB_PARAMS).item() + np.squeeze(quadrupole(u1, u2, r, QQ)).item()

    np.testing.assert_allclose(total, expected, rtol=1e-8)


def test_calc_total_energy_counts_periodic_self_images():
    """A lone molecule in a small cell still interacts with its own images.

    Regression: deduplicating pairs with ``i < j`` also discarded every
    ``i == j`` self-image pair, so any cell with a lattice vector shorter than
    the cutoff was under-counted -- and a one-molecule cell came back as
    exactly 0.0. Only cells smaller than the cutoff are affected; MC boxes are
    much larger, so sampler results are unchanged.
    """
    frame = ase.Atoms("X", positions=[[0., 0., 0.]], cell=[6., 6., 6.], pbc=True)
    frame.arrays["or_vec"] = np.array([[0., 0., 1.]])

    cutoff = 10.0
    energy = calc_total_energy(frame, cutoff)
    assert energy != 0.0

    # Independent check: enumerate every image displacement inside the cutoff
    # and halve, since each pair is shared between the two molecules it joins.
    u = np.array([[0., 0., 1.]])
    expected = 0.0
    reach = int(np.ceil(cutoff / 6.0))
    for n in itertools.product(range(-reach, reach + 1), repeat=3):
        r = np.array([n], dtype=float) * 6.0
        d = np.linalg.norm(r)
        if 0.0 < d <= cutoff:
            expected += gb(u, u, r, **GB_PARAMS).item()
            expected += np.squeeze(quadrupole(u, u, r, QQ)).item()

    np.testing.assert_allclose(energy, 0.5 * expected, rtol=1e-9)


# --- GBQPotential (loading + provenance) ---

def _write_params_json(path, values):
    """Write a fit params.json in the {value, unit} schema asmcmc.fitting emits."""
    payload = {k: {"value": v, "unit": "x"} for k, v in values.items()}
    payload["E_intra"] = {"value": -1601.0, "unit": "eV/molecule"}  # ignored on load
    path.write_text(json.dumps(payload))


def test_gbqpotential_from_json_roundtrip(tmp_path):
    """from_json reads the fit schema and derives a name from the path tail
    below a `fitting/` directory; E_intra is not a potential parameter."""
    values = {**GB_PARAMS, "Q": QQ}
    fit_dir = tmp_path / "fitting" / "campaign" / "seed_0"
    fit_dir.mkdir(parents=True)
    _write_params_json(fit_dir / "params.json", values)

    pot = GBQPotential.from_json(fit_dir / "params.json")

    assert pot.name == "campaign/seed_0"
    assert pot.gb_params_dict() == GB_PARAMS
    assert pot.Q == QQ
    assert pot.gb_args == tuple(GB_PARAMS.values())


def test_gbqpotential_pair_energy_matches_gb_plus_quadrupole():
    """pair_energy is exactly gb(...) + quadrupole(...) for the same params."""
    pot = GBQPotential(name="test", **GB_PARAMS, Q=QQ)
    u1, u2, r = make_pair(Z, X, [9., 0., 0.])
    expected = gb(u1, u2, r, **GB_PARAMS) + np.squeeze(quadrupole(u1, u2, r, QQ))
    np.testing.assert_allclose(pot.pair_energy(u1, u2, r), expected, rtol=1e-12)


def test_calc_total_energy_uses_given_potential(two_particle_frame):
    """An explicitly passed potential overrides the package default."""
    other = GBQPotential(name="other", **{**GB_PARAMS, "eps0": GB_PARAMS["eps0"] * 2}, Q=QQ)
    nl_cutoff = [15.] * len(two_particle_frame)
    default = calc_total_energy(two_particle_frame, nl_cutoff)
    with_other = calc_total_energy(two_particle_frame, nl_cutoff, potential=other)
    assert not np.isclose(default, with_other)
