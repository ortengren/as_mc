import os

import ase
import ase.io
import numpy as np
import pytest

from asmcmc.utils.polymorph import (
    contact_invariants,
    disorder_amplitude,
    neighbour_shell,
    pair_geometry,
    rescale,
    thermal_jitter,
)
from asmcmc.utils.geometry import coarse_grain_frame

CIF = os.path.join(os.path.dirname(__file__), "..", "data", "benzene_pbca_cod_7238223.cif")


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def herringbone():
    """The experimental Pbca crystal, coarse-grained and tiled.

    The tracked cif is a single 4-molecule cell; repeating it gives enough
    environments to be a meaningful reference while staying the same packing.
    """
    return coarse_grain_frame(ase.io.read(CIF)).repeat((3, 3, 3))


def dimer(u1, u2, separation, box=40.0, pbc=True):
    """Two discs at a chosen relative geometry, isolated in a large box.

    The box is far wider than any cutoff used below, so the neighbour list sees
    only the intended pair and not its periodic images.
    """
    frame = ase.Atoms(
        "XX",
        positions=[[0.0, 0.0, 0.0], separation],
        cell=np.diag([box] * 3),
        pbc=pbc,
    )
    frame.arrays["or_vec"] = np.array([u1, u2], dtype=float)
    return frame


Z = [0.0, 0.0, 1.0]
X = [1.0, 0.0, 0.0]

# --- pair geometry and shell selection --------------------------------------


def test_pair_geometry_matches_extract_periodic_pairs(herringbone):
    """Same invariants as the fitting code, so statistics stay comparable."""
    from asmcmc.fitting_gbq.data import extract_periodic_pairs

    reference = extract_periodic_pairs(herringbone, "or_vec", 7.0)
    pairs = pair_geometry(herringbone, 7.0)
    mine = np.stack([pairs["r"], pairs["a_i"], pairs["a_j"], pairs["b"]], axis=1)
    # Neighbour-list ordering is an implementation detail; compare as sets.
    order_ref = np.lexsort(reference.T[::-1])
    order_mine = np.lexsort(mine.T[::-1])
    np.testing.assert_allclose(mine[order_mine], reference[order_ref], atol=1e-10)


def test_neighbour_shell_returns_exactly_k_per_molecule(herringbone):
    k = 12
    shell = neighbour_shell(herringbone, k=k)
    counts = np.bincount(shell["i"], minlength=len(herringbone))
    assert set(counts) == {k}


def test_neighbour_shell_takes_the_nearest(herringbone):
    """The kept pairs are each molecule's closest, not an arbitrary k."""
    k = 6
    shell = neighbour_shell(herringbone, k=k)
    everything = pair_geometry(herringbone, 12.0)
    for centre in range(0, len(herringbone), 7):
        kept = np.sort(shell["r"][shell["i"] == centre])
        allr = np.sort(everything["r"][everything["i"] == centre])[:k]
        np.testing.assert_allclose(kept, allr, atol=1e-10)


def test_neighbour_shell_counts_periodic_images():
    """Under PBC a molecule's neighbours legitimately include images of itself
    and its partner, which is how a small unit cell -- the 4-molecule
    coarse-grained cif -- still yields a full shell."""
    shell = neighbour_shell(dimer(Z, Z, [5.0, 0.0, 0.0], box=12.0), k=12)
    assert set(np.bincount(shell["i"], minlength=2)) == {12}


def test_neighbour_shell_refuses_impossible_k():
    """Without PBC there are only N-1 neighbours to be had, so the search
    terminates instead of growing without bound."""
    frame = dimer(Z, Z, [5.0, 0.0, 0.0], box=200.0, pbc=False)
    with pytest.raises(ValueError, match="cannot find 12 neighbours"):
        neighbour_shell(frame, k=12)


# --- pair invariants --------------------------------------------------------


@pytest.mark.parametrize(
    "u1,u2,separation,gamma,a_min,a_max",
    [
        # parallel normals, neighbour stacked over the face
        (Z, Z, [0.0, 0.0, 5.0], 0.0, 1.0, 1.0),
        # parallel normals, neighbour out in the shared plane
        (Z, Z, [5.0, 0.0, 0.0], 0.0, 0.0, 0.0),
        # orthogonal normals, separation along the first
        (Z, X, [0.0, 0.0, 5.0], 90.0, 0.0, 1.0),
    ],
)
def test_contact_invariants_of_known_dimers(u1, u2, separation, gamma, a_min, a_max):
    """The two numbers that fix a disc pair up to rotation, on geometries whose
    values can be read off by hand."""
    shell = neighbour_shell(dimer(u1, u2, separation, box=40.0, pbc=False), k=1)
    g, amin, amax = contact_invariants(shell)
    assert g[0] == pytest.approx(gamma, abs=1e-6)
    assert amin[0] == pytest.approx(a_min, abs=1e-6)
    assert amax[0] == pytest.approx(a_max, abs=1e-6)


def test_contact_invariants_are_scale_invariant(herringbone):
    """A k-nearest shell keeps the same pairs under rescaling, so the angles it
    reports do not carry density -- the reason the shell is sized by count and
    not by a fixed cutoff."""
    v = herringbone.get_volume() / len(herringbone)
    reference = contact_invariants(neighbour_shell(herringbone))
    for factor in (0.7, 1.6):
        scaled = contact_invariants(neighbour_shell(rescale(herringbone, factor * v)))
        for mine, theirs in zip(scaled, reference):
            np.testing.assert_allclose(np.sort(mine), np.sort(theirs), atol=1e-9)


def test_contact_invariants_are_inversion_blind(herringbone):
    """Taken on absolute values, so a pair and its inversion are one point --
    rotation-invariant, and by the same token unable to tell them apart."""
    shell = neighbour_shell(herringbone)
    flipped = dict(shell)
    flipped["a_i"] = -shell["a_i"]
    flipped["a_j"] = -shell["a_j"]
    flipped["b"] = -shell["b"]
    for mine, theirs in zip(contact_invariants(flipped), contact_invariants(shell)):
        np.testing.assert_allclose(mine, theirs, atol=1e-12)


# --- density matching -------------------------------------------------------


def test_rescale_hits_the_target_volume_and_keeps_orientations(herringbone):
    scaled = rescale(herringbone, 150.0)
    assert scaled.get_volume() / len(scaled) == pytest.approx(150.0)
    np.testing.assert_allclose(
        scaled.arrays["or_vec"], herringbone.arrays["or_vec"], atol=1e-12
    )
    # The original must be untouched.
    assert herringbone.get_volume() / len(herringbone) == pytest.approx(123.6, abs=0.5)


# --- thermal disorder: jitter and its calibration ---------------------------


def test_thermal_jitter_at_zero_amplitude_changes_nothing(herringbone):
    still = thermal_jitter(herringbone, 0.0, 0.0, seed=0)
    np.testing.assert_allclose(still.positions, herringbone.positions, atol=1e-12)
    np.testing.assert_allclose(
        still.arrays["or_vec"], herringbone.arrays["or_vec"], atol=1e-12
    )
    # And the original is left alone.
    assert still is not herringbone


@pytest.mark.parametrize("tilt_deg", [3.0, 9.0, 25.0])
def test_jitter_parameter_is_the_realised_tilt(herringbone, tilt_deg):
    """The whole point of tilting perpendicular to the normal.

    A rotation about a uniformly random 3D axis would put part of its amplitude
    into spin about the normal, which no descriptor sees, so the parameter would
    not equal the disorder it produces and could not be matched to a measured
    trajectory amplitude.
    """
    jittered = thermal_jitter(herringbone, 0.0, np.radians(tilt_deg), seed=3)
    u, v = herringbone.arrays["or_vec"], jittered.arrays["or_vec"]
    angles = np.arccos(np.clip(np.abs(np.einsum("nk,nk->n", u, v)), 0.0, 1.0))
    realised = np.degrees(np.sqrt((angles**2).mean()))
    assert realised == pytest.approx(tilt_deg, rel=0.12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_thermal_jitter_positions_match_the_requested_sigma(herringbone):
    jittered = thermal_jitter(herringbone, 0.2, 0.0, seed=5)
    delta = jittered.positions - herringbone.positions
    assert np.sqrt((delta**2).mean()) == pytest.approx(0.2, rel=0.15)


def test_thermal_jitter_is_seeded(herringbone):
    same = thermal_jitter(herringbone, 0.1, 0.1, seed=7)
    again = thermal_jitter(herringbone, 0.1, 0.1, seed=7)
    other = thermal_jitter(herringbone, 0.1, 0.1, seed=8)
    np.testing.assert_allclose(same.positions, again.positions, atol=1e-12)
    assert not np.allclose(same.positions, other.positions)


def test_disorder_amplitude_recovers_the_amplitude_it_was_given(herringbone):
    """The round trip the calibration depends on.

    A trajectory of independently jittered copies of one structure has, by
    construction, a known spread about its mean; `disorder_amplitude` has to
    report that same number back in the units `thermal_jitter` consumes, or the
    reference cannot be disordered to match a real run.
    """
    pos_sigma, tilt_sigma = 0.18, np.radians(11.0)
    frames = [
        thermal_jitter(herringbone, pos_sigma, tilt_sigma, seed=s) for s in range(40)
    ]
    measured = disorder_amplitude(frames)
    assert measured["pos_sigma"] == pytest.approx(pos_sigma, rel=0.1)
    assert measured["tilt_deg"] == pytest.approx(11.0, rel=0.1)


def test_disorder_amplitude_ignores_collective_drift(herringbone):
    """Translating the whole box is not disorder."""
    frames = []
    for step, s in enumerate(range(30)):
        frame = thermal_jitter(herringbone, 0.12, np.radians(6.0), seed=s)
        frame.positions += step * np.array([0.5, -0.4, 0.3])
        frames.append(frame)
    corrected = disorder_amplitude(frames, remove_drift=True)["pos_sigma"]
    assert corrected == pytest.approx(0.12, rel=0.15)
    # Left in, the ramp adds itself in quadrature and inflates the estimate.
    assert disorder_amplitude(frames, remove_drift=False)["pos_sigma"] > 2 * corrected


def test_disorder_amplitude_is_blind_to_which_packing(herringbone):
    """It measures spread about a site, not what the sites are -- which is what
    lets a slipped-parallel run calibrate a herringbone reference."""
    dense = rescale(herringbone, 96.5)
    common = dict(pos_sigma=0.15, tilt_sigma=np.radians(9.0))
    loose = [thermal_jitter(herringbone, seed=s, **common) for s in range(30)]
    tight = [thermal_jitter(dense, seed=s, **common) for s in range(30)]
    assert disorder_amplitude(loose)["tilt_deg"] == pytest.approx(
        disorder_amplitude(tight)["tilt_deg"], rel=0.05
    )


def test_disorder_amplitude_needs_more_frames_than_the_lag(herringbone):
    with pytest.raises(ValueError, match="lag"):
        disorder_amplitude([herringbone, herringbone], lag=5)
