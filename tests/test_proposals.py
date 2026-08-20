"""One generator per contact motif, and the contract that they classify back.

The load-bearing test here is :func:`test_each_generator_produces_its_own_motif`.
A generator that draws plausible-looking geometry but lands on the wrong side of
a classifier cut is the worst failure mode available: every frame looks fine,
nothing raises, and the campaign's stated composition is quietly false. That is
not hypothetical -- it is what the previous components did, measuring stacking
height and slip about molecule 0's normal while the classifier measures them
about the *more aligned* normal of the pair.

No MLIP anywhere.
"""

import numpy as np
import pytest

from asmcmc.data_preparation import cluster_analysis as ca
from asmcmc.data_preparation.proposals import (
    COFACIAL,
    FAR_SLIPPED,
    MOTIF_DRAWS,
    MOTIFS,
    PARALLEL_DISPLACED,
    RadialProposal,
    T_SHAPED,
    UNIFORM,
    classify_pair,
    draw_motif,
    motif_reference,
    orthonormal_basis,
    pair_invariants,
    rotation_to_normal,
)

NAMED_MOTIFS = [m for m in MOTIFS if m != UNIFORM]


@pytest.fixture
def rng():
    return np.random.default_rng(20260811)


# --- generator / classifier agreement ----------------------------------------


def test_motif_names_match_the_classifier():
    """The four strings are written down twice; keep them one set.

    ``proposals`` cannot import ``cluster_analysis`` at module scope without
    closing an import cycle, so the names are mirrored. This is the pin that
    makes mirroring safe.
    """
    assert {COFACIAL, PARALLEL_DISPLACED, T_SHAPED, FAR_SLIPPED} == {
        ca.COFACIAL,
        ca.PARALLEL_DISPLACED,
        ca.T_SHAPED,
        ca.FAR_SLIPPED,
    }


def test_classify_pair_agrees_with_motif_masks(rng):
    """``classify_pair`` must be ``motif_masks`` applied to one pair.

    Checked against the classifier itself rather than a restatement of its cuts,
    so the two cannot drift even if the thresholds move.
    """
    u0 = rng.normal(size=(300, 3))
    u1 = rng.normal(size=(300, 3))
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    r_hat = rng.normal(size=(300, 3))
    r_hat /= np.linalg.norm(r_hat, axis=1, keepdims=True)
    r_vec = rng.uniform(3.0, 7.0, size=300)[:, None] * r_hat

    records = {
        "r": np.linalg.norm(r_vec, axis=1),
        "a_i": np.einsum("ni,ni->n", r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True), u0),
        "a_j": np.einsum("ni,ni->n", r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True), u1),
        "b": np.einsum("ni,ni->n", u0, u1),
    }
    masks = ca.motif_masks(records)

    for k in range(300):
        mine = classify_pair(u0[k], u1[k], r_vec[k])
        theirs = [name for name, mask in masks.items() if mask[k]]
        assert theirs == ([mine] if mine is not None else [])


@pytest.mark.parametrize("motif", NAMED_MOTIFS)
def test_each_generator_produces_its_own_motif(motif, rng):
    """The contract the restructure exists to provide.

    Without it a campaign's requested composition and its measured composition
    are unrelated quantities.
    """
    for _ in range(200):
        assert classify_pair(*draw_motif(motif, rng)) == motif


@pytest.mark.parametrize("motif", NAMED_MOTIFS)
def test_raw_generators_are_already_close_before_verification(motif, rng):
    """The retry loop should be a guard, not the mechanism.

    If a generator's raw hit rate collapsed, ``draw_motif`` would still return
    correct geometry while silently costing many draws per config -- so pin the
    rate rather than only the outcome.
    """
    hits = sum(
        classify_pair(*MOTIF_DRAWS[motif](rng, None)) == motif for _ in range(200)
    )
    assert hits / 200 > 0.6


def test_uniform_is_exempt_from_classification(rng):
    """It is deliberately unconstrained: most draws are no motif at all."""
    labels = [classify_pair(*draw_motif(UNIFORM, rng)) for _ in range(200)]
    assert sum(label is None for label in labels) > 100


def test_draw_motif_rejects_an_unknown_motif(rng):
    with pytest.raises(ValueError, match="unknown motif"):
        draw_motif("herringbone", rng)


# --- the reference geometry --------------------------------------------------


def test_motif_reference_matches_the_cacelli_minima():
    """Derived from the ab initio file, not hardcoded -- so pin what it derives.

    The number an earlier ``motif_masks`` got wrong: parallel-displaced sits at
    a_hi ~ 0.91, not below 0.6, which is why slip and not a_hi separates it from
    cofacial.
    """
    ref = motif_reference()

    r, b, a_i, _ = ref["cofacial"]
    assert r == pytest.approx(3.90, abs=0.01)
    assert abs(b) == pytest.approx(1.0, abs=0.01)
    assert abs(a_i) == pytest.approx(1.0, abs=0.01)

    r, b, a_i, _ = ref["parallel_displaced"]
    assert r == pytest.approx(3.85, abs=0.01)
    assert abs(a_i) == pytest.approx(0.909, abs=0.01)
    assert r * np.sqrt(1 - a_i**2) == pytest.approx(1.6, abs=0.05)

    r, b, _, a_j = ref["t_shaped"]
    assert r == pytest.approx(5.0, abs=0.01)
    assert abs(b) == pytest.approx(0.0, abs=0.01)
    assert abs(a_j) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "motif,family",
    [
        (COFACIAL, "cofacial"),
        (PARALLEL_DISPLACED, "parallel_displaced"),
        (T_SHAPED, "t_shaped"),
    ],
)
def test_each_generator_reaches_its_own_cacelli_minimum(motif, family, rng):
    """A generator confined to the right label is not enough -- it also has to
    reach the geometry the ab initio minimum actually sits at."""
    r_ref, _, a_i, a_j = motif_reference()[family]
    # a_hi, not a_i: slip is resolved about the *more aligned* normal, and for
    # the T-shaped minimum the two differ completely (a_i = 0, a_j = 1).
    a_hi = max(abs(a_i), abs(a_j))
    slip_ref = r_ref * np.sqrt(max(1.0 - a_hi**2, 0.0))

    best = min(
        (
            (abs(inv[0] - r_ref) + abs(inv[5] - slip_ref))
            for inv in (pair_invariants(*draw_motif(motif, rng)) for _ in range(600))
        )
    )
    assert best < 0.35


def test_pair_invariants_are_sign_and_order_free(rng):
    """A disc normal is defined up to sign and the pair is unordered, so neither
    flipping a normal nor swapping the molecules may move the invariants."""
    u0, u1, r_vec = draw_motif(PARALLEL_DISPLACED, rng)
    base = pair_invariants(u0, u1, r_vec)

    assert pair_invariants(-u0, u1, r_vec) == pytest.approx(base)
    assert pair_invariants(u0, -u1, r_vec) == pytest.approx(base)
    # Swapping molecules negates the displacement as well as exchanging normals.
    assert pair_invariants(u1, u0, -r_vec) == pytest.approx(base)


# --- radial sampling ---------------------------------------------------------


def _bin_probabilities(pdf, edges, n=4001):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        grid = np.linspace(lo, hi, n)
        out.append(np.trapezoid(pdf(grid), grid))
    return np.asarray(out)


@pytest.mark.parametrize("mode", ["mixture", "volume-uniform"])
def test_radial_proposal_normalises(mode):
    radial = RadialProposal(mode=mode)
    r = np.linspace(radial.lo, radial.hi, 400_001)
    assert np.trapezoid(radial.pdf(r), r) == pytest.approx(1.0, rel=1e-4)


def test_radial_sampling_matches_its_density(rng):
    """Bin-integrated, not centre-sampled: the mixture density is discontinuous
    at its 6 A cap, so a bin straddling it has no meaningful midpoint value."""
    radial = RadialProposal(mode="mixture")
    drawn = radial.sample(rng, 200_000)

    edges = np.linspace(3.4, 15.0, 30)
    observed, _ = np.histogram(drawn, bins=edges)
    observed = observed / observed.sum()
    expected = _bin_probabilities(radial.pdf, edges)

    assert expected.sum() == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(observed, expected, rtol=0.05, atol=3e-4)


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


# --- the point of the whole exercise -----------------------------------------


def test_motif_seeding_clears_the_hard_core_far_more_often(rng):
    """Motif-seeded proposals convert MLIP budget into short-range samples where
    uniform ones convert it into rejections."""
    from asmcmc.data_preparation.cluster_dataset import (
        build_reference_benzene,
        minimum_inter_molecular_distance,
    )

    positions = build_reference_benzene().get_positions()

    def yield_rate(motif, n=400):
        cleared = close = 0
        for _ in range(n):
            u0, u1, r_vec = draw_motif(motif, rng)
            if np.linalg.norm(r_vec) > 5.0:
                continue  # only the close geometries are informative here
            close += 1
            rot = rotation_to_normal(
                np.stack([u0, u1]), rng.uniform(0, 2 * np.pi, size=2)
            )
            a = positions @ rot[0].T
            b = positions @ rot[1].T + r_vec
            cleared += minimum_inter_molecular_distance(a, b) >= 2.4
        return cleared / max(close, 1)

    assert yield_rate(COFACIAL) > 0.6
    assert yield_rate(COFACIAL) > 3 * yield_rate(UNIFORM)
