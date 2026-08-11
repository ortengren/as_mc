"""Motif-seeded pair proposals and their densities.

The load-bearing tests here are the **normalisation** ones. A proposal that
draws correctly but reports a density with a missing Jacobian factor is the
worst failure mode available: every frame looks fine, nothing raises, and every
downstream importance weight is silently wrong. Integrating the declared density
is what catches it -- drop the ``1/(2 pi s)`` from ``StackedProposal`` and
``test_positional_factor_integrates_to_one`` fails while every other test still
passes.

No MLIP anywhere.
"""

import numpy as np
import pytest

from asmcmc.data_preparation.proposals import (
    LOG_4PI,
    AxialConcentration,
    MixtureProposal,
    RadialProposal,
    StackedProposal,
    TShapedProposal,
    UniformProposal,
    default_mixture,
    motif_reference,
    orthonormal_basis,
    rotation_to_normal,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260811)


# --- the reference geometry --------------------------------------------------


def test_motif_reference_matches_the_cacelli_minima():
    """Derived from the ab initio set, not hardcoded -- so it cannot drift away
    from what dimer_benchmark scores against."""
    ref = motif_reference()

    r, b, a_i, a_j = ref["cofacial"]
    assert r == pytest.approx(3.90, abs=0.01)
    assert abs(b) == pytest.approx(1.0, abs=1e-6)
    assert abs(a_i) == pytest.approx(1.0, abs=1e-6)

    r, b, a_i, a_j = ref["parallel_displaced"]
    assert r == pytest.approx(3.85, abs=0.01)
    assert abs(b) == pytest.approx(1.0, abs=1e-6)
    # The number the shipped motif_masks got wrong: PD is a_hi ~ 0.91, not < 0.6.
    assert abs(a_i) == pytest.approx(0.909, abs=0.005)
    assert r * np.sqrt(1 - a_i**2) == pytest.approx(1.6, abs=0.02)  # slip

    r, b, a_i, a_j = ref["t_shaped"]
    assert r == pytest.approx(5.0, abs=0.01)
    assert abs(b) == pytest.approx(0.0, abs=1e-6)
    assert abs(a_j) == pytest.approx(1.0, abs=1e-6)


# --- building blocks ---------------------------------------------------------


@pytest.mark.parametrize("kappa", [-12.0, -3.0, 0.0, 3.0, 12.0])
def test_axial_concentration_normalises_on_the_sphere(kappa):
    """2 pi int_-1^1 f(t) dt = 1, the sphere measure for a density in u.v alone."""
    axial = AxialConcentration(kappa)
    t = np.linspace(-1.0, 1.0, 20001)
    assert 2 * np.pi * np.trapezoid(np.exp(axial.log_pdf(t)), t) == pytest.approx(1.0, rel=1e-6)


def test_axial_concentration_is_even_in_t():
    """A disc normal is defined up to sign, so u and -u must score identically."""
    axial = AxialConcentration(9.0)
    t = np.linspace(-1, 1, 51)
    assert np.allclose(axial.log_pdf(t), axial.log_pdf(-t))


def _bin_probabilities(pdf, edges, n_sub=400):
    """Integrate ``pdf`` across each bin.

    Comparing a histogram against the density at the bin *centre* is wrong
    wherever the density is steep or discontinuous -- a bin holds the density's
    average, not its midpoint value. Both round-trip tests below would otherwise
    fail on curvature alone and say nothing about correctness.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        grid = np.linspace(lo, hi, n_sub)
        out.append(np.trapezoid(pdf(grid), grid))
    return np.asarray(out)


def test_axial_sampling_matches_its_own_density(rng):
    """Round-trip: the drawn cosines follow the density that is reported."""
    axial = AxialConcentration(6.0)
    drawn = axial.sample_cosine(rng, 200_000)

    edges = np.linspace(-1, 1, 21)
    observed, _ = np.histogram(drawn, bins=edges)
    observed = observed / observed.sum()
    # Marginal of t under the sphere measure is 2 pi f(t).
    expected = _bin_probabilities(lambda t: 2 * np.pi * np.exp(axial.log_pdf(t)), edges)

    assert expected.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.allclose(observed, expected, rtol=0.03, atol=1e-4)


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


def test_orthonormal_basis_is_orthonormal(rng):
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    e1, e2 = orthonormal_basis(u)

    for a, b in [(e1, e1), (e2, e2)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 1.0)
    for a, b in [(e1, e2), (e1, u), (e2, u)]:
        assert np.allclose(np.einsum("ni,ni->n", a, b), 0.0, atol=1e-12)


# --- rotations ---------------------------------------------------------------


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
    """Spin is a real atomistic degree of freedom that the CG coordinates -- and
    so the recorded density -- cannot see."""
    from asmcmc.data_preparation.cluster_dataset import build_reference_benzene

    target = np.array([[0.0, 0.0, 1.0]])
    a = rotation_to_normal(target, np.array([0.0]))[0]
    b = rotation_to_normal(target, np.array([0.7]))[0]

    positions = build_reference_benzene().get_positions()
    assert not np.allclose(positions @ a.T, positions @ b.T)
    assert np.allclose(a @ np.array([0, 0, 1.0]), b @ np.array([0, 0, 1.0]), atol=1e-12)


# --- densities: the Jacobian checks ------------------------------------------


def _positional_integral(component, u0, u1, extent, n=260):
    """Integrate exp(log_density) over r_vec on a Cartesian grid, at fixed normals.

    Cartesian on purpose: integrating in the same cylindrical coordinates the
    stacked proposal is *written* in would divide out the very Jacobian this is
    meant to check. Evaluated a z-slice at a time so the grid can be fine enough
    to resolve the support's curved boundary without a huge allocation.
    """
    axis = np.linspace(-extent, extent, n)
    step = axis[1] - axis[0]
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    flat_x, flat_y = gx.ravel(), gy.ravel()

    total = 0.0
    for z in axis:
        points = np.stack([flat_x, flat_y, np.full_like(flat_x, z)], axis=-1)
        u0_rep = np.broadcast_to(u0, points.shape)
        u1_rep = np.broadcast_to(u1, points.shape)
        total += float(np.sum(np.exp(component.log_density(u0_rep, u1_rep, points))))
    return total * step**3


@pytest.mark.parametrize(
    "component, extent",
    [
        (StackedProposal(), 6.0),
        (StackedProposal(slip_max=5.0, height_max=6.0), 6.4),
        (TShapedProposal(), 7.0),
    ],
)
def test_positional_factor_integrates_to_one(component, extent):
    """**The Jacobian test.** The positional part of the density must integrate
    to 1 over R^3, leaving only the (1/4pi) for u0 and the orientational factor.

    Removing the ``1/(2 pi s)`` from StackedProposal -- the easiest thing to get
    wrong, since the draw still looks perfectly sensible -- fails here.
    """
    u0 = np.array([0.0, 0.0, 1.0])
    u1 = np.array([0.0, 1.0, 0.0]) if component.name == "t_shaped" else u0

    total = _positional_integral(component, u0, u1, extent)

    orientational = np.exp(-LOG_4PI + _orientation_log_pdf(component, u0, u1))
    assert total == pytest.approx(orientational, rel=0.02)


def _orientation_log_pdf(component, u0, u1):
    t = float(np.dot(u0, u1))
    if component.name == "t_shaped":
        return component._perpendicular.log_pdf(t)
    return component._alignment.log_pdf(t)


def test_uniform_component_reproduces_the_radial_density():
    """Its positional part is p_r(r)/(4 pi r^2); check against p_r directly."""
    component = UniformProposal()
    u0 = np.array([[0.0, 0.0, 1.0]])
    u1 = np.array([[1.0, 0.0, 0.0]])

    for r in (4.0, 6.0, 11.0):
        r_vec = np.array([[0.0, 0.0, r]])
        expected = (
            -2 * LOG_4PI
            + component.radial.log_pdf(np.array([r]))[0]
            - LOG_4PI
            - 2 * np.log(r)
        )
        assert component.log_density(u0, u1, r_vec)[0] == pytest.approx(expected)


def test_stacked_draws_reproduce_the_declared_height_and_slip_laws(rng):
    """Round-trip between draw() and the analytic marginals log_density assumes:
    height uniform over +/-[h_min, h_max], slip uniform *in area*."""
    component = StackedProposal(height_min=3.2, height_max=5.5, slip_max=2.0)
    u0, _, r_vec = component.draw(rng, 200_000)

    height = np.einsum("ni,ni->n", r_vec, u0)
    slip = np.linalg.norm(r_vec - height[:, None] * u0, axis=1)

    assert np.all(np.abs(height) >= 3.2 - 1e-9)
    assert np.all(np.abs(height) <= 5.5 + 1e-9)
    assert np.mean(height > 0) == pytest.approx(0.5, abs=0.01)

    observed, edges = np.histogram(np.abs(height), bins=12, range=(3.2, 5.5), density=True)
    assert np.allclose(observed, 1.0 / (5.5 - 3.2), rtol=0.05)

    # Uniform in area means P(s < x) = (x/s_max)^2.
    for x in (0.5, 1.0, 1.5):
        assert np.mean(slip < x) == pytest.approx((x / 2.0) ** 2, abs=0.01)


def test_every_component_draws_inside_its_own_support(rng):
    """A draw the density scores as impossible would make log_q infinite."""
    for component in (
        StackedProposal(),
        StackedProposal(slip_max=5.0, height_max=6.0),
        TShapedProposal(),
        UniformProposal(),
    ):
        u0, u1, r_vec = component.draw(rng, 4000)
        assert np.all(np.isfinite(component.log_density(u0, u1, r_vec))), component.name


# --- motif recovery ----------------------------------------------------------


def test_stacked_draws_are_near_parallel(rng):
    u0, u1, _ = StackedProposal().draw(rng, 20_000)
    assert np.mean(np.abs(np.einsum("ni,ni->n", u0, u1)) > 0.9) > 0.75


def test_stacked_tight_brackets_the_cofacial_and_pd_minima(rng):
    """The family has to actually reach the geometries it exists to sample."""
    ref = motif_reference()
    component = StackedProposal(slip_max=2.0)
    u0, _, r_vec = component.draw(rng, 50_000)

    height = np.einsum("ni,ni->n", r_vec, u0)
    slip = np.linalg.norm(r_vec - height[:, None] * u0, axis=1)

    cofacial_r = ref["cofacial"][0]
    assert np.any((np.abs(np.abs(height) - cofacial_r) < 0.2) & (slip < 0.4))

    pd_r, _, pd_a, _ = ref["parallel_displaced"]
    pd_slip = pd_r * np.sqrt(1 - pd_a**2)
    assert np.any((np.abs(np.abs(height) - pd_r * pd_a) < 0.2) & (np.abs(slip - pd_slip) < 0.3))


def test_t_shaped_draws_are_perpendicular_with_axial_separation(rng):
    u0, u1, r_vec = TShapedProposal().draw(rng, 20_000)
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)

    assert np.mean(np.abs(np.einsum("ni,ni->n", u0, u1)) < 0.3) > 0.75
    assert np.mean(np.abs(np.einsum("ni,ni->n", r_hat, u1)) > 0.9) > 0.75


# --- the mixture -------------------------------------------------------------


def test_mixture_density_is_the_weighted_logsumexp(rng):
    mixture = default_mixture()
    u0, u1, r_vec, _ = mixture.draw(rng, 500)

    terms = np.stack([c.log_density(u0, u1, r_vec) for c in mixture.components])
    expected = np.log(
        np.sum(np.asarray(mixture.weights)[:, None] * np.exp(terms), axis=0)
    )
    assert np.allclose(mixture.log_density(u0, u1, r_vec), expected)


def test_mixture_density_is_finite_everywhere_it_draws(rng):
    """What the defensive uniform component buys: no drawn configuration can
    fall outside the mixture's support."""
    mixture = default_mixture()
    u0, u1, r_vec, _ = mixture.draw(rng, 20_000)
    assert np.all(np.isfinite(mixture.log_density(u0, u1, r_vec)))


def test_mixture_component_shares_match_the_weights(rng):
    mixture = default_mixture(uniform_weight=0.4)
    _, _, _, which = mixture.draw(rng, 40_000)
    for k, weight in enumerate(mixture.weights):
        assert np.mean(which == k) == pytest.approx(weight, abs=0.01)


def test_dropping_the_uniform_component_loses_support(rng):
    """Why default_mixture always keeps it: without a defensive component the
    mixture scores legitimate far-field geometries as impossible."""
    sharp = MixtureProposal(components=(StackedProposal(),), weights=(1.0,))
    u0 = np.array([[0.0, 0.0, 1.0]])
    far = np.array([[0.0, 0.0, 12.0]])
    assert not np.isfinite(sharp.log_density(u0, u0, far)[0])
    assert np.isfinite(default_mixture().log_density(u0, u0, far)[0])


def test_mixture_rejects_malformed_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        MixtureProposal(components=(StackedProposal(), UniformProposal()), weights=(0.3, 0.3))
    with pytest.raises(ValueError, match="equal length"):
        MixtureProposal(components=(StackedProposal(),), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="non-negative"):
        MixtureProposal(
            components=(StackedProposal(), UniformProposal()), weights=(-0.5, 1.5)
        )


# --- the point of the whole exercise -----------------------------------------


def test_motif_seeding_clears_the_hard_core_far_more_often(rng):
    """Motif-seeded proposals convert MLIP budget into short-range samples where
    uniform ones convert it into rejections."""
    from asmcmc.data_preparation.cluster_dataset import (
        build_reference_benzene,
        minimum_inter_molecular_distance,
    )

    positions = build_reference_benzene().get_positions()

    def yield_rate(component, n=600):
        u0, u1, r_vec = component.draw(rng, n)
        spin = rng.uniform(0, 2 * np.pi, size=(2, n))
        rot_a = rotation_to_normal(u0, spin[0])
        rot_b = rotation_to_normal(u1, spin[1])

        cleared = 0
        for k in range(n):
            if np.linalg.norm(r_vec[k]) > 5.0:
                continue  # only the close geometries are informative here
            a = positions @ rot_a[k].T
            b = positions @ rot_b[k].T + r_vec[k]
            cleared += minimum_inter_molecular_distance(a, b) >= 2.4
        close = np.sum(np.linalg.norm(r_vec, axis=1) <= 5.0)
        return cleared / max(close, 1)

    stacked = yield_rate(StackedProposal(height_min=3.4, height_max=4.2, slip_max=1.5))
    uniform = yield_rate(UniformProposal())
    assert stacked > 0.6
    assert stacked > 3 * uniform
