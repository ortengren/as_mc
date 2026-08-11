"""Motif-seeded joint proposals for benzene pair geometry.

``cluster_dataset.make_cluster`` draws orientations uniformly and then rejects
clashes. At close approach that is almost all rejection: at 3.6 A only ~3% of
random-orientation placements clear a 2.4 A hard core, against ~86% for a
cofacial-seeded one (measured by ``cluster_analysis.hard_core_pass_rate``).
Reshaping the *radial* proposal cannot fix that -- the extra short-range draws
are simply rejected and redrawn -- because the hard core couples position to
orientation. So they have to be proposed together, which is what this module
does.

**Every component is a normalised density over ``(u0, u1, r_vec)`` that can be
evaluated at an arbitrary configuration**, not only at the one it drew. That is
what makes a mixture density well defined, and it is what lets a campaign record
its own ``log_q`` so a downstream fit can reweight to whatever target it wants
long after the MLIP budget is spent.

**No latent variables.** The obvious way to write a stacked proposal -- draw a
common axis ``n``, then put both normals near it -- makes ``n`` latent, so
``q(x) = int q(x|n) p(n) dn`` has no closed form and any recorded density would
be wrong. Everything here is instead conditioned on **molecule 0's own normal**,
which the stored frame carries, so the density factorises exactly::

    q(u0, u1, r_vec) = (1/4pi) * p(u1 | u0) * p(r_vec | u0, u1)

The pair is treated as *ordered* -- molecule 0 is the reference -- because that
is how the generator emits it. Do not symmetrise.

**Spin is factored out.** Rotating a molecule about its own disc normal is a
real degree of freedom of the atomistic structure but does not move
``(u0, u1, r_vec)``. Every component draws it uniformly, so it contributes the
same constant to all of them and cancels from the mixture. ``log_q`` is
therefore a density over the coarse-grained coordinates only.

Motif geometry is **derived** from the Cacelli ab initio minima rather than
hardcoded -- see :func:`motif_reference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

LOG_4PI = float(np.log(4.0 * np.pi))

# The reference benzene from ``build_reference_benzene`` lies in the xy-plane.
REFERENCE_NORMAL = np.array([0.0, 0.0, 1.0])


@lru_cache(maxsize=1)
def motif_reference():
    """``{family: (r, b, a_i, a_j)}`` at each Cacelli family's energy minimum.

    Read from the ab initio set through :mod:`asmcmc.utils.validation` so the
    numbers this module is tuned against are the same ones
    ``dimer_benchmark`` scores a potential on, rather than a second copy that
    can drift. Cached because it parses a file.

    For the record, the values it returns: cofacial ``r=3.90, a=1.000``;
    parallel-displaced ``r=3.85, a=0.909`` (stacking height 3.5 A, slip 1.6 A);
    T-shaped ``r=5.00, b=0.000, a_j=1.000``.
    """
    from asmcmc.utils.validation import _family_masks, load_cacelli_dimers

    data = load_cacelli_dimers()
    out = {}
    for family, mask in _family_masks(data).items():
        idx = np.flatnonzero(mask)
        best = idx[np.argmin(data.energy_kcal[idx])]
        r_vec = data.r[best]
        r = float(np.linalg.norm(r_vec))
        r_hat = r_vec / r
        out[family] = (
            r,
            float(data.uhat1[best] @ data.uhat2[best]),
            float(r_hat @ data.uhat1[best]),
            float(r_hat @ data.uhat2[best]),
        )
    return out


# --- geometry helpers --------------------------------------------------------


def orthonormal_basis(u):
    """Two unit vectors completing ``u`` (shape ``(n, 3)``) to a right-handed frame."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    seed = np.where(
        (np.abs(u[:, 0]) > 0.9)[:, None],
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = seed - np.einsum("ni,ni->n", seed, u)[:, None] * u
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(u, e1)


def _in_plane(e1, e2, angle):
    return np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2


def rotation_to_normal(normals, spin, reference_normal=REFERENCE_NORMAL):
    """Rotations carrying ``reference_normal`` onto each of ``normals``.

    ``spin`` is an additional rotation about the target normal. It leaves the
    disc normal -- and so the whole coarse-grained geometry -- unchanged, but it
    does move the atoms, so it is a genuine degree of freedom of the atomistic
    configuration and is drawn uniformly.
    """
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    spin = np.atleast_1d(np.asarray(spin, dtype=float))
    n0 = np.asarray(reference_normal, dtype=float)

    axis = np.cross(np.broadcast_to(n0, normals.shape), normals)
    cos = normals @ n0
    eye = np.broadcast_to(np.eye(3), (len(normals), 3, 3))

    K = _skew(axis)
    # Rodrigues in the (I + K + K^2/(1+cos)) form. The 1+cos denominator blows
    # up for an exactly antiparallel target, handled separately below.
    with np.errstate(divide="ignore", invalid="ignore"):
        align = eye + K + K @ K / (1.0 + cos)[:, None, None]

    flipped = cos < -1.0 + 1e-12
    if np.any(flipped):
        # A 180 degree turn about any axis perpendicular to n0.
        perp = orthonormal_basis(np.broadcast_to(n0, (int(flipped.sum()), 3)))[0]
        align[flipped] = 2.0 * np.einsum("ni,nj->nij", perp, perp) - np.eye(3)

    degenerate = cos > 1.0 - 1e-12
    if np.any(degenerate):
        align[degenerate] = np.eye(3)

    K_spin = _skew(normals)
    spin_rot = (
        eye
        + np.sin(spin)[:, None, None] * K_spin
        + (1.0 - np.cos(spin))[:, None, None] * (K_spin @ K_spin)
    )
    return spin_rot @ align


def _skew(v):
    zero = np.zeros(len(v))
    return np.stack(
        [
            np.stack([zero, -v[:, 2], v[:, 1]], axis=-1),
            np.stack([v[:, 2], zero, -v[:, 0]], axis=-1),
            np.stack([-v[:, 1], v[:, 0], zero], axis=-1),
        ],
        axis=1,
    )


# --- the two building-block densities ----------------------------------------


class AxialConcentration:
    """Density on the sphere depending only on ``t = u . v``, as ``exp(kappa t^2)``.

    ``kappa > 0`` concentrates near the poles (``|t| -> 1``, aligned),
    ``kappa < 0`` near the equator (``t -> 0``, perpendicular), ``kappa = 0`` is
    uniform. The ``t^2`` makes it **even in t**, which is required: a disc
    normal is defined only up to sign, so a density that distinguished ``u``
    from ``-u`` would assign two different numbers to one physical geometry.

    Normalised over the sphere (``2 pi int_-1^1 f dt = 1``) by Gauss-Legendre
    quadrature rather than the ``erfi`` closed form -- exact to machine
    precision for an integrand this smooth, and one fewer special function to
    get wrong. The linear grid is only for inverse-CDF sampling, where its
    accuracy requirement is much weaker.
    """

    def __init__(self, kappa, n_grid=8001, n_quadrature=256):
        self.kappa = float(kappa)
        # Shift the exponent before exponentiating so large kappa cannot overflow.
        self._shift = max(self.kappa, 0.0)

        nodes, quad_weights = np.polynomial.legendre.leggauss(n_quadrature)
        self._integral = float(
            quad_weights @ np.exp(self.kappa * nodes**2 - self._shift)
        )
        self.log_norm = float(np.log(2.0 * np.pi) + np.log(self._integral) + self._shift)

        self._t = np.linspace(-1.0, 1.0, n_grid)
        weight = np.exp(self.kappa * self._t**2 - self._shift)
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(self._t))]
        )
        self._cdf = cdf / cdf[-1]

    def log_pdf(self, t):
        """Log density **on the sphere** at a point with ``u . v = t``."""
        t = np.clip(np.asarray(t, dtype=float), -1.0, 1.0)
        return self.kappa * t**2 - self.log_norm

    def sample_cosine(self, rng, size):
        return np.interp(rng.random(size), self._cdf, self._t)

    def sample_direction(self, rng, axis):
        """Draw unit vectors around ``axis`` (shape ``(n, 3)``) from this density."""
        axis = np.atleast_2d(np.asarray(axis, dtype=float))
        t = self.sample_cosine(rng, len(axis))
        e1, e2 = orthonormal_basis(axis)
        azimuth = rng.uniform(0.0, 2.0 * np.pi, size=len(axis))
        return np.sqrt(np.clip(1.0 - t**2, 0.0, None))[:, None] * _in_plane(
            e1, e2, azimuth
        ) + t[:, None] * axis


@dataclass(frozen=True)
class RadialProposal:
    """The existing ``sample_radius`` behaviour, as an evaluable density.

    Mirrors ``cluster_dataset.sample_radius`` so the defensive uniform component
    reproduces exactly what the current generator does. The 100-attempt retry
    loop there falls through to volume-uniform if every draw misses the window;
    that happens with probability ~1e-160 and is ignored here.
    """

    lo: float = 3.4
    hi: float = 15.0
    mode: str = "mixture"
    compact_probability: float = 0.70
    mu: float = 4.70
    sigma: float = 0.60
    cap: float = 6.0

    @property
    def _window(self):
        return self.lo, min(self.cap, self.hi)

    def _log_volume_uniform(self, r):
        with np.errstate(divide="ignore"):
            return np.log(3.0 * r**2) - np.log(self.hi**3 - self.lo**3)

    def pdf(self, r):
        r = np.asarray(r, dtype=float)
        inside = (r >= self.lo) & (r <= self.hi)
        volume_uniform = np.where(inside, np.exp(self._log_volume_uniform(np.abs(r))), 0.0)
        if self.mode != "mixture":
            return volume_uniform

        a, b = self._window
        mass = norm.cdf(b, self.mu, self.sigma) - norm.cdf(a, self.mu, self.sigma)
        truncated = np.where(
            (r >= a) & (r <= b), norm.pdf(r, self.mu, self.sigma) / mass, 0.0
        )
        return (
            self.compact_probability * truncated
            + (1.0 - self.compact_probability) * volume_uniform
        )

    def log_pdf(self, r):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(r))

    def sample(self, rng, size):
        out = (self.lo**3 + rng.random(size) * (self.hi**3 - self.lo**3)) ** (1.0 / 3.0)
        if self.mode != "mixture":
            return out
        a, b = self._window
        use = rng.random(size) < self.compact_probability
        for i in np.flatnonzero(use):
            for _ in range(100):
                r = rng.normal(self.mu, self.sigma)
                if a <= r <= b:
                    out[i] = r
                    break
        return out


# --- pair proposal components ------------------------------------------------


class PairProposal:
    """A normalised density over ``(u0, u1, r_vec)`` that can also draw from itself."""

    name = "pair"

    def draw(self, rng, size):
        raise NotImplementedError

    def log_density(self, u0, u1, r_vec):
        raise NotImplementedError


@dataclass(frozen=True)
class StackedProposal(PairProposal):
    """Near-parallel discs, parameterised by stacking height and slip.

    Deliberately **one continuous family** rather than separate cofacial and
    parallel-displaced components: the Cacelli minima differ only by slip
    (0.0 A vs 1.6 A at essentially the same stacking height), so splitting them
    into discrete labels invents a boundary the physics does not have. Two
    instances with different ``slip_max`` cover the tight and the far-slipped
    regions with independent mixture weights.

    ``r_vec`` is built in cylindrical coordinates about ``u0``: height
    ``h = r_vec . u0``, slip ``s = |r_vec - h u0|``, azimuth uniform. The
    Jacobian is ``d3r = s ds dphi dh``, so the Lebesgue density carries a
    ``1/(2 pi s)``. Drawing ``s`` uniformly *in area* (``p(s) = 2s/s_max^2``)
    cancels that ``s`` exactly, which is what keeps the density finite on the
    axis -- a slip density that did not vanish at ``s = 0`` would diverge there.
    """

    kappa_align: float = 12.0
    height_min: float = 3.2
    height_max: float = 5.5
    slip_max: float = 2.0
    name: str = "stacked"

    @property
    def _alignment(self):
        return _axial(self.kappa_align)

    def draw(self, rng, size):
        u0 = _uniform_sphere(rng, size)
        u1 = self._alignment.sample_direction(rng, u0)

        e1, e2 = orthonormal_basis(u0)
        height = rng.uniform(self.height_min, self.height_max, size=size)
        height *= np.where(rng.random(size) < 0.5, -1.0, 1.0)
        # sqrt makes this uniform over the disc of radius slip_max.
        slip = self.slip_max * np.sqrt(rng.random(size))
        azimuth = rng.uniform(0.0, 2.0 * np.pi, size=size)

        r_vec = height[:, None] * u0 + slip[:, None] * _in_plane(e1, e2, azimuth)
        return u0, u1, r_vec

    def log_density(self, u0, u1, r_vec):
        height = np.einsum("ni,ni->n", r_vec, u0)
        slip = np.linalg.norm(r_vec - height[:, None] * u0, axis=1)

        supported = (
            (np.abs(height) >= self.height_min)
            & (np.abs(height) <= self.height_max)
            & (slip <= self.slip_max)
        )
        # p_h * p_s / (2 pi s) with p_h = 1/(2 dh) and p_s = 2s/s_max^2: the slip
        # cancels and what remains is flat over the cylinder.
        log_position = -np.log(
            2.0 * (self.height_max - self.height_min)
        ) - np.log(np.pi * self.slip_max**2)

        value = (
            -LOG_4PI
            + self._alignment.log_pdf(np.einsum("ni,ni->n", u0, u1))
            + log_position
        )
        return np.where(supported, value, -np.inf)


@dataclass(frozen=True)
class TShapedProposal(PairProposal):
    """One disc edge-on to the other: normals perpendicular, separation along
    the second normal.

    Matches the Cacelli T-shaped minimum (``b = 0``, ``a_i = 0``, ``a_j = 1``).
    Its radial window starts at 4.2 A because a T contact is geometrically
    impossible much below that -- the edge-on ring runs into the face -- which
    is why it needs a window of its own rather than sharing the stacked one.
    """

    kappa_perp: float = -12.0
    kappa_axis: float = 12.0
    r_min: float = 4.2
    r_max: float = 6.5
    name: str = "t_shaped"

    @property
    def _perpendicular(self):
        return _axial(self.kappa_perp)

    @property
    def _axis(self):
        return _axial(self.kappa_axis)

    def draw(self, rng, size):
        u0 = _uniform_sphere(rng, size)
        u1 = self._perpendicular.sample_direction(rng, u0)
        r_hat = self._axis.sample_direction(rng, u1)
        r = rng.uniform(self.r_min, self.r_max, size=size)
        return u0, u1, r[:, None] * r_hat

    def log_density(self, u0, u1, r_vec):
        r = np.linalg.norm(r_vec, axis=1)
        supported = (r >= self.r_min) & (r <= self.r_max) & (r > 0)
        safe_r = np.where(r > 0, r, 1.0)
        r_hat = r_vec / safe_r[:, None]

        # d3r = r^2 dr dOmega, hence the -2 log r.
        value = (
            -LOG_4PI
            + self._perpendicular.log_pdf(np.einsum("ni,ni->n", u0, u1))
            + self._axis.log_pdf(np.einsum("ni,ni->n", r_hat, u1))
            - np.log(self.r_max - self.r_min)
            - 2.0 * np.log(safe_r)
        )
        return np.where(supported, value, -np.inf)


@dataclass(frozen=True)
class UniformProposal(PairProposal):
    """The current generator: both normals uniform, direction uniform, radius
    from ``RadialProposal``.

    **Keep this in any mixture.** It is what guarantees support everywhere, and
    support is the one property reweighting cannot restore -- a target density
    can always be recovered from a badly *shaped* proposal, never from a region
    that was never sampled. It also bounds the importance weights.
    """

    radial: RadialProposal = RadialProposal()
    name: str = "uniform"

    def draw(self, rng, size):
        u0 = _uniform_sphere(rng, size)
        u1 = _uniform_sphere(rng, size)
        r_hat = _uniform_sphere(rng, size)
        r = self.radial.sample(rng, size)
        return u0, u1, r[:, None] * r_hat

    def log_density(self, u0, u1, r_vec):
        r = np.linalg.norm(r_vec, axis=1)
        supported = r > 0
        safe_r = np.where(supported, r, 1.0)
        with np.errstate(divide="ignore"):
            value = (
                -3.0 * LOG_4PI
                + self.radial.log_pdf(safe_r)
                - 2.0 * np.log(safe_r)
            )
        return np.where(supported, value, -np.inf)


@dataclass(frozen=True)
class MixtureProposal:
    """Weighted mixture of :class:`PairProposal` components.

    The component is redrawn on every attempt, including after a hard-core
    rejection. That matters for the recorded density: with a single top-level
    rejection the realised density is ``q(x) * 1[no clash] / Z`` for one global
    ``Z``, which cancels from every *relative* weight. Retrying inside a
    component would instead leave per-component normalisations that do not
    cancel and would have to be estimated from rejection counts.
    """

    components: tuple
    weights: tuple

    def __post_init__(self):
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have equal length")
        if len(self.components) == 0:
            raise ValueError("a mixture needs at least one component")
        if min(self.weights) < 0:
            raise ValueError("weights must be non-negative")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError(f"weights must sum to 1, got {sum(self.weights)}")

    @property
    def names(self):
        return tuple(c.name for c in self.components)

    def draw(self, rng, size=1):
        """``(u0, u1, r_vec, component_index)``, each of length ``size``."""
        which = rng.choice(len(self.components), size=size, p=np.asarray(self.weights))
        u0 = np.empty((size, 3))
        u1 = np.empty((size, 3))
        r_vec = np.empty((size, 3))
        for k, component in enumerate(self.components):
            picked = np.flatnonzero(which == k)
            if len(picked) == 0:
                continue
            a, b, c = component.draw(rng, len(picked))
            u0[picked], u1[picked], r_vec[picked] = a, b, c
        return u0, u1, r_vec, which

    def log_density(self, u0, u1, r_vec):
        u0 = np.atleast_2d(np.asarray(u0, dtype=float))
        u1 = np.atleast_2d(np.asarray(u1, dtype=float))
        r_vec = np.atleast_2d(np.asarray(r_vec, dtype=float))

        terms = np.stack(
            [c.log_density(u0, u1, r_vec) for c in self.components], axis=0
        )
        log_weights = np.log(np.asarray(self.weights))[:, None]
        return logsumexp(terms + log_weights, axis=0)


def default_mixture(radial=None, uniform_weight=0.40):
    """The recommended mixture: tight stack, wide stack, T-shaped, uniform.

    Weights are a starting point to be tuned on a small pilot, not an optimum.
    ``uniform_weight`` is the defensive share -- lowering it sharpens the
    proposal but widens the importance weights, so check the realised weight
    spread before reducing it.
    """
    remainder = 1.0 - uniform_weight
    if not 0.0 < uniform_weight <= 1.0:
        raise ValueError("uniform_weight must be in (0, 1]")

    components = (
        StackedProposal(slip_max=2.0, name="stacked_tight"),
        StackedProposal(slip_max=5.0, height_max=6.0, name="stacked_wide"),
        TShapedProposal(),
        UniformProposal(radial=radial or RadialProposal()),
    )
    weights = (0.35 * remainder, 0.25 * remainder, 0.40 * remainder, uniform_weight)
    return MixtureProposal(components=components, weights=weights)


@lru_cache(maxsize=8)
def _axial(kappa):
    """Cached: building one costs an 8001-point quadrature."""
    return AxialConcentration(kappa)


def _uniform_sphere(rng, size):
    v = rng.normal(size=(size, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
