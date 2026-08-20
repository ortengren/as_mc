"""One geometry generator per contact motif.

``make_cluster`` used to draw orientations uniformly and reject clashes. At close
approach that is almost all rejection: at 3.6 A only ~3% of random-orientation
placements clear a 2.4 A hard core, against ~86% for a cofacial-seeded one
(measured by ``cluster_analysis.hard_core_pass_rate``). Reshaping the *radial*
proposal cannot fix that -- the extra short-range draws are simply rejected and
redrawn -- because the hard core couples position to orientation. So position and
orientation have to be proposed together, which is what this module does.

**A generator per motif, and the campaign asks for a quota of each.** Composition
is then a stated fact of a campaign rather than an emergent property of a
sampling density, which is what lets a coverage problem be fixed by asking for
more of the motif that is short.

**Reversal, recorded deliberately.** An earlier version of this module fused
cofacial and parallel-displaced into a single ``StackedProposal`` spanning both,
arguing that the Cacelli minima "differ only by slip (0.0 A vs 1.6 A at
essentially the same stacking height), so splitting them into discrete labels
invents a boundary the physics does not have". That argument is overturned here
on evidence, not taste: ``cluster_analysis.motif_masks`` already cuts on exactly
that boundary, and the first AniSOAP Delta-learning sweep found the fit error
concentrated on one side of it -- every one of 64 hyperparameter points
over-bound the cofacial stack while leaving parallel-displaced almost exact. A
boundary you can measure error across is one you can also sample across.

**Generators and the classifier must agree.** Each ``draw_*`` returns a pair that
:func:`classify_pair` -- which applies ``cluster_analysis``'s own cuts -- labels
as that motif, verified per draw rather than hoped for. The previous components
did not have this property: they measured stacking height and slip about
molecule 0's normal while the classifier measures them about the *more aligned*
normal of the pair, and their radial windows spilled outside ``WELL_RANGE`` at
both ends, so a large share of every component's draws was classified as nothing
at all.

**Spin is a free degree of freedom.** Rotating a molecule about its own disc
normal moves the atoms but not ``(u0, u1, r_vec)``, so every generator leaves it
to the caller to draw uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import norm

LOG_4PI = float(np.log(4.0 * np.pi))

# The reference benzene from ``build_reference_benzene`` lies in the xy-plane.
REFERENCE_NORMAL = np.array([0.0, 0.0, 1.0])

# Motif names. These mirror ``cluster_analysis``'s constants rather than importing
# them, so this module stays a leaf that nothing in data_preparation has to be
# loaded before -- ``cluster_analysis`` imports ``cluster_dataset``, which imports
# this, so an import the other way would close a cycle. The numeric cuts are *not*
# duplicated (see :func:`_motif_cuts`), and a test pins these four strings against
# the classifier's.
COFACIAL = "cofacial"
T_SHAPED = "T-shaped"
PARALLEL_DISPLACED = "parallel-displaced"
FAR_SLIPPED = "far-slipped"
UNIFORM = "uniform"

MOTIFS = (COFACIAL, PARALLEL_DISPLACED, T_SHAPED, FAR_SLIPPED, UNIFORM)

# Draws are kept this far inside every classifier cut. The classifier measures
# slip about the pair's more-aligned normal, which need not be molecule 0's, so a
# draw placed exactly on a boundary can land on either side of it; a margin makes
# the round trip robust instead of marginal.
_MARGIN = 0.05


@lru_cache(maxsize=1)
def motif_reference():
    """``{family: (r, b, a_i, a_j)}`` at each Cacelli family's energy minimum.

    Read from the ab initio set through :mod:`asmcmc.utils.validation` so the
    numbers this module is tuned against are the same ones ``dimer_benchmark``
    scores a potential on, rather than a second copy that can drift. Cached
    because it parses a file.

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


@lru_cache(maxsize=1)
def _motif_cuts():
    """The classifier's own thresholds, imported once and cached.

    Deferred so this module imports nothing from ``data_preparation`` at module
    scope. There is exactly one definition of every number here -- the
    generators sample inside these cuts, and :func:`classify_pair` applies them.
    """
    from asmcmc.data_preparation import cluster_analysis as ca

    return {
        "well": ca.WELL_RANGE,
        "cofacial_max_slip": ca.COFACIAL_MAX_SLIP,
        "displaced_max_slip": ca.DISPLACED_MAX_SLIP,
        "parallel_min_b": ca.PARALLEL_MIN_B,
        "t_max_b": ca.T_MAX_B,
    }


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


@dataclass(frozen=True)
class RadialProposal:
    """Centre-centre separations: volume-uniform, or concentrated on the wells.

    The single radial sampler in the package -- ``cluster_dataset.sample_radius``
    was a second copy of this, kept in sync by hand, and is gone. ``pdf`` exists
    so ``sample`` can be checked against the density it claims to draw from.
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


# --- classification ----------------------------------------------------------


def pair_invariants(u0, u1, r_vec):
    """``(r, |b|, a_hi, a_lo, height, slip)`` for one pair.

    The classifier's coordinates exactly: ``a_hi``/``a_lo`` are the sorted
    absolute projections of the two normals onto ``r_hat`` (absolute because a
    disc normal is defined only up to sign, sorted because the pair is
    unordered), and height/slip resolve the separation about the **more aligned**
    normal -- not about ``u0``.
    """
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        return 0.0, 1.0, 1.0, 1.0, 0.0, 0.0
    r_hat = np.asarray(r_vec, dtype=float) / r
    abs_b = abs(float(np.dot(u0, u1)))
    proj = sorted((abs(float(np.dot(r_hat, u0))), abs(float(np.dot(r_hat, u1)))))
    a_lo, a_hi = proj
    height = r * a_hi
    slip = r * float(np.sqrt(max(1.0 - a_hi**2, 0.0)))
    return r, abs_b, a_hi, a_lo, height, slip


def classify_pair(u0, u1, r_vec):
    """The motif ``cluster_analysis.motif_masks`` would label this pair, or ``None``.

    ``None`` is a real outcome, not a failure: the cuts are not a partition, and
    the band ``0.35 <= |b| <= 0.8`` belongs to no motif.
    """
    cuts = _motif_cuts()
    lo, hi = cuts["well"]
    r, abs_b, _, a_lo, _, slip = pair_invariants(u0, u1, r_vec)

    if not (lo <= r < hi):
        return None
    if abs_b > cuts["parallel_min_b"]:
        if slip < cuts["cofacial_max_slip"]:
            return COFACIAL
        if slip <= cuts["displaced_max_slip"]:
            return PARALLEL_DISPLACED
        return FAR_SLIPPED
    if abs_b < cuts["t_max_b"] and a_lo < cuts["t_max_b"]:
        return T_SHAPED
    return None


# --- one generator per motif -------------------------------------------------


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _at_cosine(rng, axis, cosine):
    """A unit vector making the given cosine with ``axis``, azimuth uniform."""
    e1, e2 = orthonormal_basis(np.asarray(axis, dtype=float)[None])
    phi = rng.uniform(0.0, 2.0 * np.pi)
    perp = np.cos(phi) * e1[0] + np.sin(phi) * e2[0]
    return cosine * np.asarray(axis, dtype=float) + np.sqrt(
        max(1.0 - cosine**2, 0.0)
    ) * perp


def _stacked(rng, r, slip):
    """A near-parallel pair at a given separation and lateral offset.

    Both normals are placed near a common direction and ``r_vec`` is built from
    the requested slip, so ``|r_vec| = r`` exactly and the classifier -- which
    resolves about whichever normal is more aligned -- recovers the slip that was
    asked for whenever that normal is ``u0``. When it is ``u1`` instead the
    realised slip is slightly *smaller*, which is why the bands carry a margin
    and why :func:`draw_motif` verifies rather than assumes.
    """
    cuts = _motif_cuts()
    u0 = _unit(rng)
    # Uniform across the parallel band rather than concentrated at |b| = 1: the
    # aim is coverage of the motif, not a peak at its centre.
    cosine = rng.uniform(cuts["parallel_min_b"] + _MARGIN, 1.0)
    u1 = _at_cosine(rng, u0, cosine * (1.0 if rng.random() < 0.5 else -1.0))

    e1, e2 = orthonormal_basis(u0[None])
    phi = rng.uniform(0.0, 2.0 * np.pi)
    height = np.sqrt(max(r * r - slip * slip, 0.0))
    height *= 1.0 if rng.random() < 0.5 else -1.0
    r_vec = height * u0 + slip * (np.cos(phi) * e1[0] + np.sin(phi) * e2[0])
    return u0, u1, r_vec


def draw_cofacial(rng, radial=None):
    """Face-to-face: parallel normals, almost no lateral offset."""
    cuts = _motif_cuts()
    lo, hi = cuts["well"]
    # sqrt keeps the offset uniform over the disc it is drawn from.
    slip = (cuts["cofacial_max_slip"] - _MARGIN) * np.sqrt(rng.random())
    return _stacked(rng, rng.uniform(lo + _MARGIN, hi - _MARGIN), slip)


def draw_parallel_displaced(rng, radial=None):
    """Parallel normals slipped sideways -- benzene's deepest dimer well."""
    cuts = _motif_cuts()
    lo, hi = cuts["well"]
    slip = rng.uniform(
        cuts["cofacial_max_slip"] + _MARGIN, cuts["displaced_max_slip"] - _MARGIN
    )
    # r > slip always holds here: slip stays under 3 A and the well starts at 3.4.
    return _stacked(rng, rng.uniform(lo + _MARGIN, hi - _MARGIN), slip)


def draw_far_slipped(rng, radial=None):
    """Parallel normals, slipped past the displaced band -- nearly coplanar.

    Kept as its own motif because it is where the GB+Q baseline is worst
    (Delta rms 2.0 kcal/mol against parallel-displaced's 0.45), so it is the
    region a correction has most to learn from. The separation is drawn from the
    upper well only: at the bottom of the well a slip this large puts the two
    rings edge-to-edge inside the hard core, and every such draw would be
    rejected downstream.
    """
    cuts = _motif_cuts()
    _, hi = cuts["well"]
    r = rng.uniform(4.0, hi - _MARGIN)
    slip = rng.uniform(cuts["displaced_max_slip"] + _MARGIN, r - _MARGIN)
    return _stacked(rng, r, slip)


def draw_t_shaped(rng, radial=None):
    """Edge-on contact: normals perpendicular, separation along the second normal.

    The radial floor is 4.2 A because a T contact is geometrically impossible
    much below that -- the edge-on ring runs into the face.
    """
    cuts = _motif_cuts()
    _, hi = cuts["well"]
    u0 = _unit(rng)
    perpendicular = rng.uniform(-(cuts["t_max_b"] - _MARGIN), cuts["t_max_b"] - _MARGIN)
    u1 = _at_cosine(rng, u0, perpendicular)
    # Separation close to u1 makes a_j large and a_i small, which is what the
    # classifier's a_lo < 0.35 asks for.
    axial = rng.uniform(0.9, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
    r_hat = _at_cosine(rng, u1, axial)
    return u0, u1, rng.uniform(4.2, hi - _MARGIN) * r_hat


def draw_uniform(rng, radial=None):
    """Everything uniform, separation from ``radial``.

    **Keep a share of this in any campaign.** It is what gives the dataset
    support outside the well region at all -- the motif generators are confined
    to ``WELL_RANGE`` by construction, and no amount of reweighting recovers a
    region that was never sampled.
    """
    radial = radial or RadialProposal()
    return _unit(rng), _unit(rng), float(radial.sample(rng, 1)[0]) * _unit(rng)


MOTIF_DRAWS = {
    COFACIAL: draw_cofacial,
    PARALLEL_DISPLACED: draw_parallel_displaced,
    T_SHAPED: draw_t_shaped,
    FAR_SLIPPED: draw_far_slipped,
    UNIFORM: draw_uniform,
}


def draw_motif(motif, rng, radial=None, max_attempts=64):
    """``(u0, u1, r_vec)`` for one pair of the requested motif.

    Verified against :func:`classify_pair` and redrawn on a miss, so the campaign
    a caller asks for is the campaign the analysis will report. ``UNIFORM`` is
    exempt: it is deliberately unconstrained and most of its draws classify as no
    motif at all.
    """
    try:
        draw = MOTIF_DRAWS[motif]
    except KeyError:
        raise ValueError(
            f"unknown motif {motif!r}; expected one of {', '.join(MOTIFS)}"
        ) from None

    if motif == UNIFORM:
        return draw(rng, radial)

    for _ in range(max_attempts):
        u0, u1, r_vec = draw(rng, radial)
        if classify_pair(u0, u1, r_vec) == motif:
            return u0, u1, r_vec
    raise RuntimeError(
        f"{motif!r} generator failed to produce a configuration its own "
        f"classifier accepts in {max_attempts} attempts."
    )
