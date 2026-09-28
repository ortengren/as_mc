"""Geometry primitives for cluster construction: rotations and radial sampling.

Formerly one generator per contact motif (cofacial / parallel-displaced /
T-shaped / far-slipped), each redrawing until :func:`classify_pair` confirmed
the requested label. Retired 2026-09: the two campaigns built on quota'd motif
composition both failed the UMA dimer-well gate, while the un-engineered
uniform campaign passed 63/64 -- composition was never the lever. What remains
is what :func:`draw_uniform` always was: uniform orientations, separation from
:class:`RadialProposal`, filtered downstream by geometry rather than by label
(``cluster_dataset.SamplingSettings.min_atom_distance``/``max_atom_distance``).

A uniform draw clears the hard core far less often than a motif-seeded one at
close range (measured: ~3% vs ~86% at 3.6 A), but the rejection is pure
geometry with no MLIP cost, so it costs CPU attempts, not budget -- see
``cluster_dataset.make_cluster``'s placement loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The reference benzene from ``build_reference_benzene`` lies in the xy-plane.
REFERENCE_NORMAL = np.array([0.0, 0.0, 1.0])


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


def rotation_to_normal(normals, spin, reference_normal=REFERENCE_NORMAL):
    """Rotations carrying ``reference_normal`` onto each of ``normals``.

    ``spin`` is an additional rotation about the target normal. It leaves the
    coarse-grained system unchanged, but it does move the atoms, so it is a genuine
    degree of freedom of the atomistic configuration and is drawn uniformly.
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
    """Centre-centre separations: volume-uniform, or concentrated on the wells."""

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


# --- the one remaining generator ----------------------------------------------


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def draw_uniform(rng, radial=None):
    """Both orientations and the separation drawn uniformly."""
    radial = radial or RadialProposal()
    return _unit(rng), _unit(rng), float(radial.sample(rng, 1)[0]) * _unit(rng)
