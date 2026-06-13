import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ase
import pytest

from asmcmc.potentials import gb, quadrupole, GB_PARAMS, QQ
from asmcmc.fitting.data import gbq, extract_periodic_pairs, FitData
from asmcmc.fitting.fit import (
    predict_per_mol,
    boltzmann_weights,
    residuals,
    DEFAULT_ALPHA,
)

# theta order: [sigma0, eps0, kappa, kappa_prime, mu, nu, Q, E_intra]
THETA = [*GB_PARAMS.values(), QQ, 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _random_unit_vectors(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _synthetic_fitdata(n_frames=4, pairs_per_frame=6, seed=1):
    """A small FitData with plausible-range geometry and absolute targets.

    The pair invariants are drawn in their valid ranges (|a|, |b| <= 1); their
    exact values are irrelevant to the algebra these tests exercise (shape, the
    E_intra offset, the residual/merit identity), only that gbq stays finite.
    """
    rng = np.random.default_rng(seed)
    P = n_frames * pairs_per_frame
    return FitData(
        r_mag=rng.uniform(6.0, 14.0, P),
        a_i=rng.uniform(-1.0, 1.0, P),
        a_j=rng.uniform(-1.0, 1.0, P),
        b_ij=rng.uniform(-1.0, 1.0, P),
        frame_index=np.repeat(np.arange(n_frames), pairs_per_frame),
        n_mol=rng.integers(1, 3, n_frames).astype(float),
        target_per_mol=rng.uniform(-1602.0, -1600.0, n_frames),
        cutoff=15.0,
    )


# ---------------------------------------------------------------------------
# Math fidelity: the duplicated gbq must equal the canonical potentials.py
# ---------------------------------------------------------------------------

def test_gbq_matches_potentials_gb_plus_quadrupole():
    """gbq(invariants) == potentials.gb + quadrupole(vectors) on random pairs.

    Guards the vectorised re-implementation in data.py from drifting away from
    the exact functions the MC uses.
    """
    rng = np.random.default_rng(0)
    n = 300
    u1 = _random_unit_vectors(n, rng)
    u2 = _random_unit_vectors(n, rng)
    r_mag = rng.uniform(5.5, 13.0, n)
    r_vec = _random_unit_vectors(n, rng) * r_mag[:, None]

    r_hat = r_vec / r_mag[:, None]
    a_i = np.einsum("pk,pk->p", r_hat, u1)
    a_j = np.einsum("pk,pk->p", r_hat, u2)
    b_ij = np.einsum("pk,pk->p", u1, u2)

    got = gbq(r_mag, a_i, a_j, b_ij, *GB_PARAMS.values(), QQ)
    ref = gb(u1, u2, r_vec, **GB_PARAMS) + np.squeeze(quadrupole(u1, u2, r_vec, QQ))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Lattice-sum foundation: extraction + the 1/2 vs an independent image sum
# ---------------------------------------------------------------------------

def test_self_image_lattice_sum_matches_bruteforce():
    """predict_per_mol on a 1-particle crystal == a brute-force self-image sum.

    A single particle in a small cubic cell interacts only with its own
    periodic images. extract_periodic_pairs + predict_per_mol's 0.5 factor must
    reproduce an independent triple-loop over image shifts evaluated with the
    canonical potentials.py functions.
    """
    L = 7.0
    cutoff = 16.0
    u = _unit([1.0, 2.0, 3.0])
    frame = ase.Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.diag([L, L, L]), pbc=True)
    frame.new_array("or_vec", u[None, :].copy())

    pairs = extract_periodic_pairs(frame, "or_vec", cutoff)
    data = FitData(
        r_mag=pairs[:, 0],
        a_i=pairs[:, 1],
        a_j=pairs[:, 2],
        b_ij=pairs[:, 3],
        frame_index=np.zeros(len(pairs), dtype=int),
        n_mol=np.array([1.0]),
        target_per_mol=np.array([0.0]),
        cutoff=cutoff,
    )
    pred = predict_per_mol(THETA, data)[0]

    # Independent self-image lattice sum (potentials.py math). Each image is one
    # directed pair; the per-molecule energy is half their sum.
    total = 0.0
    n_images = 0
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                if n1 == n2 == n3 == 0:
                    continue
                r_vec = np.array([[n1 * L, n2 * L, n3 * L]])
                if np.linalg.norm(r_vec) >= cutoff:
                    continue
                total += gb(u[None], u[None], r_vec, **GB_PARAMS).item()
                total += np.squeeze(quadrupole(u[None], u[None], r_vec, QQ)).item()
                n_images += 1

    np.testing.assert_allclose(pred, 0.5 * total, rtol=1e-9, atol=1e-12)
    # All shells with sum(n^2) <= 5 fit inside 16 Å at L = 7 -> 56 images.
    assert len(pairs) == n_images == 56


# ---------------------------------------------------------------------------
# Boltzmann weights: finite on absolute energies, physically ordered, invariant
# ---------------------------------------------------------------------------

def test_boltzmann_weights_stable_and_invariant():
    """Weights stay finite on ~-1600 eV targets and are shift-invariant."""
    target = np.array([-1601.0, -1600.5, -1602.3, -1599.8, -1601.7])
    w = boltzmann_weights(target)

    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    # most-bound (lowest-energy) frame receives the largest weight
    assert np.argmax(w) == np.argmin(target)
    # exact stabilised form: w = exp(-alpha (E - min E))
    # (rtol loosened from machine eps for the differing float associativity:
    # the code subtracts (-alpha E).max(), the reference factors out E.min())
    np.testing.assert_allclose(
        w, np.exp(-DEFAULT_ALPHA * (target - target.min())), rtol=1e-9
    )
    # a naive exp(-alpha E) overflows at these energies; the stabilised one must not
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(-DEFAULT_ALPHA * target)).all()
    # normalised weights are invariant to a global shift of every energy
    w_shift = boltzmann_weights(target + 137.0)
    np.testing.assert_allclose(w / w.sum(), w_shift / w_shift.sum(), rtol=1e-9)


# ---------------------------------------------------------------------------
# predict_per_mol: shape and the additive E_intra offset
# ---------------------------------------------------------------------------

def test_predict_per_mol_shape_and_offset():
    """One prediction per frame; E_intra shifts every prediction equally."""
    data = _synthetic_fitdata()
    pred = predict_per_mol(THETA, data)

    assert pred.shape == (data.n_frames,)
    assert np.all(np.isfinite(pred))

    shifted = list(THETA)
    shifted[-1] = THETA[-1] + 5.0
    np.testing.assert_allclose(predict_per_mol(shifted, data) - pred, 5.0, atol=1e-9)


# ---------------------------------------------------------------------------
# residuals: the least-squares bridge to Cacelli's merit function F
# ---------------------------------------------------------------------------

def test_residuals_equal_cacelli_merit():
    """sum(residuals**2) == sum_k (w_k / sum w)(pred_k - E_k)**2 = F."""
    data = _synthetic_fitdata()
    weights = boltzmann_weights(data.target_per_mol)
    res = residuals(THETA, data, weights)

    pred = predict_per_mol(THETA, data)
    w = weights / weights.sum()
    F = np.sum(w * (pred - data.target_per_mol) ** 2)
    np.testing.assert_allclose(np.sum(res ** 2), F, rtol=1e-12)


def test_residuals_idx_subsets_and_renormalises():
    """idx restricts to a frame subset and renormalises weights within it."""
    data = _synthetic_fitdata()
    weights = boltzmann_weights(data.target_per_mol)
    idx = np.array([0, 2])
    res = residuals(THETA, data, weights, idx=idx)

    assert res.shape == (len(idx),)
    pred = predict_per_mol(THETA, data)[idx]
    target = data.target_per_mol[idx]
    w = weights[idx] / weights[idx].sum()
    np.testing.assert_allclose(np.sum(res ** 2), np.sum(w * (pred - target) ** 2), rtol=1e-12)
