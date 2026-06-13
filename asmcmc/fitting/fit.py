"""Boltzmann-weighted Gay-Berne + quadrupole potential fit.

Objective (Cacelli et al. 2004, eq. 5):
    F = sum_k w_k (E_pred_k - E_target_k)^2 / sum_k w_k,
    with Boltzmann weights w_k = exp(-alpha * E_target_k).

Model (per molecule):
    E_pred = 0.5 * (sum over directed pairs of U_GBQ) / N_mol + E_intra

Parameter vector ``theta`` order:
    [sigma0, eps0, kappa, kappa_prime, mu, nu, Q, E_intra]
"""

import numpy as np

from asmcmc.fitting.data import gbq

PARAM_NAMES = ["sigma0", "eps0", "kappa", "kappa_prime", "mu", "nu", "Q", "E_intra"]

# Cacelli's empirical weight scale a = 0.4 / (kcal/mol), converted to 1/eV.
KCAL_PER_MOL_IN_EV = 0.0433641043
DEFAULT_ALPHA = 0.4 / KCAL_PER_MOL_IN_EV  # ~9.22 / eV


def predict_per_mol(theta, data):
    """Per-molecule energy for every frame (eV/molecule).

    The directed-pair sum is halved because extract_periodic_pairs lists every
    pair in both directions (see data.py); E_intra is added once per molecule.
    """
    sigma0, eps0, kappa, kappa_prime, mu, nu, Q, E_intra = theta
    pair_e = gbq(
        data.r_mag, data.a_i, data.a_j, data.b_ij,
        sigma0, eps0, kappa, kappa_prime, mu, nu, Q,
    )
    frame_e = np.bincount(data.frame_index, weights=pair_e, minlength=data.n_frames)
    return 0.5 * frame_e / data.n_mol + E_intra


def boltzmann_weights(target, alpha=DEFAULT_ALPHA):
    """w_k = exp(-alpha * E_k), with max-subtraction for numerical stability.

    The subtracted constant cancels in the normalised objective, so it only
    keeps the exponent finite on absolute DFT energies (E ~ -1600 eV). The
    most-bound (lowest-energy) frame receives the largest weight.
    """
    arg = -alpha * np.asarray(target, dtype=float)
    arg = arg - arg.max()
    return np.exp(arg)


def residuals(theta, data, weights, idx=None):
    """Weighted residual vector: sqrt(w_k / sum w) * (pred_k - target_k).

    least_squares minimises 0.5 * sum(residuals**2), which is the Cacelli merit
    function F up to the constant factor 0.5. ``idx`` restricts to a frame subset
    (e.g. the training split); ``None`` uses all frames.
    """
    pred = predict_per_mol(theta, data)
    target = data.target_per_mol
    if idx is not None:
        pred, target, weights = pred[idx], target[idx], weights[idx]
    w = weights / weights.sum()
    return np.sqrt(w) * (pred - target)
