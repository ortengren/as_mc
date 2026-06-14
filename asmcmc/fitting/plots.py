"""Diagnostic plots for a fitted GBQ potential.

Each function takes the already-computed per-frame predictions and targets
(eV/molecule) rather than ``(theta, data)``, so a caller predicts once (e.g. via
``predict_per_mol``) and threads the same arrays through every figure. Figures
are written with the Agg backend -- this is a batch/reporting module, never an
interactive one.
"""

import os

import matplotlib

matplotlib.use("Agg")  # batch reporting: write figures to file, never open a window
import matplotlib.pyplot as plt
import numpy as np

from asmcmc.fitting.fit import predict_per_mol


def parity_plot(pred, target, train_idx=None, test_idx=None, path=None):
    """Predicted vs target per-molecule energy, with the ideal ``y = x`` line.

    When a ``train_idx`` / ``test_idx`` split is supplied the two partitions are
    drawn in different colours so over/under-fitting is visible; otherwise every
    frame is one series. Axes share an equal, square range so vertical deviation
    from the diagonal reads directly as prediction error. Returns the Figure and
    saves it to ``path`` (dpi 150) when given.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    if train_idx is None and test_idx is None:
        ax.scatter(target, pred, s=10, alpha=0.5, label="all")
    else:
        if train_idx is not None:
            ax.scatter(
                target[train_idx], pred[train_idx], s=10, alpha=0.5, label="train"
            )
        if test_idx is not None:
            ax.scatter(target[test_idx], pred[test_idx], s=10, alpha=0.5, label="test")

    lo = float(min(target.min(), pred.min()))
    hi = float(max(target.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("target energy (eV/molecule)")
    ax.set_ylabel("predicted energy (eV/molecule)")
    ax.set_title("GBQ parity")
    ax.legend()

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def nearest_neighbor_distance(data):
    """Smallest pair separation in each frame (length ``n_frames``, Angstrom).

    ``data.r_mag`` lists every directed pair distance over all frames;
    ``frame_index`` maps each to its frame. A segment-min collapses them to the
    closest contact per frame. Frames with no pairs within the cutoff stay
    ``inf`` (none in practice for a periodic crystal).
    """
    nn = np.full(data.n_frames, np.inf)
    np.minimum.at(nn, data.frame_index, data.r_mag)
    return nn


def residuals_vs_nn_distance(
    pred, target, data, train_idx=None, test_idx=None, path=None
):
    """Residual ``(pred - target)`` against each frame's nearest-neighbor distance.

    Small separations sit deep on the repulsive wall, where the ``r^-12`` term is
    most sensitive to ``sigma0``/``kappa``; residuals blowing up at the left edge
    flag a mis-fit short-range shape rather than a global energy-scale error.
    Train/test are separate series when a split is given. Returns the Figure and
    saves to ``path`` (dpi 150) when given.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    resid = pred - target
    nn = nearest_neighbor_distance(data)

    fig, ax = plt.subplots(figsize=(7, 5))
    if train_idx is None and test_idx is None:
        ax.scatter(nn, resid, s=10, alpha=0.5, label="all")
    else:
        if train_idx is not None:
            ax.scatter(nn[train_idx], resid[train_idx], s=10, alpha=0.5, label="train")
        if test_idx is not None:
            ax.scatter(nn[test_idx], resid[test_idx], s=10, alpha=0.5, label="test")

    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("nearest-neighbor distance (Angstrom)")
    ax.set_ylabel("residual pred - target (eV/molecule)")
    ax.set_title("GBQ residuals vs nearest-neighbor distance")
    ax.legend()

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def residuals_vs_energy(pred, target, train_idx=None, test_idx=None, path=None):
    """Residual ``(pred - target)`` against target energy, with a zero line.

    A horizontal band of points scattered evenly about zero is a clean fit; a
    trend (e.g. residuals sloping up at the most-bound, low-energy frames) flags
    a systematic bias the unweighted RMSE alone would hide. Train/test are drawn
    as separate series when a split is given. Returns the Figure and saves to
    ``path`` (dpi 150) when given.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    resid = pred - target

    fig, ax = plt.subplots(figsize=(7, 5))
    if train_idx is None and test_idx is None:
        ax.scatter(target, resid, s=10, alpha=0.5, label="all")
    else:
        if train_idx is not None:
            ax.scatter(
                target[train_idx], resid[train_idx], s=10, alpha=0.5, label="train"
            )
        if test_idx is not None:
            ax.scatter(target[test_idx], resid[test_idx], s=10, alpha=0.5, label="test")

    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("target energy (eV/molecule)")
    ax.set_ylabel("residual pred - target (eV/molecule)")
    ax.set_title("GBQ residuals vs energy")
    ax.legend()

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def write_plots(out_dir, theta, data, train_idx=None, test_idx=None, pred=None):
    """Write all diagnostic PNGs for a fitted ``theta`` to ``out_dir``.

    Predicts once (or reuses ``pred``) and threads the arrays through every
    figure -- the plotting analogue of :func:`asmcmc.fitting.report.write_artifacts`.
    Each Figure is closed after saving so a batch run does not accumulate open
    figures. Returns ``{name: path}`` for the files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    if pred is None:
        pred = predict_per_mol(theta, data)
    target = data.target_per_mol

    specs = {
        "parity": parity_plot(pred, target, train_idx, test_idx),
        "residuals_vs_energy": residuals_vs_energy(pred, target, train_idx, test_idx),
        "residuals_vs_nn_distance": residuals_vs_nn_distance(
            pred, target, data, train_idx, test_idx
        ),
    }
    paths = {}
    for name, fig in specs.items():
        path = os.path.join(out_dir, name + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = path
    return paths
