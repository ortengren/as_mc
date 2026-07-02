"""Cross-run summary plots for the GBQ fit campaign.

Where :mod:`asmcmc.fitting.plots` makes per-fit diagnostics (one fitted ``theta``
-> a few figures, written inside ``fit_variant``), this module is the *cross-run*
analogue: it scans the ``results/fitting/`` campaign tree produced by
``scripts/run_fits.sh`` (``alpha_scan/alpha_<value>/`` + ``alpha_scan/uniform/``)
and ``scripts/run_fit_seeds.sh`` (``multiseed/<campaign>/seed_<n>/``, the
seed-reproducibility repeats), rebuilds the dataset once from the shared cache
(``results/fitting/cache/``), recomputes per-frame predictions/weights for each
fit, and reduces the campaign to a handful of comparison figures.

Run with ``python -m asmcmc.fitting.summary``.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # batch reporting: write figures to file, never open a window
import matplotlib.pyplot as plt
import numpy as np

from asmcmc.fitting.data import build_dataset
from asmcmc.fitting.fit import (
    PARAM_NAMES,
    boltzmann_weights,
    predict_per_mol,
    train_test_split,
)
from asmcmc.fitting.run import DEFAULT_CUTOFF, DEFAULT_DATA
from asmcmc.potentials import gb, quadrupole

DEFAULT_RESULTS_ROOT = "results/fitting"
DEFAULT_OUT = "results/fitting/summary"
# Campaign subdirs under the fitting root.
ALPHA_SCAN_SUBDIR = "alpha_scan"
MULTISEED_SUBDIR = "multiseed"
# Multi-seed campaigns whose seeds back the reproducibility figures: the uniform
# (unweighted) production fit and the alpha=2.90 Boltzmann fit, kept only as a
# weighted-sweep reference.
UNIFORM_CAMPAIGN = "uniform"
BOLTZMANN_CAMPAIGN = "alpha_2.90"
# The split is held fixed across the whole campaign (run_fit_seeds.sh pins
# --split-seed 0; run_fits.sh leaves it at the default 0), so the held-out test
# frames are identical for every run and we can reconstruct them here.
SPLIT_SEED = 0
TEST_FRAC = 0.2
# A representative sweep alpha used only to illustrate Boltzmann-weighting
# behaviour (effective sample size, weight concentration). The production fit is
# the uniform (unweighted) one, so this is a reference point, not the chosen
# weighting.
REFERENCE_ALPHA = 2.90


# --------------------------------------------------------------------------- #
# Loading helpers: turn a results/ directory into (theta, dataset, split).
# --------------------------------------------------------------------------- #
def load_theta(params_path):
    """Load a fitted parameter vector from a ``params.json`` (order PARAM_NAMES)."""
    with open(params_path) as f:
        params = json.load(f)
    return np.array([params[name]["value"] for name in PARAM_NAMES], dtype=float)


def _find_params(run_dir):
    """Return the ``params.json`` path inside a run dir, or ``None``.

    A run writes its fit under ``<run_dir>/<weighting>/`` (``boltzmann`` or
    ``uniform``); dirs that hold only a cache return ``None`` and are skipped by
    the discovery helpers.
    """
    for weighting in ("boltzmann", "uniform"):
        p = os.path.join(run_dir, weighting, "params.json")
        if os.path.exists(p):
            return p
    return None


def discover_alpha_runs(results_root=DEFAULT_RESULTS_ROOT):
    """Map ``alpha -> theta`` for every ``alpha_scan/alpha_<value>/`` run.

    Parses the alpha out of the directory name (``alpha_2.90`` -> 2.90); the
    ``uniform`` sibling does not match and is handled by
    :func:`discover_uniform_run`.
    """
    runs = {}
    scan = os.path.join(results_root, ALPHA_SCAN_SUBDIR)
    for d in sorted(glob.glob(os.path.join(scan, "alpha_*"))):
        m = re.fullmatch(r"alpha_([0-9.]+)", os.path.basename(d))
        if m is None:
            continue
        p = _find_params(d)
        if p is not None:
            runs[float(m.group(1))] = load_theta(p)
    return runs


def discover_uniform_run(results_root=DEFAULT_RESULTS_ROOT):
    """Return the uniform (unweighted) fit's ``theta``, or ``None`` if absent."""
    p = _find_params(os.path.join(results_root, ALPHA_SCAN_SUBDIR, "uniform"))
    return load_theta(p) if p is not None else None


def discover_boltzmann_run(results_root=DEFAULT_RESULTS_ROOT):
    """Return the alpha=2.90 Boltzmann fit's ``theta``, or ``None``.

    This is the weighted-sweep reference (no longer the production choice, which
    is the uniform fit -- see :func:`discover_uniform_production_run`). Prefers
    the seed-0 finalisation run and falls back to the alpha-sweep's 2.90 point.
    """
    cands = (
        os.path.join(results_root, MULTISEED_SUBDIR, BOLTZMANN_CAMPAIGN, "seed_0"),
        os.path.join(results_root, ALPHA_SCAN_SUBDIR, "alpha_2.90"),
    )
    for cand in cands:
        p = _find_params(cand)
        if p is not None:
            return load_theta(p)
    return None


def discover_uniform_production_run(results_root=DEFAULT_RESULTS_ROOT):
    """Return the finalised uniform fit's ``theta``, or ``None``.

    The uniform analogue of :func:`discover_boltzmann_run`: prefers the
    higher-budget multi-seed finalisation (``multiseed/uniform/seed_0``) and
    falls back to the alpha-sweep's same-budget ``uniform`` run.
    """
    cands = (
        os.path.join(results_root, MULTISEED_SUBDIR, UNIFORM_CAMPAIGN, "seed_0"),
        os.path.join(results_root, ALPHA_SCAN_SUBDIR, "uniform"),
    )
    for cand in cands:
        p = _find_params(cand)
        if p is not None:
            return load_theta(p)
    return None


def discover_seed_runs(results_root=DEFAULT_RESULTS_ROOT, campaign=UNIFORM_CAMPAIGN):
    """Map ``seed -> theta`` for every ``multiseed/<campaign>/seed_<n>/`` run.

    These are the ``run_fit_seeds.sh`` repeats: identical settings and split,
    only the differential_evolution seed varies, so they are a clean
    reproducibility check on the located minimum.
    """
    runs = {}
    base = os.path.join(results_root, MULTISEED_SUBDIR, campaign)
    for d in sorted(glob.glob(os.path.join(base, "seed_*"))):
        m = re.fullmatch(r"seed_([0-9]+)", os.path.basename(d))
        if m is None:
            continue
        p = _find_params(d)
        if p is not None:
            runs[int(m.group(1))] = load_theta(p)
    return runs


def load_dataset(dataset=DEFAULT_DATA, cutoff=DEFAULT_CUTOFF, cache_dir=None):
    """Build the :class:`FitData`, reusing any existing per-run cache.

    The cached ``.npz`` is keyed by (file, cutoff, mtime) and holds only pair
    geometry, so the shared campaign cache is valid here and spares the slow
    neighbour-list rebuild over all 6826 frames. Falls back to that shared dir.
    """
    if cache_dir is None:
        cache_dir = os.path.join(DEFAULT_RESULTS_ROOT, "cache")
    return build_dataset(dataset, cutoff, cache_dir=cache_dir)


def _test_metrics_by_region(theta, data, test_idx):
    """Unweighted test RMSE / MAE / R^2, overall and split attractive vs repulsive.

    Judges a fit by a *common* yardstick -- the plain (unweighted) error on the
    shared held-out frames -- so weightings are directly comparable. The split is
    taken relative to the *dataset-mean* energy (``target_per_mol`` is the
    absolute ~-1601 eV/molecule pedestal, so a split at 0 is meaningless): frames
    below the mean are attractive / cohesive (~68%), those above are repulsive
    (~32%, the "repulsive" subset report.py refers to). This is the same energy
    ordering the Boltzmann weight ranks, so the split exposes the trade-off the
    weighting is meant to buy -- accuracy on the most-bound configurations at the
    expense of the rest. Returns ``{region: {rmse, mae, r2, n}}`` for
    ``all`` / ``attractive`` / ``repulsive``.
    """
    pred = predict_per_mol(theta, data)
    target = data.target_per_mol
    mean = float(target.mean())  # dataset-wide reference (matches E_intra centring)
    test_idx = np.asarray(test_idx)
    regions = {
        "all": test_idx,
        "attractive": test_idx[target[test_idx] < mean],
        "repulsive": test_idx[target[test_idx] >= mean],
    }
    out = {}
    for name, idx in regions.items():
        err = pred[idx] - target[idx]
        ss_tot = (
            float(np.sum((target[idx] - target[idx].mean()) ** 2)) if idx.size else 0.0
        )
        out[name] = {
            "rmse": float(np.sqrt(np.mean(err**2))) if idx.size else float("nan"),
            "mae": float(np.mean(np.abs(err))) if idx.size else float("nan"),
            "r2": (
                (1.0 - float(np.sum(err**2)) / ss_tot) if ss_tot > 0 else float("nan")
            ),
            "n": int(idx.size),
        }
    return out


# --------------------------------------------------------------------------- #
# Figure 1: weighting comparison -- unweighted vs Boltzmann at varying alpha.
# --------------------------------------------------------------------------- #
def alpha_quality_plot(data, alpha_runs, uniform_theta, test_idx, path=None):
    """Held-out test error vs Boltzmann alpha, with uniform placed at alpha = 0.

    Every fit is scored on the same yardstick -- the *unweighted* test error on
    the shared held-out frames -- so the weighted and unweighted objectives are
    comparable (their raw objective values are not, each being normalised by its
    own ``sum(w)``). The uniform (unweighted) fit is the alpha -> 0 limit
    (``w_k = exp(-0 * E_k) = 1``), so it is plotted at x = 0 -- but as a *detached*
    marker, not joined to the weighted curve, since no fits were run between 0 and
    the lowest sampled alpha and a connecting line there would imply unsampled
    interpolation. Two panels (RMSE, R^2) each carry three curves: overall,
    attractive (below the dataset-mean energy), and repulsive (above it). A
    A vertical marker flags the production fit -- the uniform (unweighted) one at
    alpha = 0. A weighting that earns its keep should lower attractive-region
    error as alpha grows from 0, even as overall/repulsive error rises; here it
    does not, which is why uniform was chosen.
    """
    region_styles = {
        "all": ("C0", "o"),
        "attractive": ("C1", "s"),
        "repulsive": ("C2", "^"),
    }

    alphas = sorted(alpha_runs)
    metrics = {
        a: _test_metrics_by_region(alpha_runs[a], data, test_idx) for a in alphas
    }
    uniform = (
        _test_metrics_by_region(uniform_theta, data, test_idx)
        if uniform_theta is not None
        else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, ylabel in (
        (axes[0], "rmse", "test RMSE (eV/molecule)"),
        (axes[1], "r2", "test R^2"),
    ):
        for region, (color, marker) in region_styles.items():
            ys = [metrics[a][region][key] for a in alphas]
            ax.plot(alphas, ys, color=color, marker=marker, label=region)
            if uniform is not None:
                # Detached alpha=0 (uniform) point: same colour, no joining line.
                ax.plot(0.0, uniform[region][key], color=color, marker=marker)
        ax.axvline(0.0, color="k", ls=":", lw=1, label="production (uniform)")
        ax.set_xlabel(r"Boltzmann weight scale $\alpha$ (1/eV)  (0 = uniform)")
        ax.set_ylabel(ylabel)
        ax.legend(title="region")
    axes[0].set_title("Fit accuracy vs weighting")
    axes[1].set_title("Explained variance vs weighting")
    fig.suptitle("Unweighted vs Boltzmann-weighted fit on test data")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Figure 2: seed reproducibility -- does DE find the same minimum each time?
# --------------------------------------------------------------------------- #
def seed_reproducibility_plot(
    data, seed_runs, test_idx, label="uniform (production)", path=None
):
    """Agreement across differential_evolution seeds at fixed settings/split.

    The ``run_fit_seeds.sh`` repeats vary only the DE seed, so disagreement among
    them is the optimiser failing to relocate the same minimum -- not a data or
    model effect. ``label`` names the weighting campaign (uniform production vs
    the alpha=2.90 Boltzmann reference) for the figure title. Two panels:

    - **Left (parameter spread):** each parameter's three seed values as dots,
      with a min--max whisker, expressed as % deviation from that parameter's
      cross-seed mean (so the 8 wildly different scales/units share one axis).
      A well-determined, reproducible minimum sits as a tight cluster on the 0%
      line; a tall whisker flags a parameter the DE seed moves (a sloppy /
      poorly constrained direction). Unlike a single CV number, this shows the
      actual scatter -- e.g. one outlying seed vs. evenly split.
    - **Right (prediction agreement):** the empirical CDF of the per-frame
      cross-seed prediction spread (std of the predicted energy across seeds,
      meV/molecule) -- the fraction of test frames whose seeds agree to within a
      given tolerance. Parameters can differ slightly yet predict identically (a
      flat objective direction); this checks agreement at the level that actually
      feeds the MC. A curve that shoots to 1 hard against 0 (median + 95th pct
      marked) means the seeds are interchangeable. The energy axis is dropped
      because the spread is uniformly negligible, not because its location
      matters.

    Returns the Figure (and saves to ``path`` when given).
    """
    seeds = sorted(seed_runs)
    thetas = np.array([seed_runs[s] for s in seeds])  # (n_seeds, n_params)
    mean = thetas.mean(axis=0)
    # Per-parameter % deviation of each seed from the cross-seed mean: collapses
    # the 8 disparate scales/units onto one comparable axis centred at 0%.
    dev = 100.0 * (thetas - mean) / np.abs(mean)  # (n_seeds, n_params)

    # Predictions per seed, restricted to the shared held-out test frames.
    test_idx = np.asarray(test_idx)
    preds = np.array(
        [predict_per_mol(seed_runs[s], data)[test_idx] for s in seeds]
    )  # (n_seeds, n_test)
    pred_std_mev = 1000.0 * preds.std(axis=0)  # meV/molecule

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(PARAM_NAMES))
    ax = axes[0]
    ax.axhline(0.0, color="gray", lw=1)
    # Min-max whisker per parameter, with each seed's value overlaid as a dot.
    ax.vlines(x, dev.min(axis=0), dev.max(axis=0), color="gray", lw=1, zorder=1)
    for j, s in enumerate(seeds):
        ax.plot(x, dev[j], "o", ms=6, alpha=0.8, label="seed {}".format(s), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax.set_ylabel("deviation from cross-seed mean (%)")
    ax.set_title("Spread of each fitted parameter across seeds")
    ax.legend(title="DE seed", fontsize="small")

    # ECDF of the per-frame cross-seed prediction spread: fraction of test frames
    # whose seeds agree to within a given tolerance. Curve hard against 0 = the
    # seeds are interchangeable at the level (predicted energy) the MC consumes.
    spread_sorted = np.sort(pred_std_mev)
    ecdf = np.arange(1, spread_sorted.size + 1) / spread_sorted.size
    median = float(np.median(pred_std_mev))
    pct95 = float(np.percentile(pred_std_mev, 95))

    ax = axes[1]
    ax.plot(spread_sorted, ecdf, color="C1", lw=1.5)
    ax.plot([median], [0.5], "o", color="C0")
    ax.plot([pct95], [0.95], "o", color="C0")
    ax.annotate(
        "median {:.2g} meV/mol".format(median),
        xy=(median, 0.5),
        xytext=(0.4, 0.3),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.annotate(
        "95% agree to\nwithin {:.2g} meV/mol".format(pct95),
        xy=(pct95, 0.95),
        xytext=(0.45, 0.6),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("cross-seed prediction spread (meV/molecule)")
    ax.set_ylabel("fraction of test frames differing by less than x")
    ax.set_title(
        "Predictions agree across seeds (max {:.2g} meV/mol)".format(pred_std_mev.max())
    )

    fig.suptitle(
        "Reproducibility of the {} GBQ fit across the {} optimizer seeds {}".format(
            label, len(seeds), seeds
        )
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3A: how much of the dataset actually constrains a Boltzmann fit?
# --------------------------------------------------------------------------- #
def _kish_ess(weights):
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (a frame count).

    The number of equally-weighted frames that would carry the same statistical
    information as the given weights: ``N`` when weights are uniform, ``1`` when
    one frame holds all the weight. Invariant to the overall scale of ``weights``
    (so ``boltzmann_weights``' max-subtraction does not affect it).
    """
    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    return s * s / float(np.sum(w**2)) if s > 0 else 0.0


def weight_concentration_plot(
    data, train_idx, alpha_grid=None, sampled_alphas=(), path=None
):
    """How many frames actually constrain the Boltzmann-weighted fit.

    A Boltzmann objective puts almost all weight on the most-bound frames, so the
    fit can be determined by far fewer than the nominal 6826 frames -- the central
    validity question behind the alpha choice (fit.py: higher alpha "deprioritizes
    too much of the dataset"). Computed on the *training* frames, since that is
    what the objective sees. Two panels:

    - **Left (effective sample size vs alpha):** Kish ESS over a fine alpha grid,
      from ``N_train`` at alpha=0 (uniform) decaying toward 1. The reference
      alpha and the sampled sweep alphas are marked; a low ESS there means the
      potential rests on a small subset of configurations.
    - **Right (weight concentration at the reference alpha):** the cumulative
      ("Lorenz") curve -- fraction of total objective weight held by the top-k
      heaviest frames -- against the uniform diagonal. Annotated with the share of
      frames that together hold 90% of the weight.

    This concentration is exactly why the production fit is the *uniform* one:
    the reference alpha is shown only to make that trade-off visible.

    Returns the Figure (and saves to ``path`` when given).
    """
    target_tr = data.target_per_mol[np.asarray(train_idx)]
    n_train = target_tr.size
    if alpha_grid is None:
        hi = max([REFERENCE_ALPHA, *sampled_alphas]) * 1.05
        alpha_grid = np.linspace(0.0, hi, 200)

    ess = np.array([_kish_ess(boltzmann_weights(target_tr, a)) for a in alpha_grid])
    ess_ref = _kish_ess(boltzmann_weights(target_tr, REFERENCE_ALPHA))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(alpha_grid, ess, color="C0")
    ax.axhline(
        n_train, color="gray", ls=":", lw=1, label="all {} train frames".format(n_train)
    )
    ax.axvline(REFERENCE_ALPHA, color="k", ls=":", lw=1)
    for a in sorted(sampled_alphas):
        ax.plot(a, _kish_ess(boltzmann_weights(target_tr, a)), "o", color="C1")
    ax.annotate(
        "reference alpha={:.2f}\nESS = {:.0f} of {} ({:.1f}%)".format(
            REFERENCE_ALPHA, ess_ref, n_train, 100 * ess_ref / n_train
        ),
        xy=(REFERENCE_ALPHA, ess_ref),
        xytext=(0.45, 0.55),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    # Linear axis anchored at 0 so the height reads directly as the retained
    # fraction of the dataset (a log axis would compress exactly that).
    ax.set_ylim(0, n_train * 1.05)
    ax.set_xlabel(r"Boltzmann weight scale $\alpha$ (1/eV)  (0 = uniform)")
    ax.set_ylabel("effective sample size (frames)")
    ax.set_title("How many frames effectively constrain the fit")
    ax.legend()

    ax = axes[1]
    w = np.sort(boltzmann_weights(target_tr, REFERENCE_ALPHA))[::-1]
    cum_w = np.cumsum(w) / w.sum()
    frac_frames = np.arange(1, n_train + 1) / n_train
    # Share of (heaviest) frames that together hold 90% of the objective weight.
    k90 = int(np.searchsorted(cum_w, 0.90)) + 1
    frac90 = k90 / n_train
    ax.plot(
        frac_frames,
        cum_w,
        color="C0",
        label="Boltzmann (alpha={:.2f})".format(REFERENCE_ALPHA),
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform")
    ax.axhline(0.90, color="gray", ls=":", lw=1)
    ax.axvline(frac90, color="gray", ls=":", lw=1)
    ax.annotate(
        "top {:.1f}% of frames\nhold 90% of weight".format(100 * frac90),
        xy=(frac90, 0.90),
        xytext=(0.4, 0.45),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.set_xlabel("fraction of frames (heaviest first)")
    ax.set_ylabel("cumulative share of objective weight")
    ax.set_title("Weight concentration at the reference alpha")
    ax.legend(loc="lower right")

    fig.suptitle("How much of the dataset informs the Boltzmann-weighted fit")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3D: dimer energy curves -- does the fitted potential give sensible
# orientation-dependent energetics? (external physical-validity test)
# --------------------------------------------------------------------------- #
# Canonical relative orientations of two uniaxial (oblate) particles, reused from
# notebooks/fitting_gb.ipynb. ``u1``/``u2`` are the symmetry axes (or_vec, the
# thin axis of the disk); ``r_dir`` is the unit separation direction.
DIMER_ORIENTATIONS = {
    "face-to-face (stacked)": ([0, 0, 1], [0, 0, 1], [0, 0, 1]),
    "edge-to-edge (side-by-side)": ([0, 0, 1], [0, 0, 1], [1, 0, 0]),
    "T-shaped": ([0, 0, 1], [0, 1, 0], [0, 0, 1]),
}


def _dimer_components(theta, u1, u2, r_dir, distances):
    """GB, quadrupole and net pair energy vs ``distances`` for one orientation.

    ``u1``/``u2`` are the two symmetry axes and ``r_dir`` the separation
    direction; returns ``(gb_e, qq_e, net)`` arrays the length of ``distances``.
    """
    sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q = theta[:8]
    u1 = np.tile(np.asarray(u1, dtype=float), (distances.size, 1))
    u2 = np.tile(np.asarray(u2, dtype=float), (distances.size, 1))
    r = np.outer(distances, np.asarray(r_dir, dtype=float))
    gb_e = np.asarray(gb(u1, u2, r, sigma0, eps0, kappa, kappa_prime, mu, nu, xi))
    qq_e = np.asarray(quadrupole(u1, u2, r, Q))
    return gb_e, qq_e, gb_e + qq_e


def dimer_energy_curves(theta, label="production (uniform)", distances=None, path=None):
    """GB + quadrupole pair energy vs separation for canonical dimer orientations.

    The training frames carry only 1-2 distinct orientations, so the fit's
    *angular* behaviour is largely unconstrained by the data -- this is the
    external sanity check on it. For each orientation (face-to-face / stacked,
    edge-to-edge / side-by-side, T-shaped) the net energy is split into its GB and
    quadrupole parts (reusing the production MC ``potentials.gb``/``quadrupole``),
    and a fourth panel overlays the three net curves so the orientational
    *preference* reads off directly. For benzene-like oblate quadrupoles the
    physical expectation is that T-shaped/edge contacts are favoured over the
    face-to-face sandwich (like-quadrupole stacking repulsion); a fit that
    inverts this is suspect even if its crystal-energy metrics look good.

    Returns the Figure (and saves to ``path`` when given).
    """
    if distances is None:
        distances = np.linspace(2.5, 16.0, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    flat = axes.ravel()
    nets = {}
    for ax, (name, (u1, u2, r_dir)) in zip(flat, DIMER_ORIENTATIONS.items()):
        gb_e, qq_e, net = _dimer_components(theta, u1, u2, r_dir, distances)
        nets[name] = net

        ax.plot(distances, net, color="C0", label="net")
        ax.plot(distances, gb_e, color="C1", ls="--", label="GB")
        ax.plot(distances, qq_e, color="C2", ls=":", label="quadrupole")
        ax.axhline(0.0, color="gray", lw=1)
        # Focus the y-range on the attractive well when one exists.
        lo = float(net.min())
        if lo < 0:
            ax.set_ylim(lo * 1.5, -lo)
            i = int(np.argmin(net))
            ax.plot(distances[i], lo, "ko", ms=4)
            # Add label for minimums
            xmin_frac = (distances[i] - distances[0]) / (distances[-1] - distances[0])
            tx, ha = (0.7, "right") if xmin_frac < 0.5 else (0.05, "left")
            ax.annotate(
                "min {:.3f} eV\nat {:.1f} A".format(lo, distances[i]),
                xy=(distances[i], lo),
                xytext=(tx, 0.1),
                textcoords="axes fraction",
                ha=ha,
                arrowprops=dict(arrowstyle="->", color="k"),
            )
        ax.set_xlabel("separation (Å)")
        ax.set_ylabel("energy (eV)")
        ax.set_title(name)
        ax.legend(loc="upper right")

    # Fourth panel: overlay the net curves so orientational preference is direct.
    ax = flat[3]
    for name, net in nets.items():
        ax.plot(distances, net, label=name)
    ax.axhline(0.0, color="gray", lw=1)
    lo = min(float(n.min()) for n in nets.values())
    ax.set_ylim(lo * 1.5, -lo)
    ax.set_xlabel("separation (Å)")
    ax.set_ylabel("energy (eV)")
    ax.set_title("orientation comparison")
    ax.legend(loc="upper right")

    fig.suptitle("Dimer energy curves for the {} GBQ fit".format(label))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def dimer_curves_vs_alpha(alpha_thetas, uniform_theta=None, distances=None, path=None):
    """Dimer energy curves overlaid across Boltzmann alpha (uniform = alpha 0).

    Shows how the weighting reshapes the (largely data-unconstrained)
    orientation-dependent energetics: one panel per orientation overlays the net
    pair-energy curve for each fit, coloured from uniform (alpha=0) to the
    strongest weighting. The fourth panel summarises each orientation's well
    depth (min net energy) vs alpha, so drift in the binding strength -- and in
    the orientational *ordering* -- is read off directly. As with the alpha
    quality figure, uniform is the alpha=0 limit but is drawn detached in the
    well-depth panel (no fits were run between 0 and the lowest sampled alpha).

    ``alpha_thetas`` maps ``alpha -> theta`` (weighted fits); ``uniform_theta`` is
    the unweighted fit, plotted at alpha=0.
    """
    if distances is None:
        distances = np.linspace(2.5, 16.0, 200)
    weighted = sorted(alpha_thetas)
    # All fits to overlay, in alpha order (uniform first as alpha=0).
    curves = ([(0.0, uniform_theta)] if uniform_theta is not None else []) + [
        (a, alpha_thetas[a]) for a in weighted
    ]
    cmap = plt.get_cmap("viridis")
    hi = max(a for a, _ in curves) or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    flat = axes.ravel()
    depths = {name: [] for name in DIMER_ORIENTATIONS}
    for ax, (name, (u1, u2, r_dir)) in zip(flat, DIMER_ORIENTATIONS.items()):
        lo = 0.0
        for a, theta in curves:
            _, _, net = _dimer_components(theta, u1, u2, r_dir, distances)
            depths[name].append(float(net.min()))
            lo = min(lo, float(net.min()))
            label = "uniform" if a == 0.0 else r"$\alpha$={:g}".format(a)
            ax.plot(distances, net, color=cmap(a / hi), label=label)
        ax.axhline(0.0, color="gray", lw=1)
        if lo < 0:
            ax.set_ylim(lo * 1.5, -lo)
        ax.set_xlabel("separation (Å)")
        ax.set_ylabel("energy (eV)")
        ax.set_title(name)

    # Fourth panel: well depth vs alpha per orientation. Weighted alphas joined;
    # uniform (alpha=0) drawn detached to avoid implying unsampled interpolation.
    ax = flat[3]
    for i, name in enumerate(DIMER_ORIENTATIONS):
        d = depths[name]
        color = "C{}".format(i)
        if uniform_theta is not None:
            ax.plot(0.0, d[0], "o", color=color)
            ax.plot(weighted, d[1:], marker="o", color=color, label=name)
        else:
            ax.plot(weighted, d, marker="o", color=color, label=name)
    ax.set_xlabel(r"Boltzmann weight scale $\alpha$ (1/eV)  (0 = uniform)")
    ax.set_ylabel("well depth (eV)")
    ax.set_title("binding depth vs weighting")
    ax.legend(loc="best", fontsize="small")

    # One shared legend for the alpha colours (identical across the 3 orientation
    # panels), instead of repeating it in each, placed along the figure bottom.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(curves),
        fontsize="small",
        title="weighting",
    )
    fig.suptitle("Dimer energy curves across Boltzmann weighting (alpha)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# Per-fit diagnostic PNGs written by run.py, in display order.
FIT_DIAGNOSTIC_PNGS = (
    "parity.png",
    "residuals_vs_energy.png",
    "residuals_vs_nn_distance.png",
)


def combine_fit_plots(run_dir, pngs=FIT_DIAGNOSTIC_PNGS, path=None):
    """Lay a fit's existing diagnostic PNGs out side by side in one figure.

    ``run_dir`` is a single fit directory (e.g.
    ``results/fitting/multiseed/uniform/seed_0/uniform``) holding the per-fit
    plots written by run.py. Each PNG is read back as an image and shown in its
    own panel -- no re-fitting or recomputation, just a composite for viewing
    the three diagnostics together. Missing PNGs are skipped.
    """
    found = [(p, os.path.join(run_dir, p)) for p in pngs]
    found = [(name, plt.imread(fp)) for name, fp in found if os.path.exists(fp)]
    if not found:
        return None
    # Size each panel to its image's aspect ratio so the panels are filled edge
    # to edge instead of letterboxed (which made the plots look small). Column
    # widths are proportional to each image's width:height.
    panel_h = 4.5
    aspects = [img.shape[1] / img.shape[0] for _, img in found]
    fig, axes = plt.subplots(
        1,
        len(found),
        figsize=(panel_h * sum(aspects), panel_h),
        gridspec_kw={"width_ratios": aspects},
    )
    if len(found) == 1:
        axes = [axes]
    for ax, (_, img) in zip(axes, found):
        ax.imshow(img)
        ax.set_axis_off()
    fig.suptitle("Uniform GBQ fit diagnostics", y=0.99)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0.02)
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUT, exist_ok=True)
    _data = load_dataset()
    _train_idx, _test_idx = train_test_split(_data.n_frames, TEST_FRAC, SPLIT_SEED)
    _alpha_runs = discover_alpha_runs()
    _uniform = discover_uniform_run()
    alpha_quality_plot(
        _data,
        _alpha_runs,
        _uniform,
        _test_idx,
        path=os.path.join(DEFAULT_OUT, "alpha_quality.png"),
    )
    print("Wrote", os.path.join(DEFAULT_OUT, "alpha_quality.png"))

    # One reproducibility figure per multi-seed campaign (uniform production +
    # the alpha=2.90 Boltzmann reference).
    for _campaign, _label, _fname in (
        (UNIFORM_CAMPAIGN, "uniform (production)", "seed_reproducibility_uniform.png"),
        (BOLTZMANN_CAMPAIGN, "alpha=2.90 Boltzmann", "seed_reproducibility.png"),
    ):
        _seed_runs = discover_seed_runs(campaign=_campaign)
        if not _seed_runs:
            continue
        seed_reproducibility_plot(
            _data,
            _seed_runs,
            _test_idx,
            label=_label,
            path=os.path.join(DEFAULT_OUT, _fname),
        )
        print("Wrote", os.path.join(DEFAULT_OUT, _fname))

    weight_concentration_plot(
        _data,
        _train_idx,
        sampled_alphas=tuple(_alpha_runs),
        path=os.path.join(DEFAULT_OUT, "weight_concentration.png"),
    )
    print("Wrote", os.path.join(DEFAULT_OUT, "weight_concentration.png"))

    # Production dimer curves = the uniform (unweighted) fit; the alpha=2.90
    # Boltzmann fit is kept alongside only as a weighted-sweep comparison.
    _prod = discover_uniform_production_run()
    if _prod is not None:
        dimer_energy_curves(
            _prod,
            label="production (uniform)",
            path=os.path.join(DEFAULT_OUT, "dimer_energy_curves.png"),
        )
        print("Wrote", os.path.join(DEFAULT_OUT, "dimer_energy_curves.png"))

    _boltz = discover_boltzmann_run()
    if _boltz is not None:
        dimer_energy_curves(
            _boltz,
            label="Boltzmann (alpha=2.90)",
            path=os.path.join(DEFAULT_OUT, "dimer_energy_curves_alpha2.90.png"),
        )
        print("Wrote", os.path.join(DEFAULT_OUT, "dimer_energy_curves_alpha2.90.png"))

    dimer_curves_vs_alpha(
        _alpha_runs,
        uniform_theta=_uniform,
        path=os.path.join(DEFAULT_OUT, "dimer_curves_vs_alpha.png"),
    )
    print("Wrote", os.path.join(DEFAULT_OUT, "dimer_curves_vs_alpha.png"))

    # Composite of the uniform fit's three diagnostic PNGs, side by side.
    _uniform_fit_dir = os.path.join(
        DEFAULT_RESULTS_ROOT, MULTISEED_SUBDIR, UNIFORM_CAMPAIGN, "seed_0", "uniform"
    )
    if (
        combine_fit_plots(
            _uniform_fit_dir,
            path=os.path.join(DEFAULT_OUT, "uniform_fit_diagnostics.png"),
        )
        is not None
    ):
        print("Wrote", os.path.join(DEFAULT_OUT, "uniform_fit_diagnostics.png"))
