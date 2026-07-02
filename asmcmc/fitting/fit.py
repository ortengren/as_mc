"""Boltzmann-weighted Gay-Berne + quadrupole potential fit.

Objective (Cacelli et al. 2004, eq. 5):
    F = sum_k w_k (E_pred_k - E_target_k)^2 / sum_k w_k,
    with Boltzmann weights w_k = exp(-alpha * E_target_k).

Model (per molecule):
    E_pred = 0.5 * (sum over directed pairs of U_GBQ) / N_mol + E_intra

Parameter vector ``theta`` order:
    [sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q, E_intra]
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm

from asmcmc.fitting.data import precompute_dots_gb

PARAM_NAMES = ["sigma0", "eps0", "kappa", "kappa_prime", "mu", "nu", "xi", "Q", "E_intra"]

# Boltzmann weight scale alpha = 1/(k_B*T), in 1/eV. The production fit is the
# *uniform* (unweighted) one; this default only sets the reference Boltzmann
# point. 2.90 (weighting T ~ 4000 K) was the best of the alpha sweep
# (scripts/run_fits.sh): the gentlest weighting that stays in the stable
# Boltzmann basin -- best-scoring weighted fit, deepest (most physical)
# cohesion. Pushing alpha higher deprioritizes too much of the dataset. Alpha
# could be raised if the dataset were augmented with more dilute configurations.
# (Cacelli et al. 2004 used a ~ 0.4/(kcal/mol) ~ 9.22/eV for reference.)
DEFAULT_ALPHA = 2.90

# Search box for the 8 physical params (differential_evolution requires finite
# bounds on every parameter). E_intra is data-dependent (a ~-1601 eV/molecule
# pedestal) and is appended per-call in default_bounds.
DEFAULT_BOUNDS = {
    "sigma0": (3.0, 9.0),
    "eps0": (1e-4, 1.0),
    "kappa": (0.4, 0.99),  # oblate shape of ellipsoids
    "kappa_prime": (0.001, 12.0),
    "mu": (-11.0, 8.0),  # mu and nu both have wide literature ranges
    "nu": (-12.46, 5.0),
    "xi": (0.5, 2.0),  # GB range scale; 1.0 is the standard identity
    "Q": (-10.0, 0.0),  # only appears as Q^2, so pinned < 0
}
# Half-width (eV/molecule) of the E_intra search window around the mean target;
# the GB+Q lattice correction sits at ~+-0.5 eV on top of the pedestal.
E_INTRA_HALF_WINDOW = 5.0

# Objective value returned for a non-finite (NaN/inf) evaluation, so DE treats
# such parameter regions as very bad rather than crashing.
PENALTY = 1e30

# Per-worker objective context. With workers!=1 the ~25 MB dataset would, if
# passed through differential_evolution's `args`, be re-pickled to a worker on
# every objective call (~popsize x maxiter times).  Instead each worker process
# stores (data, weights, idx) ONCE at pool start (via _init_worker) and the map'd
# objective reads it from here, so the dataset crosses the process boundary a
# single time per worker.
_WORKER_CTX = {}


def _init_worker(data, weights, idx):
    """Pool initializer: stash the (data, weights, idx) the objective needs.

    Runs once per worker process at pool creation; ``data`` is pickled
    here a single time rather than on every evaluation.
    """
    _WORKER_CTX["args"] = (data, weights, idx)


def _objective_global(trial_theta):
    """``objective_function`` bound to this worker's stored context.

    The callable handed to ``differential_evolution(workers=pool.map)`` -- it
    takes only the trial parameter vector ``trial_theta`` so nothing but
    ``trial_theta`` is pickled per call (the dataset is loaded once per worker
    via _init_worker).
    """
    return objective_function(trial_theta, *_WORKER_CTX["args"])


def predict_per_mol(theta, data):
    """Per-molecule energy for every frame (eV/molecule).

    The directed-pair sum is halved because extract_periodic_pairs lists every
    pair in both directions (see data.py); E_intra is added once per molecule.

    GB and quadrupole are summed separately: only GB is recomputed per call,
    while the quadrupole's geometry-only part is precomputed once
    (``data.quad_geom_per_frame``) and scaled here by the sole quadrupole
    parameter ``Q**2``.  Algebraically identical to summing ``gbq`` per pair,
    but it drops the per-pair quadrupole work from the fit's inner loop.
    """
    sigma0, eps0, kappa, kappa_prime, mu, nu, xi, Q, E_intra = theta
    sum_sq, diff_sq, b_sq = data.gb_geom
    gb_pair = precompute_dots_gb(
        data.r_mag,
        data.a_i,
        data.a_j,
        data.b_ij,
        sigma0,
        eps0,
        kappa,
        kappa_prime,
        mu,
        nu,
        xi,
        sum_sq=sum_sq,
        diff_sq=diff_sq,
        b_sq=b_sq,
    )
    gb_frame = np.bincount(data.frame_index, weights=gb_pair, minlength=data.n_frames)
    return 0.5 * gb_frame / data.n_mol + Q**2 * data.quad_geom_per_frame + E_intra


def boltzmann_weights(target, alpha=DEFAULT_ALPHA):
    """w_k = exp(-alpha * E_k), with max-subtraction for numerical stability.

    The subtracted constant cancels in the normalised objective, so it only
    keeps the exponent finite on absolute DFT energies (E ~ -1600 eV). The
    most-bound (lowest-energy) frame receives the largest weight.
    """
    arg = -alpha * np.asarray(target, dtype=float)
    arg = arg - arg.max()
    return np.exp(arg)


def objective_function(theta, data, weights, idx=None):
    """Scalar merit ``0.5 * sum_k (w_k / sum w)(pred_k - target_k)^2`` (Cacelli F/2).

    This is what :func:`run_fit` minimises. ``differential_evolution`` optimises
    a scalar, so the per-frame weighted squared errors are collapsed to one
    number. ``idx`` restricts to a frame subset (e.g. the training split) and
    renormalises the weights within it; ``None`` uses all frames.

    Non-finite or undefined evaluations -- the GB directional term can go
    negative, making ``eps2**mu`` complex for fractional ``mu``, and ``mu`` near
    0 enters as ``1/mu`` (NumPy -> inf, a scalar 0.0 -> ZeroDivisionError) -- are
    mapped to a large finite ``PENALTY`` so DE avoids those parameter regions
    instead of crashing.
    """
    try:
        with np.errstate(all="ignore"):
            pred = predict_per_mol(theta, data)
            target = data.target_per_mol
            w = weights
            if idx is not None:
                pred, target, w = pred[idx], target[idx], w[idx]
            w = w / w.sum()
            val = 0.5 * float(np.sum(w * (pred - target) ** 2))
    except (ArithmeticError, ValueError):
        return PENALTY
    return val if np.isfinite(val) else PENALTY


def train_test_split(n_frames, test_frac=0.2, seed=0):
    """Random partition of ``range(n_frames)`` into ``(train_idx, test_idx)``.

    Each frame is an independent structure, so a plain frame-level shuffle is
    the structure-level split. Deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_frames)
    n_test = int(round(test_frac * n_frames))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def default_bounds(data, e_intra_half_window=E_INTRA_HALF_WINDOW, idx=None):
    """``(lo, hi)`` per parameter for ``differential_evolution``.

    The 8 physical params use ``DEFAULT_BOUNDS``; ``E_intra`` is centred on the
    mean target energy (the ~-1601 eV/molecule intramolecular pedestal), widened
    by ``e_intra_half_window``. ``idx`` restricts the mean to a frame subset
    (e.g. the training split) so the box never peeks at held-out energies.
    """
    target = data.target_per_mol if idx is None else data.target_per_mol[idx]
    mean = float(np.mean(target))
    bounds = [DEFAULT_BOUNDS[name] for name in PARAM_NAMES[:-1]]
    bounds.append((mean - e_intra_half_window, mean + e_intra_half_window))
    return bounds


def run_fit(
    data,
    weights=None,
    alpha=DEFAULT_ALPHA,
    bounds=None,
    idx=None,
    seed=0,
    workers=1,
    polish=True,
    progress=False,
    **de_kwargs,
):
    """Fit ``theta`` by global search with ``scipy.optimize.differential_evolution``.

    DE is a single-call global optimiser. It minimises :func:`objective_function`
    (the Cacelli merit F/2) over the finite ``bounds``.

    ``idx`` fits on a frame subset (the training split); ``weights`` defaults to
    :func:`boltzmann_weights` of the full target vector (``objective_function``
    renormalises within ``idx``). ``workers=-1`` parallelises across all cores
    (``>1`` for a fixed count) via a spawn ``ProcessPoolExecutor`` whose
    initializer ships the dataset to each worker once (see ``_init_worker``);
    ``workers=1`` stays fully serial. In practice the per-eval gbq is
    memory-bandwidth-bound, so parallel speedup plateaus around ~3x rather than
    scaling with core count. ``polish`` runs a final local refinement (in the
    main process). ``progress`` shows a tqdm bar advancing one step per DE
    generation (its ETA assumes the full ``maxiter``; the run may stop earlier on
    ``tol`` convergence, and a final ``polish`` step runs after the bar closes).
    Extra ``de_kwargs`` (``maxiter``, ``popsize``, ``tol``, ...) are forwarded.
    Returns the ``OptimizeResult`` (``.x`` = fitted theta).
    """
    if weights is None:
        weights = boltzmann_weights(data.target_per_mol, alpha)
    if bounds is None:
        bounds = default_bounds(data, idx=idx)

    user_cb = de_kwargs.pop("callback", None)
    callback = user_cb
    bar = None
    if progress:
        bar = tqdm(total=de_kwargs.get("maxiter", 1000), desc="DE fit", unit="gen")

        def callback(intermediate_result):
            bar.update(1)
            bar.set_postfix(best="{:.4g}".format(float(intermediate_result.fun)))
            if user_cb is not None:
                user_cb(intermediate_result)

    try:
        if workers == 1:
            # Serial: no pool; objective_function reads (data, weights, idx)
            # straight from `args` -- the dataset never crosses a process.
            return differential_evolution(
                objective_function,
                bounds,
                args=(data, weights, idx),
                rng=seed,
                workers=1,
                polish=polish,
                callback=callback,
                **de_kwargs,
            )

        # Parallel: ship (data, weights, idx) to each worker ONCE via the pool
        # initializer and map the context-bound _objective_global (takes only
        # trial_theta), so the ~25 MB dataset is pickled once per worker, not
        # once per eval.  The main process also needs the context populated because
        # polish runs its final L-BFGS-B step here, not through the pool. spawn
        # matches the codebase convention (nvt_scan) and avoids forking a process
        # safely.  Switching to fork could provide a speedup.
        _init_worker(data, weights, idx)
        max_workers = None if workers < 0 else workers
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(data, weights, idx),
        ) as pool:
            return differential_evolution(
                _objective_global,
                bounds,
                rng=seed,
                workers=pool.map,  # type: ignore[arg-type]  # scipy: map-like callable, untyped as int
                polish=polish,
                callback=callback,
                **de_kwargs,
            )
    finally:
        if bar is not None:
            bar.close()
