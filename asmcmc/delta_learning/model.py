"""The Delta-learning model: ridge regression on AniSOAP descriptors.

The model fits ``Delta = E_UMA - E_GBQ`` per frame. Three of the choices here
follow from the physics rather than the statistics, and each is checked by a test.

There is no intercept and no feature centring. A frame with nothing inside the
cutoff has an all-zero descriptor (see :func:`descriptors.descriptors`), and the
model must give exactly zero Delta for it, since a correction that tends to a
constant at infinite separation is wrong. Centring the features would break this,
because a zero descriptor would no longer map to a zero feature vector, so the
features are only divided by a single scale factor. This matters in practice: in
MC the per-centre energies are summed over N = 400 particles, so a constant per
centre would add a large, spurious shift to the total energy.

Descriptors are summed over centres rather than averaged (in
:func:`descriptors.descriptors`), so a linear model on them is a sum of
per-centre energies.

The ridge penalty alpha is chosen separately at each hyperparameter point. The
number of features runs from 64 to 490 across the sweep, so a single fixed alpha
would regularise the wide representations differently from the narrow ones, and
the comparison would end up measuring the penalty instead of the representation.

Every metric is reported alongside ``null_rmse``, the RMSE of always predicting
zero. Delta is small and concentrated in the wells, so a model can get a
flattering overall RMSE by predicting almost nothing everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.delta_learning.dataset_analysis import WELL_RANGE
from asmcmc.units import EV_TO_KCAL
from asmcmc.delta_learning.descriptors import (
    Hypers,
    descriptors,
    make_ellipsoid_frame,
    quaternions_from_normals,
)

# Log-spaced ridge penalties. The descriptor scale is normalised away before the
# fit, so this range is dimensionless and does not need retuning per point.
DEFAULT_ALPHAS = np.logspace(-10, 2, 25)

# A frame counts as "well region" if any pair is inside the Cacelli well range.
# dataset_analysis.WELL_RANGE is the same cut its radial tables use, imported
# rather than restated so the two cannot drift apart.
WELL_MAX_R = WELL_RANGE[1]


def train_test_split(n, test_frac=0.2, seed=0):
    """Index split, seeded once and reused at every hyperparameter point.

    Every point has to be scored on the same held-out frames, otherwise the sweep
    compares splits as much as representations. ``fitting_gbq.run.main`` does the
    same across its weighting variants.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


@dataclass(frozen=True)
class DeltaModel:
    """A fitted correction: ``Delta_hat = (X / scale) @ coef``, in eV.

    It keeps its own :class:`~asmcmc.delta_learning.descriptors.Hypers`, because
    the coefficients only make sense with the descriptors they were fitted on.
    """

    hypers: Hypers
    coef: np.ndarray
    scale: float
    alpha: float

    def predict(self, X):
        return (np.asarray(X, dtype=float) / self.scale) @ self.coef

    def predict_frames(self, frames):
        """Delta (eV) for AniSOAP-ready ellipsoid frames."""
        return self.predict(descriptors(frames, self.hypers))


@dataclass(frozen=True)
class FitMetrics:
    """Held-out scores in kcal/mol, plus the zero-prediction reference."""

    rmse: float
    mae: float
    r2: float
    well_rmse: float
    well_n: int
    null_rmse: float
    null_well_rmse: float
    n: int

    @property
    def skill(self):
        """Fraction of the zero-prediction error removed. 0 = learned nothing."""
        return 1.0 - self.rmse / self.null_rmse if self.null_rmse else float("nan")

    @property
    def well_skill(self):
        return (
            1.0 - self.well_rmse / self.null_well_rmse
            if self.null_well_rmse
            else float("nan")
        )

    def to_dict(self):
        out = {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "well_rmse": self.well_rmse,
            "well_n": self.well_n,
            "null_rmse": self.null_rmse,
            "null_well_rmse": self.null_well_rmse,
            "n": self.n,
        }
        out["skill"] = self.skill
        out["well_skill"] = self.well_skill
        return out


def _metrics(y_true, y_pred, well, units=EV_TO_KCAL):
    truth = np.asarray(y_true, dtype=float) * units
    pred = np.asarray(y_pred, dtype=float) * units
    residual = pred - truth

    total = np.sum((truth - truth.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / total if total else float("nan")

    def rms(values):
        return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")

    return FitMetrics(
        rmse=rms(residual),
        mae=float(np.mean(np.abs(residual))),
        r2=float(r2),
        well_rmse=rms(residual[well]),
        well_n=int(np.sum(well)),
        null_rmse=rms(truth),
        null_well_rmse=rms(truth[well]),
        n=len(truth),
    )


@dataclass(frozen=True)
class FitResult:
    model: DeltaModel
    test: FitMetrics
    train: FitMetrics
    n_zero_train: int
    n_zero_test: int
    extras: dict = field(default_factory=dict)


def fit_delta(
    X,
    delta,
    train_idx,
    test_idx,
    min_pair_r=None,
    hypers=None,
    alphas=DEFAULT_ALPHAS,
    cv_folds=5,
    cv_seed=0,
):
    """Fit the correction on ``train_idx``, score it on ``test_idx``.

    ``delta`` is in eV; metrics come back in kcal/mol. ``min_pair_r`` selects
    the well-region subset -- pass ``None`` to score every frame as one group.
    """
    X = np.asarray(X, dtype=float)
    delta = np.asarray(delta, dtype=float)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    X_train = X[train_idx]
    # A single scale factor, so a zero descriptor stays zero (see the module docstring).
    scale = float(np.sqrt(np.mean(X_train**2)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    if min_pair_r is None:
        well = np.ones(len(delta), dtype=bool)
    else:
        well = np.asarray(min_pair_r, dtype=float) < WELL_MAX_R

    # RidgeCV picks alpha inside the training set only; the held-out frames
    # never influence the penalty.
    folds = KFold(n_splits=min(cv_folds, len(train_idx)), shuffle=True, random_state=cv_seed)
    search = RidgeCV(alphas=np.asarray(alphas, dtype=float), fit_intercept=False, cv=folds)
    search.fit(X_train / scale, delta[train_idx])

    ridge = Ridge(alpha=float(search.alpha_), fit_intercept=False)
    ridge.fit(X_train / scale, delta[train_idx])

    model = DeltaModel(
        hypers=hypers or Hypers(),
        coef=np.asarray(ridge.coef_, dtype=float),
        scale=scale,
        alpha=float(search.alpha_),
    )

    is_zero = np.abs(X).max(axis=1) == 0.0
    return FitResult(
        model=model,
        test=_metrics(delta[test_idx], model.predict(X[test_idx]), well[test_idx]),
        train=_metrics(delta[train_idx], model.predict(X_train), well[train_idx]),
        n_zero_train=int(is_zero[train_idx].sum()),
        n_zero_test=int(is_zero[test_idx].sum()),
        extras={"n_features": int(X.shape[1]), "scale": scale},
    )


class AniSOAPDeltaPotential:
    """A baseline potential (GBQIII by default) plus a fitted correction.

    It has the ``pair_energy`` method and ``name`` of a ``Potential``, which is
    all :func:`~asmcmc.delta_learning.dimer_benchmark.dimer_benchmark` needs to
    score it. A good fit RMSE isn't enough on its
    own; the correction also has to improve the dimer wells.

    Building a two-bead frame from a pair of disc normals means making up the
    azimuth about each normal. MC frames have the same problem, and it's harmless
    for the same reason: ``semiaxis_ab`` is used for both in-plane axes, so
    spinning a particle about its normal doesn't change its descriptor.
    """

    def __init__(self, model, baseline=CACELLI_POTENTIAL, batch_size=512):
        self.model = model
        self.baseline = baseline
        self.batch_size = int(batch_size)
        self.name = f"anisoap_delta[{model.hypers.key}]"

    def delta(self, uhat1, uhat2, r):
        """The learned correction on its own (eV), for looking into benchmark results."""
        uhat1 = np.atleast_2d(np.asarray(uhat1, dtype=float))
        uhat2 = np.atleast_2d(np.asarray(uhat2, dtype=float))
        r = np.atleast_2d(np.asarray(r, dtype=float))

        q1 = quaternions_from_normals(uhat1)
        q2 = quaternions_from_normals(uhat2)
        origin = np.zeros(3)

        out = np.empty(len(r), dtype=float)
        for start in range(0, len(r), self.batch_size):
            stop = min(start + self.batch_size, len(r))
            frames = [
                make_ellipsoid_frame(
                    np.stack([origin, r[k]]),
                    np.stack([q1[k], q2[k]]),
                    self.model.hypers,
                )
                for k in range(start, stop)
            ]
            out[start:stop] = self.model.predict_frames(frames)
        return out

    def pair_energy(self, uhat1, uhat2, r):
        """Baseline plus correction, in eV, as ``Potential.pair_energy`` returns."""
        return self.baseline.pair_energy(uhat1, uhat2, r) + self.delta(uhat1, uhat2, r)
