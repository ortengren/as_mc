"""The Delta-learning model: ridge on AniSOAP descriptors, and its physics gate.

Fits ``Delta = E_UMA - E_GBQ`` per cluster. Three choices here are physics, not
statistics, and each is pinned by a test:

**No intercept, and no feature centring.** A frame with nothing inside the
cutoff has an all-zero descriptor (see :func:`data.descriptors`), and the model
must return exactly zero Delta there -- a truncated correction that returns a
constant at infinite separation is wrong. Centring the features would destroy
that (a zero descriptor would map to a nonzero feature vector), so scaling is a
single scalar divide. This is not cosmetic: in deployment the per-centre
energies are summed over N=400 particles, so a constant per centre becomes a
large spurious extensive shift in the total energy.

**Summed, not averaged, descriptors.** Set in :func:`data.descriptors`; a linear
model on a sum of per-centre rows *is* a sum of per-centre energies.

**Alpha chosen per hyperparameter point.** Feature count runs 64 to 490 across
the sweep, so a shared fixed alpha would regularise the wide representations
differently from the narrow ones and the comparison would measure the penalty,
not the representation.

Every metric is reported against ``null_rmse`` -- the RMSE of predicting zero --
because Delta is small and heavily concentrated in the wells, so a model can
post a flattering global RMSE by predicting nearly nothing everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from asmcmc.base.potentials import CACELLI_POTENTIAL
from asmcmc.data_preparation.cluster_analysis import WELL_RANGE
from asmcmc.fitting_anisoap.data import (
    EV_TO_KCAL,
    Hypers,
    descriptors,
    make_ellipsoid_frame,
    quaternions_from_normals,
)

# Log-spaced ridge penalties. The descriptor scale is normalised away before the
# fit, so this range is dimensionless and does not need retuning per point.
DEFAULT_ALPHAS = np.logspace(-10, 2, 25)

# A frame counts as "well region" if any pair is inside the Cacelli well range.
# cluster_analysis.WELL_RANGE is the same cut its radial tables use, imported
# rather than restated so the two cannot drift apart.
WELL_MAX_R = WELL_RANGE[1]


def train_test_split(n, test_frac=0.2, seed=0):
    """Index split, seeded once and reused at every hyperparameter point.

    Every point must be scored on identical held-out frames or the sweep is
    comparing splits as much as representations -- the same discipline
    ``fitting_gbq.run.main`` applies across its weighting variants.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


@dataclass(frozen=True)
class DeltaModel:
    """A fitted correction: ``Delta_hat = (X / scale) @ coef``, in eV.

    Carries its own :class:`~asmcmc.fitting_anisoap.data.Hypers` because a
    descriptor matrix is meaningless without the hypers that produced it, so the
    two must travel together into deployment and into the dimer gate.
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
    # One scalar, so a zero descriptor stays zero -- see the module docstring.
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
    """A fitted correction wearing the ``Potential`` pair interface.

    Exists so a candidate can be scored by the **existing**
    ``asmcmc.utils.validation.dimer_benchmark`` without that module learning
    anything about AniSOAP: it only ever calls ``pair_energy`` and reads
    ``name``. Per the GB+Q refit postmortem, fit parity is never sufficient --
    a correction that is repulsive at the cofacial stack is rejected however good
    its RMSE.

    Building a two-bead frame from a pair of disc *normals* means inventing the
    azimuth about each normal, which is exactly the deployment situation and is
    sound for the same reason: with ``semiaxis_ab`` used for both in-plane axes,
    spin about the normal moves no descriptor component.
    """

    def __init__(self, model, baseline=CACELLI_POTENTIAL, batch_size=512):
        self.model = model
        self.baseline = baseline
        self.batch_size = int(batch_size)
        self.name = f"anisoap_delta[{model.hypers.key}]"

    def delta(self, uhat1, uhat2, r):
        """The learned correction alone (eV), for diagnosing the gate."""
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
        """Baseline + correction, in eV -- the ``Potential`` contract."""
        return self.baseline.pair_energy(uhat1, uhat2, r) + self.delta(uhat1, uhat2, r)
