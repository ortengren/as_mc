"""Physics validation benchmarks for candidate potentials.

A potential can reproduce condensed-phase per-configuration energies almost
perfectly and still get the *pair interaction* badly wrong (a per-molecule
energy is a sum over many pairs, so wrong pair energies cancel in the fit
target). MC then samples exactly the geometries the fit never constrained.
The GB+Q refit to the PBE-D3 crystal dataset is the cautionary example: test
RMSE ~3 kcal/mol on the crystals, yet *repulsive* at the 3.9 A cofacial
stacking distance and anti-correlated with the true dimer wells.

This module scores an energy model against near-ground-truth pair data that
is independent of any condensed-phase fit: the ab initio (MP2/6-31G*,
supermolecule) benzene dimer interaction energies of Cacelli et al.,
J. Chem. Phys. 120, 3648 (2004) — the supplement data the GBQIII potential
was originally fit to (``data/new_data/3648_1_supplements``).

Every candidate potential (GB+Q variants, AniSOAP models) should pass
:func:`dimer_benchmark` *in addition to* condensed-phase parity; thresholds
live in ``tests/test_validation.py``.

Geometry convention (from the supplement README): molecule A is fixed at the
origin with its ring in the xz-plane, so its disc normal is +y. Each row
gives molecule B's centre of mass (X, Y, Z) and Euler angles (alpha, beta,
gamma) in degrees; B's normal is the Euler rotation applied to +y. The Euler
sequence is not stated in the README — ``zyx`` (intrinsic) was identified by
scanning all standard conventions against the 53 angle-carrying rows
(r = 0.86 vs 0.37 for the worst) and confirmed physically: the
(beta=90, gamma=90) family maps +y -> +z, the T-shaped geometry, and lands
on the known T-shaped well (~ -2.3 kcal/mol near 5.0 A).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

EV_TO_KCAL = 23.060541945329334
EULER_SEQ = "zyx"

_REPO_ROOT = Path(__file__).resolve().parents[1]
CACELLI_DIMER_PATH = (
    _REPO_ROOT / "data/new_data/3648_1_supplements/abinitio.energies.txt"
)

# A's disc normal: ring in the xz-plane.
_NORMAL_A = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class DimerData:
    """The Cacelli ab initio dimer set as ready-to-evaluate geometries.

    ``uhat1``/``uhat2``/``r`` are shaped ``(n, 3)`` and feed straight into
    ``Potential.pair_energy``; ``energy_kcal`` is the MP2 interaction energy.
    ``euler_deg`` keeps the raw (alpha, beta, gamma) so geometry families can
    be selected downstream.
    """

    uhat1: np.ndarray
    uhat2: np.ndarray
    r: np.ndarray
    energy_kcal: np.ndarray
    euler_deg: np.ndarray

    def __len__(self):
        return len(self.energy_kcal)


def load_cacelli_dimers(path=None):
    """Parse the supplement's ``abinitio.energies.txt`` into a :class:`DimerData`.

    Skips comment/blank lines; each data row is
    ``X Y Z alpha beta gamma E(kcal/mol)`` per the README convention above.
    """
    path = CACELLI_DIMER_PATH if path is None else Path(path)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append([float(x) for x in parts[:7]])
    data = np.asarray(rows)
    r = data[:, 0:3]
    euler_deg = data[:, 3:6]
    energy_kcal = data[:, 6]
    uhat1 = np.tile(_NORMAL_A, (len(data), 1))
    uhat2 = Rotation.from_euler(EULER_SEQ, euler_deg, degrees=True).apply(_NORMAL_A)
    return DimerData(uhat1, uhat2, r, energy_kcal, euler_deg)


def dimer_scan(potential, uhat1, uhat2, rhat, dists):
    """Model pair energy (kcal/mol) along one dimer ray.

    Fixed unit normals ``uhat1``/``uhat2`` and separation direction ``rhat``,
    centre-centre distance swept over ``dists`` (A).
    """
    dists = np.asarray(dists, dtype=float)
    n = len(dists)
    u1 = np.tile(np.asarray(uhat1, dtype=float), (n, 1))
    u2 = np.tile(np.asarray(uhat2, dtype=float), (n, 1))
    r = dists[:, None] * np.asarray(rhat, dtype=float)[None, :]
    return potential.pair_energy(u1, u2, r) * EV_TO_KCAL


# Canonical dimer families, matching how the ab initio set samples them.
# Each entry: (uhat2, rhat, data-row mask builder, dense scan grid).
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _angle_zero(euler_deg):
    return np.all(euler_deg == 0.0, axis=1)


def _family_masks(data):
    """Boolean masks selecting the three canonical families in the data."""
    ang0 = _angle_zero(data.euler_deg)
    x, y, z = data.r.T
    return {
        # stacked straight along the common normal
        "cofacial": ang0 & (x == 0.0) & (z == 0.0),
        # stacking height fixed at the ab initio cofacial minimum, lateral slip
        "parallel_displaced": ang0 & (x == 0.0) & (z != 0.0),
        # B's normal rotated onto +z, displaced along +z
        "t_shaped": (
            (data.euler_deg[:, 0] == 0.0)
            & (data.euler_deg[:, 1] == 90.0)
            & (data.euler_deg[:, 2] == 90.0)
            & (x == 0.0)
            & (y == 0.0)
        ),
    }


@dataclass(frozen=True)
class FamilyWell:
    """One geometry family: ab initio minimum vs the model's.

    ``model_at_ab_min`` is the model energy at the ab initio minimum-energy
    row — the single most diagnostic number (a broken model is repulsive
    there). ``model_depth``/``model_r`` come from a dense scan along the
    family's ray, so a shifted model minimum is still found.
    """

    ab_depth: float
    ab_r: float
    model_at_ab_min: float
    model_depth: float
    model_r: float


@dataclass(frozen=True)
class DimerBenchmark:
    """Scores of one potential against the ab initio dimer set (kcal/mol)."""

    name: str
    full_pearson_r: float
    full_rmse_kcal: float
    well_pearson_r: float
    well_rmse_kcal: float
    stacking_energy_kcal: float
    wells: dict = field(default_factory=dict)

    @property
    def stacking_bound(self):
        """True if the model binds the cofacial stack at the ab initio
        equilibrium separation — the check the condensed-phase refit fails."""
        return self.stacking_energy_kcal < 0.0

    def summary(self):
        lines = [
            f"Dimer benchmark — {self.name}",
            f"  all rows      : r = {self.full_pearson_r:6.3f}   "
            f"RMSE = {self.full_rmse_kcal:6.3f} kcal/mol",
            f"  wells (E < 0) : r = {self.well_pearson_r:6.3f}   "
            f"RMSE = {self.well_rmse_kcal:6.3f} kcal/mol",
            f"  cofacial stack at ab initio minimum: "
            f"{self.stacking_energy_kcal:+.3f} kcal/mol "
            f"({'bound' if self.stacking_bound else 'REPULSIVE'})",
        ]
        for fam, w in self.wells.items():
            lines.append(
                f"  {fam:18s}: ab {w.ab_depth:6.2f} @ {w.ab_r:.2f} A | "
                f"model {w.model_depth:6.2f} @ {w.model_r:.2f} A "
                f"(at ab min: {w.model_at_ab_min:6.2f})"
            )
        return "\n".join(lines)


def dimer_benchmark(potential, data=None):
    """Score ``potential`` (anything with ``pair_energy``) against the
    ab initio dimers; returns a :class:`DimerBenchmark`.

    Global metrics are computed over all rows and over the attractive subset
    (E < 0, where MC spends its time and phase behaviour is decided); the
    repulsive wall's huge dynamic range otherwise dominates — the same metric
    trap as condensed-phase parity. Family wells compare depth and location
    along each canonical ray.
    """
    if data is None:
        data = load_cacelli_dimers()
    model = potential.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL
    ab = data.energy_kcal

    def _scores(mask):
        m, a = model[mask], ab[mask]
        return (
            float(np.corrcoef(m, a)[0, 1]),
            float(np.sqrt(np.mean((m - a) ** 2))),
        )

    full_r, full_rmse = _scores(np.ones(len(data), dtype=bool))
    well_r, well_rmse = _scores(ab < 0.0)

    masks = _family_masks(data)
    dense = {
        "cofacial": (_Y, _Y, np.linspace(3.0, 9.0, 601)),
        "t_shaped": (_Z, _Z, np.linspace(4.0, 10.0, 601)),
    }

    wells = {}
    for fam, mask in masks.items():
        i_min = np.where(mask)[0][np.argmin(ab[mask])]
        ab_depth = float(ab[i_min])
        ab_r = float(np.linalg.norm(data.r[i_min]))
        model_at_ab_min = float(model[i_min])
        if fam in dense:
            u2, rhat, dists = dense[fam]
            curve = dimer_scan(potential, _NORMAL_A, u2, rhat, dists)
            k = int(np.argmin(curve))
            model_depth, model_r = float(curve[k]), float(dists[k])
        else:
            # parallel-displaced: slip laterally at the ab initio stacking
            # height rather than along a single ray through the origin
            y0 = float(data.r[i_min][1])
            slips = np.linspace(0.0, 3.0, 301)
            r = np.column_stack([np.zeros_like(slips), np.full_like(slips, y0), slips])
            n = len(slips)
            curve = (
                potential.pair_energy(
                    np.tile(_NORMAL_A, (n, 1)), np.tile(_NORMAL_A, (n, 1)), r
                )
                * EV_TO_KCAL
            )
            k = int(np.argmin(curve))
            model_depth = float(curve[k])
            model_r = float(np.linalg.norm(r[k]))
        wells[fam] = FamilyWell(ab_depth, ab_r, model_at_ab_min, model_depth, model_r)

    # cofacial stacking check at the ab initio minimum-energy separation
    cof = wells["cofacial"]
    return DimerBenchmark(
        name=getattr(potential, "name", type(potential).__name__),
        full_pearson_r=full_r,
        full_rmse_kcal=full_rmse,
        well_pearson_r=well_r,
        well_rmse_kcal=well_rmse,
        stacking_energy_kcal=cof.model_at_ab_min,
        wells=wells,
    )
