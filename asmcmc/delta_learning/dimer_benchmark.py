"""The dimer benchmark: score a potential on the benzene dimers of Cacelli et al. (2004).

A potential can fit condensed-phase energies well and still get the pair
interaction wrong, because a molecule's energy is a sum over many pairs and their
errors can cancel. So before a candidate potential is used in MC, it's scored here
on the 197 dimer geometries from Cacelli et al., J. Chem. Phys. 120, 3648 (2004)
(``data/cacelli_2004_dimers``).

The reference energies are UMA's (:data:`DEFAULT_REFERENCE`). The supplement's own
MP2 energies can be loaded with ``reference="mp2"``, but only as a diagnostic:
GBQIII was fitted to them, so they favour leaving GBQIII unchanged. The verdict is
:attr:`DimerBenchmark.improves_on_baseline`. The geometries are closely spaced
scans along lines through the three wells. They show the shape of the wells, but
they aren't a held-out test set.

The geometry convention comes from the supplement's README. Molecule A sits at
the origin with its ring in the xz-plane and two C-H bonds along the z-axis, so
its disc normal is +y. Each row gives molecule B's centre of mass (X, Y, Z) and
its Euler angles (alpha, beta, gamma) in degrees. The README doesn't say which
Euler convention it uses. :data:`EULER_SEQ` is the proper z-y-z convention
(scipy's intrinsic ``"ZYZ"``), the only standard reading that is consistent with
every row that has angles. ``docs/findings.md`` §3 gives the evidence.

The atomistic helpers at the end of the module rebuild the same rows as 24-atom
dimers, so an ASE calculator such as UMA can also be scored with
:func:`score_energies`. ``scripts/uma_cacelli_dimers.py`` uses them to regenerate
the UMA reference.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.build import molecule
from scipy.spatial.transform import Rotation

from asmcmc.paths import data_path
from asmcmc.delta_learning.uma import frame_energy
from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.units import EV_TO_KCAL

EULER_SEQ = "ZYZ"

CACELLI_DIMER_PATH = data_path("cacelli_2004_dimers", "abinitio.energies.txt")
UMA_DIMER_PATH = data_path("uma_dimers", "dimer_energies.csv")

REFERENCES = ("uma", "mp2")

#: The reference a benchmark scores against unless told otherwise.
DEFAULT_REFERENCE = "uma"

# A's disc normal: ring in the xz-plane.
_NORMAL_A = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class DimerData:
    """The Cacelli ab initio dimer set as ready-to-evaluate geometries.

    ``uhat1``/``uhat2``/``r`` are shaped ``(n, 3)`` and feed straight into
    ``Potential.pair_energy``; ``energy_kcal`` is the reference interaction
    energy (``reference`` says which). ``euler_deg`` keeps the raw
    (alpha, beta, gamma) so geometry families can be selected downstream.
    """

    uhat1: np.ndarray
    uhat2: np.ndarray
    r: np.ndarray
    energy_kcal: np.ndarray
    euler_deg: np.ndarray
    reference: str = "mp2"

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


def load_uma_dimers(path=None):
    """The Cacelli dimer geometries with UMA interaction energies.

    The geometry always comes from :func:`load_cacelli_dimers`, not from the CSV,
    so both references describe the same 197 dimers. The CSV's own coordinates
    are checked against it, and a mismatch raises an error rather than silently
    scoring a different set. Regenerate the CSV with
    ``python scripts/uma_cacelli_dimers.py``.
    """
    path = UMA_DIMER_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"UMA dimer reference not found at {path}. Regenerate it with:\n"
            "  python scripts/uma_cacelli_dimers.py"
        )

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    geometry = load_cacelli_dimers()
    if len(rows) != len(geometry):
        raise ValueError(
            f"{path} has {len(rows)} rows, the ab initio set has {len(geometry)}"
        )

    csv_r = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
    csv_euler = np.array(
        [[float(row[k]) for k in ("alpha", "beta", "gamma")] for row in rows]
    )
    # The CSV stores offsets to 4 dp and angles to 2 dp, so compare at that scale.
    if not np.allclose(csv_r, geometry.r, atol=1e-3):
        raise ValueError(f"{path} centre offsets disagree with the ab initio geometries")
    if not np.allclose(csv_euler, geometry.euler_deg, atol=1e-2):
        raise ValueError(f"{path} Euler angles disagree with the ab initio geometries")

    return DimerData(
        geometry.uhat1,
        geometry.uhat2,
        geometry.r,
        np.array([float(row["e_uma_kcal"]) for row in rows]),
        geometry.euler_deg,
        reference="uma",
    )


def load_reference_dimers(reference=DEFAULT_REFERENCE, path=None):
    """The Cacelli dimers with ``"uma"`` (default) or ``"mp2"`` reference energies."""
    if reference == "uma":
        return load_uma_dimers(path)
    if reference == "mp2":
        return load_cacelli_dimers(path)
    raise ValueError(f"unknown reference {reference!r}; expected one of {REFERENCES}")


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


# Axes of the canonical dimer families' scans.
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
    """One geometry family: the reference minimum and the model's.

    ``model_at_ab_min`` is the model's energy at the row where the reference
    energy is lowest. It's the most telling single number, since a broken model
    is repulsive there. ``model_depth`` and ``model_r`` come from a fine scan
    through the family's well, so the model's minimum is found even if it has
    shifted.
    """

    ab_depth: float
    ab_r: float
    model_at_ab_min: float
    model_depth: float
    model_r: float


@dataclass(frozen=True)
class DimerBenchmark:
    """Scores of one potential against the reference dimer set (kcal/mol)."""

    name: str
    full_pearson_r: float
    full_rmse_kcal: float
    well_pearson_r: float
    well_rmse_kcal: float
    stacking_energy_kcal: float
    wells: dict = field(default_factory=dict)
    reference: str = "mp2"
    baseline_name: str = ""
    baseline_full_rmse_kcal: float = float("nan")
    baseline_well_rmse_kcal: float = float("nan")

    @property
    def stacking_bound(self):
        """True if the model binds the cofacial stack at the reference
        equilibrium separation. The condensed-phase GB+Q refit fails this."""
        return self.stacking_energy_kcal < 0.0

    @property
    def well_rmse_gain_kcal(self):
        """Baseline well RMSE minus this model's: positive means an improvement."""
        return self.baseline_well_rmse_kcal - self.well_rmse_kcal

    @property
    def improves_on_baseline(self):
        """Whether the correction beats the uncorrected baseline on the wells.

        Rank candidates on this, not on :attr:`stacking_bound`: a model can bind
        the stack and still be worse than no correction at all.
        """
        return bool(self.well_rmse_gain_kcal > 0.0)

    def summary(self):
        lines = [
            f"Dimer benchmark — {self.name}   [reference: {self.reference}]",
            f"  all rows      : r = {self.full_pearson_r:6.3f}   "
            f"RMSE = {self.full_rmse_kcal:6.3f} kcal/mol",
            f"  wells (E < 0) : r = {self.well_pearson_r:6.3f}   "
            f"RMSE = {self.well_rmse_kcal:6.3f} kcal/mol",
        ]
        if self.baseline_name:
            lines.append(
                f"  vs baseline {self.baseline_name}: "
                f"well RMSE {self.baseline_well_rmse_kcal:6.3f} -> "
                f"{self.well_rmse_kcal:6.3f} "
                f"({self.well_rmse_gain_kcal:+.3f}, "
                f"{'IMPROVES' if self.improves_on_baseline else 'no gain'})"
            )
        lines += [
            f"  cofacial stack at reference minimum: "
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


def family_labels(data):
    """Per-row family name: ``cofacial``, ``parallel_displaced``, ``t_shaped`` or ``other``."""
    labels = np.full(len(data), "other", dtype=object)
    for fam, mask in _family_masks(data).items():
        labels[mask] = fam
    return labels


def family_scan_geometry(family, data, i_min, n_points=601):
    """The dense scan a family's well is searched along: ``(uhat2, offsets, distances)``.

    ``cofacial`` and ``t_shaped`` sweep the centre-centre distance along their
    axis. ``parallel_displaced`` slips laterally at the height of the reference
    minimum (row ``i_min``), since its well is not on a ray through the origin.
    """
    if family == "cofacial":
        dists = np.linspace(3.0, 9.0, n_points)
        return _Y, dists[:, None] * _Y[None, :], dists
    if family == "t_shaped":
        dists = np.linspace(4.0, 10.0, n_points)
        return _Z, dists[:, None] * _Z[None, :], dists
    if family == "parallel_displaced":
        y0 = float(data.r[i_min][1])
        slips = np.linspace(0.0, 3.0, max(2, n_points // 2 + 1))
        offsets = np.column_stack([np.zeros_like(slips), np.full_like(slips, y0), slips])
        return _Y, offsets, np.linalg.norm(offsets, axis=1)
    raise ValueError(f"unknown family: {family}")


def cg_scan(potential, n_points=601):
    """A ``scan_fn`` for :func:`score_energies` that evaluates a pair potential."""

    def scan(family, data, i_min):
        u2, offsets, dists = family_scan_geometry(family, data, i_min, n_points)
        n = len(offsets)
        curve = (
            potential.pair_energy(np.tile(_NORMAL_A, (n, 1)), np.tile(u2, (n, 1)), offsets)
            * EV_TO_KCAL
        )
        return curve, dists

    return scan


def score_energies(model_kcal, data, name, scan_fn=None, baseline_kcal=None, baseline_name=""):
    """Score per-row model energies (kcal/mol) against the reference dimers.

    The overall metrics are computed on all rows and separately on the
    attractive rows (E < 0), where MC spends its time; otherwise the large
    energies on the repulsive wall would dominate. ``scan_fn(family, data, i_min)
    -> (energies_kcal, distances)`` finds each family's model minimum on a fine
    scan, so a shifted well is still measured. Without it, the well is taken from
    the family's own rows. ``baseline_kcal`` holds the uncorrected potential's
    energies on the same rows, which a candidate has to beat.
    """
    model = np.asarray(model_kcal, dtype=float)
    ab = data.energy_kcal

    def _scores(mask):
        m, a = model[mask], ab[mask]
        return (
            float(np.corrcoef(m, a)[0, 1]),
            float(np.sqrt(np.mean((m - a) ** 2))),
        )

    full_r, full_rmse = _scores(np.ones(len(data), dtype=bool))
    well_r, well_rmse = _scores(ab < 0.0)

    wells = {}
    for fam, mask in _family_masks(data).items():
        i_min = np.where(mask)[0][np.argmin(ab[mask])]
        ab_depth = float(ab[i_min])
        ab_r = float(np.linalg.norm(data.r[i_min]))
        model_at_ab_min = float(model[i_min])
        if scan_fn is None:
            rows = np.where(mask)[0]
            k = rows[int(np.argmin(model[rows]))]
            model_depth, model_r = float(model[k]), float(np.linalg.norm(data.r[k]))
        else:
            curve, dists = scan_fn(fam, data, i_min)
            k = int(np.argmin(curve))
            model_depth, model_r = float(curve[k]), float(dists[k])
        wells[fam] = FamilyWell(ab_depth, ab_r, model_at_ab_min, model_depth, model_r)

    baseline_full, baseline_well = float("nan"), float("nan")
    if baseline_kcal is not None:
        base = np.asarray(baseline_kcal, dtype=float)
        attractive = ab < 0.0
        baseline_full = float(np.sqrt(np.mean((base - ab) ** 2)))
        baseline_well = float(np.sqrt(np.mean((base[attractive] - ab[attractive]) ** 2)))

    return DimerBenchmark(
        name=name,
        full_pearson_r=full_r,
        full_rmse_kcal=full_rmse,
        well_pearson_r=well_r,
        well_rmse_kcal=well_rmse,
        # the cofacial stack evaluated at the reference minimum-energy separation
        stacking_energy_kcal=wells["cofacial"].model_at_ab_min,
        wells=wells,
        reference=data.reference,
        baseline_name=baseline_name,
        baseline_full_rmse_kcal=baseline_full,
        baseline_well_rmse_kcal=baseline_well,
    )


def dimer_benchmark(potential, data=None, baseline=CACELLI_POTENTIAL):
    """Score ``potential`` (anything with ``pair_energy``) on the reference dimers.

    ``baseline`` is the uncorrected potential a candidate has to beat; pass
    ``None`` to skip it. Returns a :class:`DimerBenchmark`.
    """
    if data is None:
        data = load_reference_dimers()
    model = potential.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL
    baseline_kcal, baseline_name = None, ""
    if baseline is not None:
        baseline_kcal = baseline.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL
        baseline_name = getattr(baseline, "name", type(baseline).__name__)
    return score_energies(
        model,
        data,
        name=getattr(potential, "name", type(potential).__name__),
        scan_fn=cg_scan(potential),
        baseline_kcal=baseline_kcal,
        baseline_name=baseline_name,
    )


# --- atomistic reconstruction, for scoring an ASE calculator such as UMA -------


def benzene_monomer():
    """Molecule A as the supplement defines it: ring in the xz-plane, two C-H bonds on z.

    ASE's g2 benzene (ideal D6h: C-C 1.395 A, C-H 1.087 A) lies in the z = 0
    plane with C-H bonds along +/-y; a -90 degree turn about x puts it in the
    supplement's frame. Cacelli et al.'s MP2 monomer was not published.
    """
    mol = molecule("C6H6")
    mol.positions = Rotation.from_euler("x", -90, degrees=True).apply(mol.positions)
    mol.set_pbc(False)
    mol.translate(-mol.get_center_of_mass())
    mol.info.update({"charge": 0, "spin": 1})
    return mol


def cacelli_dimer_frames(data=None, monomer=None, euler_seq=EULER_SEQ):
    """The dimer rows as rigid 24-atom ``Atoms`` (A at the origin, B placed by the row).

    B is rotated by the same ``Rotation.from_euler`` that gives ``data.uhat2``, so
    the atomistic and coarse-grained geometries agree by construction. Each frame
    carries ``arrays["molecule_id"]`` (0 for A, 1 for B), the ``charge``/``spin``
    UMA needs, and the row's provenance in ``info``.
    """
    data = load_cacelli_dimers() if data is None else data
    mono = benzene_monomer() if monomer is None else monomer
    ref = mono.get_positions()
    rotations = Rotation.from_euler(euler_seq, data.euler_deg, degrees=True)

    frames = []
    for k, rot in enumerate(rotations):
        dimer = mono.copy()
        b = mono.copy()
        b.set_positions(rot.apply(ref) + data.r[k])
        dimer += b
        dimer.set_pbc(False)
        dimer.set_cell(np.zeros((3, 3)))
        dimer.arrays["molecule_id"] = np.repeat([0, 1], len(mono)).astype(np.int32)
        dimer.info.update(
            {
                "charge": 0,
                "spin": 1,
                "n_molecules": 2,
                "row_index": k,
                "reference_kcal": float(data.energy_kcal[k]),
                "euler_deg": np.asarray(data.euler_deg[k], dtype=float),
                "euler_seq": euler_seq,
                "com": np.asarray(data.r[k], dtype=float),
                "com_sep": float(np.linalg.norm(data.r[k])),
            }
        )
        frames.append(dimer)
    return frames


def atomistic_pair_energies(frames, calculator, monomer=None):
    """Interaction energies (kcal/mol), ``E_dimer - 2 E_monomer``, under an ASE calculator.

    The monomers are rigid copies of one geometry, so the monomer energy is
    evaluated once.
    """
    mono = benzene_monomer() if monomer is None else monomer
    e_mono = frame_energy(mono, calculator)
    dimer = np.array([frame_energy(f, calculator) for f in frames])
    return (dimer - 2.0 * e_mono) * EV_TO_KCAL


def atomistic_scan(calculator, monomer=None, euler_seq=EULER_SEQ, n_points=121):
    """A ``scan_fn`` for :func:`score_energies` that rebuilds and evaluates dimers.

    B keeps the orientation of the family's reference-minimum row. ``n_points`` is
    far below :func:`cg_scan`'s default because an MLIP call costs ~0.2 s.
    """
    mono = benzene_monomer() if monomer is None else monomer
    ref = mono.get_positions()
    e_mono = frame_energy(mono, calculator)

    def scan(family, data, i_min):
        _, offsets, dists = family_scan_geometry(family, data, i_min, n_points)
        rotated = Rotation.from_euler(euler_seq, data.euler_deg[i_min], degrees=True).apply(ref)
        curve = []
        for offset in offsets:
            dimer = mono.copy()
            b = mono.copy()
            b.set_positions(rotated + offset)
            dimer += b
            curve.append(frame_energy(dimer, calculator) - 2.0 * e_mono)
        return np.array(curve) * EV_TO_KCAL, dists

    return scan
