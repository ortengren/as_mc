from abc import ABC, abstractmethod

import numpy as np
import ase
import ase.io
from scipy.spatial.transform import Rotation
from asmcmc.base.potentials import GB_PARAMS
from asmcmc.base.trial_moves import calc_or_vec
from asmcmc.base.paths import data_path

# Package-default shape (from DEFAULT_POTENTIAL). Used only when no potential is
# supplied; the geometry a config is built at must match the potential the MC
# actually runs (different sigma0/kappa => different contact distances and a
# different meaning for the reduced density rho* = N*sigma0^3/V), so callers
# pass the simulated potential and the builders read sigma0/kappa from it.
SIGMA0 = GB_PARAMS["sigma0"]
KAPPA = GB_PARAMS["kappa"]

DEFAULT_N_PARTICLES = 210
DEFAULT_DENSITY = 0.6
# Columnar starts can reach the ordered-phase density (rho* ~ 1.3-2.0) that the
# random simple-cubic start cannot. Default sits safely below the 1/kappa packing
# ceiling so per-particle jitter never forces an overlap.
DEFAULT_COLUMNAR_DENSITY = 1.4

# Herringbone start: the experimental benzene Pbca crystal, coarse-grained to one
# oblate particle per molecule (built by scripts/process_cod_benzene_data.py). The
# motif path resolves against the package so it is found regardless of cwd.
DEFAULT_HERRINGBONE_MOTIF = data_path("benzene_herringbone_cg.xyz")
# Small default jitters: a thermal wiggle giving replica independence that leaves
# the herringbone order intact. Set both to 0 for a pristine crystal start.
DEFAULT_HERRINGBONE_POS_JITTER = 0.1  # Angstrom
DEFAULT_HERRINGBONE_OR_JITTER = 0.1  # radians


def _shape_from_potential(potential):
    """Resolve ``(sigma0, kappa)`` from a potential, falling back to the package
    defaults when ``potential`` is ``None`` or doesn't expose a shape (e.g. a
    non-Gay-Berne potential). Lets an initializer build geometry for whatever
    potential the simulation will use rather than the import-time default."""
    if potential is None:
        return SIGMA0, KAPPA
    sigma0 = getattr(potential, "sigma0", None)
    kappa = getattr(potential, "kappa", None)
    return (
        SIGMA0 if sigma0 is None else sigma0,
        KAPPA if kappa is None else kappa,
    )


class Initializer(ABC):
    """Builds the starting frame for a :class:`MetropolisCalculator` and records
    how it was built.

    Subclasses implement :meth:`generate` (return a fresh ``ase.Atoms``) and
    expose ``n_particles``/``density``/``volume``/``seed``. :meth:`provenance`
    is stamped onto the frame so the source of the initial config travels with
    the run's outputs.
    """

    n_particles = None
    density = None
    volume = None
    seed = None
    sigma0 = None  # shape the geometry is built at (None => not lattice-based)
    kappa = None

    @abstractmethod
    def generate(self) -> ase.Atoms:
        """Return a fresh ``ase.Atoms`` frame to start the simulation from."""
        ...

    def set_potential(self, potential):
        """Adopt the simulated potential's shape so the built geometry matches
        the potential the MC runs. Called by :class:`MetropolisCalculator`. The
        base implementation is a no-op (e.g. :class:`FrameInitializer`, which
        wraps an existing frame and synthesises no lattice); lattice builders
        override it, keeping any shape they were constructed with explicitly."""

    def provenance(self):
        """Compact, JSON-serialisable record of how the frame was built."""
        prov = {
            "init_n_particles": (
                None if self.n_particles is None else int(self.n_particles)
            ),
            "init_density": None if self.density is None else float(self.density),
            "init_seed": self.seed,
        }
        # rho* and contact distances are meaningless without the shape they were
        # built at, so record it whenever the start is lattice-based.
        if self.sigma0 is not None:
            prov["init_sigma0"] = float(self.sigma0)
        if self.kappa is not None:
            prov["init_kappa"] = float(self.kappa)
        return prov


class RandomLatticeInitializer(Initializer):
    """Generate a fresh jittered simple-cubic config via
    :func:`generate_random_config`."""

    def __init__(self, n_particles=None, density=None, seed=None, potential=None):
        self.n_particles = DEFAULT_N_PARTICLES if n_particles is None else n_particles
        self.density = DEFAULT_DENSITY if density is None else density
        self.seed = seed
        # SC lattice spacing only depends on sigma0; kappa is left None.
        self._potential_explicit = potential is not None
        self.sigma0, _ = _shape_from_potential(potential)

    def set_potential(self, potential):
        if not self._potential_explicit and potential is not None:
            self.sigma0, _ = _shape_from_potential(potential)

    def generate(self):
        frame = generate_random_config(
            n_particles=self.n_particles,
            density=self.density,
            seed=self.seed,
            sigma0=self.sigma0,
        )
        self.volume = frame.get_volume()
        return frame

    def provenance(self):
        prov = super().provenance()
        prov["init_packing"] = "random"
        return prov


class ColumnarLatticeInitializer(Initializer):
    """Generate a fresh, ordered columnar config via
    :func:`generate_columnar_config`.

    Builds discs stacked face-to-face into columns at near-equilibrium density —
    the *fast* equilibration direction for these oblate particles (melting an
    ordered start is barrier-free; freezing a disordered one is not). Seeded
    per-replica jitter/tilt keeps repeat trials statistically independent.
    """

    def __init__(
        self, n_particles=None, density=None, seed=None, tilt=0.15, potential=None
    ):
        self.n_particles = DEFAULT_N_PARTICLES if n_particles is None else n_particles
        self.density = DEFAULT_COLUMNAR_DENSITY if density is None else density
        self.seed = seed
        self.tilt = tilt
        self._potential_explicit = potential is not None
        self.sigma0, self.kappa = _shape_from_potential(potential)

    def set_potential(self, potential):
        if not self._potential_explicit and potential is not None:
            self.sigma0, self.kappa = _shape_from_potential(potential)

    def generate(self):
        frame = generate_columnar_config(
            n_particles=self.n_particles,
            density=self.density,
            seed=self.seed,
            tilt=self.tilt,
            sigma0=self.sigma0,
            kappa=self.kappa,
        )
        self.volume = frame.get_volume()
        return frame

    def provenance(self):
        prov = super().provenance()
        prov["init_packing"] = "columnar"
        prov["init_tilt"] = float(self.tilt)
        return prov


class HerringboneLatticeInitializer(Initializer):
    """Generate the benzene *herringbone* crystal via
    :func:`generate_herringbone_config`.

    Tiles the coarse-grained experimental benzene unit cell (Pbca) into a
    supercell: a T-shaped, low-nematic (S~0.25) arrangement — benzene's *real*
    crystal. This is the *validation* start (compare density/energy/order to
    experiment), as opposed to the parallel-stacked columnar polymorph. N is
    quantized to whole unit cells (4*nx*ny*nz), so ``n_particles`` is a target
    the realized count snaps to (e.g. 125 -> 128). Seeded position/orientation
    jitter (small by default) keeps repeat trials independent while preserving
    the herringbone order.
    """

    def __init__(
        self,
        n_particles=None,
        density=None,
        seed=None,
        pos_jitter=DEFAULT_HERRINGBONE_POS_JITTER,
        or_jitter=DEFAULT_HERRINGBONE_OR_JITTER,
        potential=None,
        motif_path=None,
    ):
        self.seed = seed
        self.pos_jitter = pos_jitter
        self.or_jitter = or_jitter
        self.motif_path = motif_path
        self._requested_n = DEFAULT_N_PARTICLES if n_particles is None else n_particles
        self._requested_density = density
        self._potential_explicit = potential is not None
        self.sigma0, self.kappa = _shape_from_potential(potential)
        self._resolve_geometry()

    def _resolve_geometry(self):
        """Snap the target N to whole unit cells, recording the realized count
        and reduced density so provenance is accurate before generate()."""
        *_, edges = _load_herringbone_motif(self.motif_path)
        nx, ny, nz = _herringbone_reps(self._requested_n, edges)
        self.reps = (nx, ny, nz)
        self.n_particles = 4 * nx * ny * nz
        v0 = float(np.prod(np.array([nx, ny, nz]) * edges))
        v = (
            v0
            if self._requested_density is None
            else self.n_particles * self.sigma0**3 / self._requested_density
        )
        self.density = self.n_particles * self.sigma0**3 / v  # realized rho*

    def set_potential(self, potential):
        if not self._potential_explicit and potential is not None:
            self.sigma0, self.kappa = _shape_from_potential(potential)
            self._resolve_geometry()

    def generate(self):
        frame = generate_herringbone_config(
            n_particles=self._requested_n,
            density=self._requested_density,
            seed=self.seed,
            pos_jitter=self.pos_jitter,
            or_jitter=self.or_jitter,
            sigma0=self.sigma0,
            kappa=self.kappa,
            motif_path=self.motif_path,
        )
        self.volume = frame.get_volume()
        return frame

    def provenance(self):
        prov = super().provenance()
        prov["init_packing"] = "herringbone"
        prov["init_pos_jitter"] = float(self.pos_jitter)
        prov["init_or_jitter"] = float(self.or_jitter)
        return prov


class FrameInitializer(Initializer):
    """Start from a caller-supplied frame, recording its derived properties."""

    def __init__(self, init_frame):
        self.init_frame = init_frame
        self.n_particles = len(init_frame)
        self.volume = init_frame.get_volume()
        self.density = self.n_particles / self.volume

    def generate(self):
        return self.init_frame


def generate_random_config(n_particles=210, density=0.6, seed=None, sigma0=None):
    """
    Build an ASE Atoms frame of N coarse-grained benzene particles on a
    jittered simple-cubic lattice with uniformly random orientations.

    density : reduced number density rho* = N * sigma0^3 / V
    seed    : integer RNG seed for reproducibility (None for random)
    sigma0  : particle size setting the box scale and hard-core spacing; defaults
              to the package default (use the simulated potential's sigma0).
    """
    if sigma0 is None:
        sigma0 = SIGMA0
    volume = n_particles * sigma0**3 / density
    box_length = volume ** (1 / 3)
    n_side = int(np.ceil(n_particles ** (1 / 3)))
    spacing = box_length / n_side

    if spacing < sigma0:
        raise ValueError(
            f"density={density} too high: lattice spacing {spacing:.2f} Å "
            f"< sigma0={sigma0:.2f} Å — hard-core overlaps unavoidable."
        )

    rng = np.random.default_rng(seed)

    # jittered SC lattice positions
    lattice = (
        np.array(
            [
                [i, j, k]
                for i in range(n_side)
                for j in range(n_side)
                for k in range(n_side)
            ]
        )
        * spacing
    )
    positions = lattice[:n_particles].copy()
    jitter_max = 0.9 * (spacing - sigma0) / 2
    positions += rng.uniform(-jitter_max, jitter_max, positions.shape)
    positions %= box_length  # wrap PBC boundary cases

    # uniformly random orientations on SO(3)
    # scipy uses scalar-last [x,y,z,w]; roll to scalar-first [w,x,y,z] for c_q
    rot = Rotation.random(n_particles, random_state=int(rng.integers(2**31)))
    quats = np.roll(rot.as_quat(), 1, axis=-1)
    or_vecs = np.array([calc_or_vec(q).squeeze() for q in quats])

    frame = ase.Atoms(
        symbols="X" * n_particles,
        positions=positions,
        cell=np.eye(3) * box_length,
        pbc=True,
    )
    frame.new_array("c_q", quats)
    frame.new_array("or_vec", or_vecs)
    return frame


def generate_columnar_config(
    n_particles=210, density=1.4, seed=None, tilt=0.15, sigma0=None, kappa=None
):
    """
    Build an ASE Atoms frame of N oblate (discotic) particles arranged as
    aligned columns: discs stacked face-to-face along z into columns, the
    columns tiling the xy-plane.

    For oblate Gay-Berne particles (kappa < 1) the face-to-face contact distance
    is kappa*sigma0 (axial) and the side-to-side contact is sigma0 (in-plane), so
    columns reach densities the random simple-cubic start cannot (rho* up to
    ~1/kappa). This is the ordered, near-equilibrium start that equilibrates in
    the fast direction for the dense phases.

    density : reduced number density rho* = N * sigma0^3 / V (realized exactly;
              box is sized for N). Raises if too high to place without overlaps.
    seed    : integer RNG seed for reproducibility (None for random); distinct
              seeds give statistically independent starts for repeat trials.
    tilt    : max per-particle orientation jitter (radians) about the column
              axis — breaks the perfect-alignment symmetry while keeping high S.
    sigma0,
    kappa   : particle size and aspect ratio setting the in-plane (sigma0) and
              axial (kappa*sigma0) contact distances; default to the package
              default (pass the simulated potential's values).
    """
    if sigma0 is None:
        sigma0 = SIGMA0
    if kappa is None:
        kappa = KAPPA
    # Near-cubic grid: discs stack along z at axial spacing kappa*s, columns tile
    # the xy-plane at in-plane spacing s. For a cubic-ish box with N ~ n_x*n_y*n_z
    # the column height is n_z ~ (N/kappa^2)^(1/3); the columns are then tiled
    # near-square in-plane (n_x ~ n_y), allowing a rectangular in-plane grid so
    # arbitrary N packs with little waste. Over-provision and take the first N
    # (whole columns first), mirroring generate_random_config's SC fill.
    n_z = max(1, round((n_particles / kappa**2) ** (1 / 3)))
    n_col = int(np.ceil(n_particles / n_z))
    n_x = int(np.ceil(np.sqrt(n_col)))
    n_y = int(np.ceil(n_col / n_x))
    n_grid = n_x * n_y * n_z

    # Box sized for N at the requested density. Every in-plane spacing equals s
    # and the axial spacing is kappa*s, so both reach contact (sigma0 in-plane,
    # kappa*sigma0 axial) together — a single condition (s >= sigma0) guarantees
    # no overlaps. V = n_grid * kappa * s^3 fixes s for the requested density.
    volume = n_particles * sigma0**3 / density
    spacing_xy = (volume / (n_grid * kappa)) ** (1 / 3)
    spacing_z = kappa * spacing_xy

    if spacing_xy < sigma0:
        max_density = n_particles / (n_grid * kappa)
        raise ValueError(
            f"density={density} too high for a columnar lattice of {n_particles} "
            f"particles: in-plane spacing {spacing_xy:.2f} Å < sigma0="
            f"{sigma0:.2f} Å — overlaps unavoidable. Max ~{max_density:.2f}."
        )

    box = np.array([n_x * spacing_xy, n_y * spacing_xy, n_z * spacing_z])

    rng = np.random.default_rng(seed)

    # column-major fill (k innermost) so the first N sites form whole columns
    sites = np.array(
        [
            [i * spacing_xy, j * spacing_xy, k * spacing_z]
            for i in range(n_x)
            for j in range(n_y)
            for k in range(n_z)
        ]
    )
    positions = sites[:n_particles].copy()

    # anisotropic jitter: free space is (spacing - contact) on each axis, and
    # axial free space = kappa * in-plane free space (spacing_z = kappa*spacing_xy)
    jitter_xy = 0.45 * (spacing_xy - sigma0)
    jitter_z = 0.45 * (spacing_z - kappa * sigma0)
    positions[:, :2] += rng.uniform(-jitter_xy, jitter_xy, (n_particles, 2))
    positions[:, 2] += rng.uniform(-jitter_z, jitter_z, n_particles)
    positions %= box  # wrap PBC boundary cases

    # orientations: symmetry axis along z (face-to-face stacking) tilted by a
    # small seeded random rotation; scipy is scalar-last [x,y,z,w], roll to
    # scalar-first [w,x,y,z] for c_q
    axes = rng.normal(size=(n_particles, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0, tilt, n_particles)
    rot = Rotation.from_rotvec(axes * angles[:, None])
    quats = np.roll(rot.as_quat(), 1, axis=-1)
    or_vecs = np.array([calc_or_vec(q).squeeze() for q in quats])

    frame = ase.Atoms(
        symbols="X" * n_particles,
        positions=positions,
        cell=np.diag(box),
        pbc=True,
    )
    frame.new_array("c_q", quats)
    frame.new_array("or_vec", or_vecs)
    return frame


def _min_center_distance(frame):
    """Smallest center-center distance (minimum-image), for the overlap floor."""
    d = frame.get_all_distances(mic=True)
    return d[np.triu_indices(len(frame), k=1)].min()


def _load_herringbone_motif(path=None):
    """Load the coarse-grained benzene herringbone unit cell, returning
    ``(coms[M,3], or_vec[M,3], c_q[M,4], edges[3])``. ``path`` (default the
    tracked motif) resolves against the package, not cwd."""
    motif = ase.io.read(str(DEFAULT_HERRINGBONE_MOTIF if path is None else path))
    assert isinstance(motif, ase.Atoms), "herringbone motif must be a single frame"
    edges = np.diag(np.asarray(motif.cell))
    return (
        motif.positions.copy(),
        motif.arrays["or_vec"].copy(),
        motif.arrays["c_q"].copy(),
        edges,
    )


def _herringbone_reps(target_n, edges, max_rep=8, max_aniso=2.0):
    """Choose supercell repeats ``(nx, ny, nz)`` tiling the herringbone unit cell
    to about ``target_n`` particles (= 4*nx*ny*nz) with the most compact box.
    Whole unit cells only, so N is quantized (e.g. 125 -> 128 as 4x4x2)."""
    target = DEFAULT_N_PARTICLES if target_n is None else target_n
    best_key, best = None, (1, 1, 1)
    for nx in range(1, max_rep + 1):
        for ny in range(1, max_rep + 1):
            for nz in range(1, max_rep + 1):
                n = 4 * nx * ny * nz
                lo, _, hi = sorted(np.array([nx, ny, nz]) * edges)
                aniso = hi / lo
                # closeness to target dominates; boxes past max_aniso are penalised
                key = (abs(n - target) + 1000 * max(0.0, aniso - max_aniso), aniso, n)
                if best_key is None or key < best_key:
                    best_key, best = key, (nx, ny, nz)
    return best


def generate_herringbone_config(
    n_particles=None,
    density=None,
    seed=None,
    pos_jitter=DEFAULT_HERRINGBONE_POS_JITTER,
    or_jitter=DEFAULT_HERRINGBONE_OR_JITTER,
    sigma0=None,
    kappa=None,
    motif_path=None,
):
    """
    Build an ASE Atoms frame of N oblate particles in the benzene *herringbone*
    crystal by tiling the coarse-grained experimental Pbca unit cell.

    The motif (``motif_path``, default the tracked benzene coarse-graining) holds
    4 particles on the crystal's inversion centers with the experimental
    ring-normal orientations — a low-nematic (S~0.25), T-shaped arrangement, i.e.
    benzene's real crystal. This is the *validation* start: it puts the system in
    benzene's actual structure so its density/energy/order can be compared to
    experiment, unlike the columnar start (a parallel-stacked polymorph).

    n_particles : target particle count. N is quantized to 4*nx*ny*nz (whole unit
                  cells keep the crystal intact); the supercell nearest the target
                  with a compact box is chosen, so realized N may differ
                  (125 -> 128 as a 4x4x2 tiling).
    density     : reduced number density rho* = N*sigma0^3/V. None keeps the
                  experimental cell (rho* ~ 1.5, rho ~ 1.05 g/mL); a value
                  isotropically rescales the crystal to that density (raises if so
                  dense that centers close inside the kappa*sigma0 contact).
    seed        : RNG seed; distinct seeds give independent jittered starts.
    pos_jitter  : max per-particle position displacement (Angstrom, per axis).
    or_jitter   : max per-particle orientation libration (radians) about a random
                  axis — keep small so the herringbone order is preserved (a
                  thermal wiggle, not a re-randomization).
    sigma0,
    kappa       : particle size / aspect ratio; sigma0 sets rho*, kappa sets the
                  overlap floor (kappa*sigma0). Default to the package default;
                  pass the simulated potential's values.
    """
    if sigma0 is None:
        sigma0 = SIGMA0
    if kappa is None:
        kappa = KAPPA

    coms0, or0, cq0, edges = _load_herringbone_motif(motif_path)
    nx, ny, nz = _herringbone_reps(n_particles, edges)
    a, b, c = edges
    ncell = nx * ny * nz
    m = len(coms0)
    n = ncell * m

    # tile the M-particle motif over the nx*ny*nz supercell (cell-major order,
    # so the tiled or_vec/c_q line up with positions)
    offsets = np.array(
        [[i * a, j * b, k * c] for i in range(nx) for j in range(ny) for k in range(nz)]
    )
    positions = (coms0[None] + offsets[:, None]).reshape(-1, 3)
    or_vecs = np.broadcast_to(or0, (ncell, m, 3)).reshape(-1, 3).copy()
    quats = np.broadcast_to(cq0, (ncell, m, 4)).reshape(-1, 4).copy()
    box = np.array([nx * a, ny * b, nz * c])

    # optionally rescale the crystal isotropically to a target reduced density
    if density is not None:
        f = (n * sigma0**3 / density / (nx * a * ny * b * nz * c)) ** (1 / 3)
        positions *= f
        box *= f

    rng = np.random.default_rng(seed)
    if pos_jitter:
        positions += rng.uniform(-pos_jitter, pos_jitter, positions.shape)
    positions %= box  # wrap PBC boundary cases

    # small orientation libration about random axes, composed onto the crystal
    # orientation; recompute or_vec/c_q from the librated rotation
    if or_jitter:
        axes = rng.normal(size=(n, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        angles = rng.uniform(0, or_jitter, n)
        librated = Rotation.from_rotvec(axes * angles[:, None]) * Rotation.from_quat(
            np.roll(quats, -1, axis=1)  # scalar-first -> scalar-last for scipy
        )
        quats = np.roll(librated.as_quat(), 1, axis=1)
        or_vecs = librated.apply([0.0, 0.0, 1.0])

    frame = ase.Atoms(symbols="X" * n, positions=positions, cell=np.diag(box), pbc=True)
    frame.new_array("c_q", quats)
    frame.new_array("or_vec", or_vecs)

    if density is not None:
        min_dist = _min_center_distance(frame)
        if min_dist < kappa * sigma0:
            raise ValueError(
                f"density={density} too high for the herringbone crystal: min "
                f"center distance {min_dist:.2f} A < kappa*sigma0={kappa * sigma0:.2f} A "
                "— overlaps unavoidable."
            )
    return frame
