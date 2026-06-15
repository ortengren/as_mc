from abc import ABC, abstractmethod

import numpy as np
import ase
from scipy.spatial.transform import Rotation
from asmcmc.potentials import GB_PARAMS
from asmcmc.trial_moves import calc_or_vec

SIGMA0 = GB_PARAMS["sigma0"]

DEFAULT_N_PARTICLES = 210
DEFAULT_DENSITY = 0.3


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

    @abstractmethod
    def generate(self) -> ase.Atoms:
        """Return a fresh ``ase.Atoms`` frame to start the simulation from."""
        ...

    def provenance(self):
        """Compact, JSON-serialisable record of how the frame was built."""
        return {
            "init_n_particles": None if self.n_particles is None else int(self.n_particles),
            "init_density": None if self.density is None else float(self.density),
            "init_seed": self.seed,
        }


class RandomLatticeInitializer(Initializer):
    """Generate a fresh jittered simple-cubic config via
    :func:`generate_random_config`."""

    def __init__(self, n_particles=None, density=None, seed=None):
        self.n_particles = DEFAULT_N_PARTICLES if n_particles is None else n_particles
        self.density = DEFAULT_DENSITY if density is None else density
        self.seed = seed

    def generate(self):
        frame = generate_random_config(
            n_particles=self.n_particles, density=self.density, seed=self.seed
        )
        self.volume = frame.get_volume()
        return frame


class FrameInitializer(Initializer):
    """Start from a caller-supplied frame, recording its derived properties."""

    def __init__(self, init_frame):
        self.init_frame = init_frame
        self.n_particles = len(init_frame)
        self.volume = init_frame.get_volume()
        self.density = self.n_particles / self.volume

    def generate(self):
        return self.init_frame


def generate_random_config(n_particles=210, density=0.3, seed=None):
    """
    Build an ASE Atoms frame of N coarse-grained benzene particles on a
    jittered simple-cubic lattice with uniformly random orientations.

    density : reduced number density rho* = N * sigma0^3 / V
    seed    : integer RNG seed for reproducibility (None for random)
    """
    volume = n_particles * SIGMA0**3 / density
    box_length = volume ** (1 / 3)
    n_side = int(np.ceil(n_particles ** (1 / 3)))
    spacing = box_length / n_side

    if spacing < SIGMA0:
        raise ValueError(
            f"density={density} too high: lattice spacing {spacing:.2f} Å "
            f"< sigma0={SIGMA0:.2f} Å — hard-core overlaps unavoidable."
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
    jitter_max = 0.9 * (spacing - SIGMA0) / 2
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
