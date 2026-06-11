import numpy as np
import ase
from scipy.spatial.transform import Rotation
from potentials import GB_PARAMS
from trial_moves import calc_or_vec

SIGMA0 = GB_PARAMS["sigma0"]


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
    lattice = np.array(
        [[i, j, k]
         for i in range(n_side)
         for j in range(n_side)
         for k in range(n_side)]
    ) * spacing
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
