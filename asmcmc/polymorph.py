"""Pair and shell geometry for coarse-grained disc frames, and disorder control.

Frame-level functions take an ``ase.Atoms`` of coarse-grained centres carrying
unit disc normals, the layout :func:`asmcmc.utils.coarse_grain_frame` produces.
Orientations may instead be passed explicitly via ``or_vecs``: ASE db rows keep
them in ``row.data``, not in ``row.toatoms()``.

What is here is deliberately only the *measurement* layer:

- :func:`pair_geometry` and :func:`neighbour_shell` enumerate contacts (the
  latter by a fixed neighbour *count*, so the shell adapts to whatever density
  an NPT box is at), and :func:`contact_invariants` reduces a pair to the two
  numbers that fix it up to rotation.
- :func:`thermal_jitter` and :func:`disorder_amplitude` are an inverse pair for
  thermal broadening: a cif is a *mean* structure and an MC frame is one
  *instantaneous* configuration, so comparing them directly charges thermal
  motion to whatever else is being measured. ``disorder_amplitude`` measures the
  spread about a site over a trajectory in exactly the units ``thermal_jitter``
  consumes. :func:`rescale` does the same job for density.

**Removed: the classification layer that used to sit on top of this.** A
contact taxonomy, an inter-normal-angle spectrum and an AniSOAP environment
distance were built here to identify which polymorph a structure is. They are
all *k*-nearest-neighbour statistics over rotation- and sign-invariant pair
quantities, which makes them blind to a difference they were being used to rule
out: two crystals built from the same motif but on different lattices score the
same. Measured on the 100 K validation run against the Cacelli energy minimum
relaxed from the cif -- both slipped-parallel, both ~33% face-to-face contacts,
stack height 2.6 vs 2.5 A and slip 4.0 vs 4.0 A, indistinguishable by every one
of those descriptors -- while the centre lattices differ by 7% on one axis
(5.37, 4.71, 5.65 A against 5.33, 4.73, 5.26 A at matched density) and the
radial distribution functions differ at 2.8x the frame-to-frame floor. The
descriptors were reporting agreement that is not there, so the claims resting
on them were withdrawn along with the code.

Anything rebuilt on this module needs at least one descriptor sensitive to the
lattice, not only to the neighbour shell.
"""

import numpy as np
from ase.neighborlist import neighbor_list

DEFAULT_K = 12  # neighbours per molecule; benzene's coordination number


def _orientations(frame, or_vecs=None):
    """Unit disc normals for ``frame``, from ``or_vecs`` if given."""
    u = frame.arrays["or_vec"] if or_vecs is None else or_vecs
    u = np.asarray(u, dtype=float)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def pair_geometry(frame, cutoff, or_vecs=None):
    """All directed neighbour pairs within ``cutoff`` (PBC, incl. self-images).

    The same quantities and conventions as
    ``fitting_gbq.data.extract_periodic_pairs`` -- so pair statistics here stay
    comparable with the fitting code -- but the centre index ``i`` is kept,
    which that function discards. Selecting the *k* nearest neighbours of each
    molecule needs it; a global sort over all pairs is not the same thing once
    the local density varies.

    Returns a dict of equal-length arrays: ``i``, ``j``, ``r`` (separation),
    ``a_i``/``a_j`` (``r_hat . u``), ``b`` (``u_i . u_j``).
    """
    u = _orientations(frame, or_vecs)
    i, j, offsets = neighbor_list("ijS", frame, cutoff)
    disp = frame.positions[j] + offsets @ np.asarray(frame.cell) - frame.positions[i]
    r = np.linalg.norm(disp, axis=1)
    r_hat = disp / r[:, None]
    return {
        "i": i,
        "j": j,
        "r": r,
        "a_i": np.einsum("pk,pk->p", r_hat, u[i]),
        "a_j": np.einsum("pk,pk->p", r_hat, u[j]),
        "b": np.einsum("pk,pk->p", u[i], u[j]),
    }


def _shell_cutoff(frame, k):
    """Radius expected to enclose ``k`` neighbours at the frame's mean density.

    Inverting n = (4/3) pi R^3 rho with a margin, so the default adapts to
    whatever volume the NPT box happens to be at instead of pinning a length
    that only suits one density.
    """
    volume_per_particle = frame.get_volume() / len(frame)
    return 1.5 * (3.0 * k * volume_per_particle / (4.0 * np.pi)) ** (1.0 / 3.0)


def neighbour_shell(frame, k=DEFAULT_K, cutoff=None, or_vecs=None):
    """The ``k`` nearest neighbours of every molecule, as a `pair_geometry` dict.

    A fixed distance cutoff would make every contact statistic a function of
    density; a fixed *count* makes them scale-invariant, so two structures at
    different volumes can be compared directly. ``cutoff`` only sets the search
    radius and is grown automatically until every molecule has ``k`` neighbours.
    """
    n = len(frame)
    search = _shell_cutoff(frame, k) if cutoff is None else cutoff
    # A generous ceiling that only stops the growth loop from running away: any
    # frame still short of k neighbours at twice its shortest lattice vector is
    # not going to reach it (a non-periodic frame with too few molecules), and
    # the neighbour list gets expensive fast. Reaching into periodic images
    # before this point is legitimate and expected -- it is how a 4-molecule
    # unit cell still yields a full shell.
    max_search = 2.0 * frame.cell.lengths().min()

    while True:
        pairs = pair_geometry(frame, search, or_vecs)
        counts = np.bincount(pairs["i"], minlength=n)
        if counts.min() >= k:
            break
        if search >= max_search:
            raise ValueError(
                f"cannot find {k} neighbours for every molecule within "
                f"{search:.1f} A (min found: {counts.min()}); the frame is too "
                "small or too dilute for this shell size"
            )
        search *= 1.5

    # Rank each centre's neighbours by distance and keep the closest k. lexsort
    # orders by centre first, distance second, so ranks are just the offset from
    # each centre's block start.
    order = np.lexsort((pairs["r"], pairs["i"]))
    starts = np.searchsorted(pairs["i"][order], np.arange(n))
    rank = np.arange(len(order)) - starts[pairs["i"][order]]
    keep = order[rank < k]
    return {key: value[keep] for key, value in pairs.items()}


def contact_invariants(shell):
    """``(gamma_deg, a_min, a_max)`` for a shell of pairs.

    The full state of a disc pair, up to rotation: the angle between the two
    normals and how axially the separation sits against the less- and
    more-aligned of them.

    Both are taken on absolute values, so a pair and its inversion give the same
    numbers. That is what makes them rotation-invariant and it is also the limit
    of what they can distinguish -- see the module docstring before building a
    structural claim on top of them.
    """
    gamma = np.degrees(np.arccos(np.clip(np.abs(shell["b"]), 0.0, 1.0)))
    a_i, a_j = np.abs(shell["a_i"]), np.abs(shell["a_j"])
    return gamma, np.minimum(a_i, a_j), np.maximum(a_i, a_j)


def rescale(frame, volume_per_particle):
    """Copy of ``frame`` scaled isotropically to a target volume per particle.

    Any descriptor that is not scale-invariant -- a radial distribution
    function, a lattice metric -- reads a density difference as a structural
    one, so comparing two structures at different densities needs a
    density-matched twin to separate "packs differently" from "packs more
    tightly".
    """
    scaled = frame.copy()
    if "or_vec" in frame.arrays:
        scaled.arrays["or_vec"] = frame.arrays["or_vec"].copy()
    factor = volume_per_particle / (frame.get_volume() / len(frame))
    scaled.set_cell(frame.cell * factor ** (1.0 / 3.0), scale_atoms=True)
    return scaled


def thermal_jitter(frame, pos_sigma, tilt_sigma, seed=None):
    """Copy of ``frame`` with centres displaced and normals tilted at random.

    Turns a crystallographic *mean* structure into something comparable with an
    *instantaneous* MC frame. Centres get isotropic Gaussian noise of standard
    deviation ``pos_sigma`` per Cartesian component; normals get tilted by a
    random angle whose RMS is ``tilt_sigma`` radians.

    The tilt axis is drawn perpendicular to the normal, not uniformly in 3D.
    Spin about a uniaxial disc's own normal is unphysical -- it moves no
    observable -- so a uniform 3D axis would spend part of its amplitude on a
    degree of freedom that does not exist and leave the parameter meaning
    something other than the tilt it produces. Here the parameter *is* the
    observable: jitter at ``tilt_sigma`` and the realised RMS of
    ``arccos|u . u'|`` comes back as ``tilt_sigma``, which is what makes
    :func:`disorder_amplitude` usable as a calibration.

    Amplitudes are absolute lengths, so matching the *relative* disorder of a
    structure at a different density means scaling ``pos_sigma`` by the ratio of
    mean spacings, ``(v_here / v_there) ** (1 / 3)``.
    """
    rng = np.random.default_rng(seed)
    jittered = frame.copy()
    jittered.positions = frame.positions + rng.normal(
        0.0, pos_sigma, frame.positions.shape
    )

    u = _orientations(frame)
    # An orthonormal basis of the plane perpendicular to each normal. The seed
    # vector only has to be non-parallel to u; the choice of which axis to cross
    # with is per-molecule so it never degenerates.
    seed_axis = np.where(
        (np.abs(u[:, 2]) < 0.9)[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    )
    e1 = np.cross(u, seed_axis)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)

    # Two independent tilt directions, so the tilt magnitude is chi with 2 dof:
    # sigma/sqrt(2) per direction makes E[alpha^2] = tilt_sigma^2 exactly.
    components = rng.normal(0.0, tilt_sigma / np.sqrt(2.0), (len(frame), 2))
    tilt = e1 * components[:, :1] + e2 * components[:, 1:]
    alpha = np.linalg.norm(tilt, axis=1, keepdims=True)
    direction = np.divide(tilt, alpha, out=np.zeros_like(tilt), where=alpha > 0)
    # Rotating u by alpha about an axis perpendicular to it tilts u towards
    # `direction`; both are unit and orthogonal, so this stays normalised.
    jittered.arrays["or_vec"] = u * np.cos(alpha) + direction * np.sin(alpha)
    return jittered


def disorder_amplitude(frames, lag=1, remove_drift=True):
    """How far molecules wander from their mean site, measured on a trajectory.

    Returns ``pos_sigma`` (A, per Cartesian component) and ``tilt_sigma`` (rad,
    RMS angle the normal moves) in exactly the convention
    :func:`thermal_jitter` consumes, so a reference structure can be disordered
    to match a trajectory instead of being compared against it as a mean
    structure. It measures *spread about a site*, not what the sites are, so it
    is unaffected by which packing the trajectory is in -- which is what lets
    the MC run calibrate the disorder of a herringbone reference.

    Both come from the two-time displacement between frames ``lag`` apart,
    halved: two decorrelated draws from the same well differ by twice the
    single-frame variance. Positions are differenced in *scaled* coordinates
    and minimum-imaged, so an NPT cell that breathes and a molecule that crosses
    a boundary neither register as motion; ``remove_drift`` additionally
    subtracts the box-wide mean displacement.

    **Check the lag dependence rather than trusting one value.** The estimate
    only means "vibrational amplitude" where it has stopped growing with
    ``lag``; if it keeps climbing, the crystal is slowly rearranging and the
    number is an upper bound that depends on how long you watched.
    """
    frames = list(frames)
    if len(frames) <= lag:
        raise ValueError(f"need more than {lag} frames to measure at lag {lag}")

    scaled = np.stack([f.get_scaled_positions() for f in frames])
    mean_cell = np.mean([np.asarray(f.cell) for f in frames], axis=0)
    delta = scaled[lag:] - scaled[:-lag]
    delta -= np.round(delta)
    displacement = delta @ mean_cell
    if remove_drift:
        displacement -= displacement.mean(axis=1, keepdims=True)
    pos_sigma = np.sqrt((displacement**2).sum(axis=2).mean() / 6.0)

    u = np.stack([_orientations(f) for f in frames])
    # +/-u are the same disc, so the angle is taken on |u . u'|.
    cosine = np.clip(np.abs(np.einsum("tnk,tnk->tn", u[lag:], u[:-lag])), 0.0, 1.0)
    tilt_sigma = np.sqrt((np.arccos(cosine) ** 2).mean() / 2.0)

    return {
        "pos_sigma": float(pos_sigma),
        "tilt_sigma": float(tilt_sigma),
        "tilt_deg": float(np.degrees(tilt_sigma)),
        "lag": int(lag),
        "n_frames": len(frames),
    }
