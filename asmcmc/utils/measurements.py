import ase
from ase.db import connect
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from asmcmc.base.potentials import calc_total_energy

BOLTZCONST = 8.617e-5  # eV / K


def nematic_q_tensor(or_vecs):
    """Symmetric, traceless ordering (Q) tensor for a set of unit axes.

        Q_ab = (1/N) sum_i (3 u_ia u_ib - delta_ab) / 2

    Its largest eigenvalue is the nematic/discotic order parameter S:
    ~0 for random orientations (isotropic), -> 1 when all axes align.
    """
    u = np.asarray(or_vecs)
    n = len(u)
    return (3.0 * np.einsum("ia,ib->ab", u, u) - n * np.eye(3)) / (2.0 * n)


class Measurement(ABC):
    """Base class for all measurements."""

    def __init__(self):
        self.results = []

    @abstractmethod
    def compute(self, frame, scalar_data, array_data):
        """Calculate metric for a single frame"""
        pass

    @abstractmethod
    def finalize(self):
        """Aggregate data across frames"""
        pass


class AverageEnergy(Measurement):
    """Mean and variance of the system energy over a trajectory.

    By default reads the energy tracked in ``scalar_data["total_energy"]`` (the
    sampler's incrementally maintained value). Pass ``recompute=True`` to
    instead recompute each frame's energy from scratch with
    ``calc_total_energy`` which avoids any drift in the incremental tracker
    over a long run, at the cost of a full O(N*neighbors) evaluation per stored
    frame. ``nl_radius`` (the per-particle neighbour-list radius the run used)
    is required when recomputing.
    """

    def __init__(self, recompute=False, nl_radius=None, potential=None):
        super().__init__()
        self.energies = []
        self.recompute = recompute
        self.nl_radius = nl_radius
        self.potential = potential
        if recompute and nl_radius is None:
            raise ValueError("recompute=True requires nl_radius")

    def compute(self, frame, scalar_data, array_data):
        if self.recompute:
            # A frame read back from a db (row.toatoms()) keeps positions/cell/
            # pbc but drops custom arrays such as or_vec, which calc_total_energy
            # needs -- restore it from array_data when present.
            if array_data is not None and "or_vec" in array_data:
                frame = frame.copy()
                frame.set_array("or_vec", np.asarray(array_data["or_vec"]))
            cutoffs = [self.nl_radius] * len(frame)
            energy = calc_total_energy(frame, cutoffs, potential=self.potential)
        else:
            energy = scalar_data["total_energy"]
        self.energies.append(energy)

    def finalize(self):
        avg_e = np.mean(self.energies)
        var_e = np.var(self.energies)
        return avg_e, var_e


class AverageEnthalpy(AverageEnergy):
    """Mean enthalpy H = U + P*V over an (NPT) trajectory.

    Reuses AverageEnergy's energy handling -- the tracked
    ``scalar_data["total_energy"]`` by default, or a from-scratch recompute with
    ``recompute=True`` (which needs ``nl_radius``) -- and adds the P*V term from
    each frame's volume (P in eV/A^3, V in A^3, matching the eV/A units). Only
    meaningful for a constant-pressure run. finalize() returns (mean, std) of the
    total enthalpy in eV; divide by the molecule count for a per-molecule value.
    """

    def __init__(self, pressure, recompute=False, nl_radius=None, potential=None):
        super().__init__(recompute=recompute, nl_radius=nl_radius, potential=potential)
        self.pressure = pressure

    def compute(self, frame, scalar_data, array_data):
        super().compute(frame, scalar_data, array_data)
        # AverageEnergy just appended this frame's U; fold in P*V to make it H
        self.energies[-1] += self.pressure * frame.get_volume()

    def finalize(self):
        h = np.asarray(self.energies)
        return float(h.mean()), float(h.std())


class RadialDistributionFunction(Measurement):
    def __init__(self, r_max, num_bins):
        super().__init__()
        self.r_max = r_max
        self.num_bins = num_bins

        self.bin_edges = np.linspace(0, r_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.shell_vols = (
            (4.0 / 3.0) * np.pi * (self.bin_edges[1:] ** 3 - self.bin_edges[:-1] ** 3)
        )

        # counts and their ideal-gas expectation are both accumulated per frame
        # so a fluctuating (NPT) box normalises correctly bin-by-bin
        self.hist_counts = np.zeros(num_bins)
        self.ideal_counts = np.zeros(num_bins)
        self.num_frames = 0

    def compute(self, frame, scalar_data, array_data):
        self.num_frames += 1
        n = len(frame)

        # The minimum-image convention only resolves separations out to half the
        # shortest box length; past that a spherical shell is no longer fully
        # contained in the cell, the histogram under-counts, and g(r) sags below
        # 1. The box floats in NPT, so cap each frame at its own L/2 and keep
        # only bins lying entirely inside it (others stay at their running value).
        half_box = frame.cell.lengths().min() / 2.0
        valid = self.bin_edges[1:] <= half_box

        dist_matrix = frame.get_all_distances(mic=True)
        i, j = np.triu_indices(n, k=1)
        counts, _ = np.histogram(dist_matrix[i, j], bins=self.bin_edges)
        self.hist_counts += np.where(valid, counts, 0)

        # ideal-gas expectation for unique pairs in each shell, with THIS frame's
        # density: N(N-1)/2 pairs each in the shell with probability shell/V
        density = n / frame.get_volume()
        ideal = (n - 1) / 2.0 * density * self.shell_vols
        self.ideal_counts += np.where(valid, ideal, 0.0)

    def finalize(self):
        g_r = np.zeros_like(self.hist_counts)
        np.divide(
            self.hist_counts,
            self.ideal_counts,
            out=g_r,
            where=(self.ideal_counts > 0),
        )
        return {
            "r": self.bin_centers,
            "g_r": g_r,
        }


class OrientationalCorrelationFunction(Measurement):
    def __init__(self, r_max, num_bins):
        super().__init__()
        self.r_max = r_max
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0, r_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.hist_counts = np.zeros(num_bins)
        self.sum_p2 = np.zeros(num_bins)

    def compute(self, frame, scalar_data, array_data):
        num_particles = len(frame)

        # get distances
        dist_matrix = frame.get_all_distances(mic=True)

        # get all dot products in a shape (N,N) array
        or_vecs = array_data["or_vec"]
        dot_matrix = or_vecs @ or_vecs.T

        # get only unique pairs
        i, j = np.triu_indices(num_particles, k=1)
        dists = dist_matrix[i, j]
        cos_thetas = dot_matrix[i, j]

        # calculate P2 (second Legendre polynomial) for all pairs
        p2_vals = 0.5 * (3.0 * cos_thetas**2 - 1.0)

        # build histogram for distances
        counts, _ = np.histogram(dists, bins=self.bin_edges)
        self.hist_counts += counts

        # build histogram for P2 values
        p2_sums, _ = np.histogram(dists, bins=self.bin_edges, weights=p2_vals)
        self.sum_p2 += p2_sums

    def finalize(self):
        # calculate average P2 value for each distance bin
        avg_p2 = np.zeros_like(self.sum_p2)
        np.divide(
            self.sum_p2, self.hist_counts, out=avg_p2, where=(self.hist_counts > 0)
        )

        return {
            "r": self.bin_centers,
            "s2_r": avg_p2,
        }


class NematicOrderParameter(Measurement):
    """Nematic order parameter S, averaged over a trajectory.

    Each frame's Q-tensor is diagonalised and its largest eigenvalue (the
    instantaneous S) is taken *before* averaging. That tracks the director
    even as it diffuses; diagonalising the frame-averaged <Q> instead would
    cancel director fluctuations and under-report order in an un-pinned box.
    The running <Q> is still accumulated and returned for a fixed-lab-frame
    cross-check and the mean director.

    finalize() returns a dict:
        S        mean per-frame order parameter (the primary estimator)
        S_std    its standard deviation across frames
        Q_mean   the frame-averaged Q-tensor (3x3)
        director eigenvector of Q_mean's largest eigenvalue
        S_lab    largest eigenvalue of Q_mean (order vs the fixed director)
    """

    def __init__(self):
        super().__init__()
        self.s_values = []
        self.q_sum = np.zeros((3, 3))
        self.num_frames = 0

    def compute(self, frame, scalar_data, array_data):
        q = nematic_q_tensor(array_data["or_vec"])
        self.s_values.append(np.linalg.eigvalsh(q)[-1])
        self.q_sum += q
        self.num_frames += 1

    def finalize(self):
        q_mean = self.q_sum / self.num_frames
        eigvals, eigvecs = np.linalg.eigh(q_mean)
        return {
            "S": float(np.mean(self.s_values)),
            "S_std": float(np.std(self.s_values)),
            "Q_mean": q_mean,
            "director": eigvecs[:, -1],
            "S_lab": float(eigvals[-1]),
        }


HC_OVER_K = 1.438777  # h c / k_B in cm K -- converts a wavenumber to a Theta

# eV/K per molecule -> J/(mol K), for comparison against calorimetry tables.
EV_PER_K_TO_J_PER_MOL_K = 96485.33

# (wavenumber [cm^-1], degeneracy) for the 20 distinct fundamentals of benzene.
# Shimanouchi's "selected frequency" values (NSRDS-NBS 39, 1972) received from
# the NIST Chemistry WebBook.
BENZENE_FUNDAMENTALS = (
    (3062.0, 1),  # 1   a1g  CH stretch
    (992.0, 1),  # 2   a1g  ring stretch (breathing)
    (1326.0, 1),  # 3   a2g  CH bend
    (673.0, 1),  # 4   a2u  CH bend (out of plane)
    (3068.0, 1),  # 5   b1u  CH stretch
    (1010.0, 1),  # 6   b1u  ring deformation
    (995.0, 1),  # 7   b2g  CH bend
    (703.0, 1),  # 8   b2g  ring deformation
    (1310.0, 1),  # 9   b2u  ring stretch
    (1150.0, 1),  # 10  b2u  CH bend
    (849.0, 2),  # 11  e1g  CH bend
    (3063.0, 2),  # 12  e1u  CH stretch
    (1486.0, 2),  # 13  e1u  ring stretch + deformation
    (1038.0, 2),  # 14  e1u  CH bend
    (3047.0, 2),  # 15  e2g  CH stretch
    (1596.0, 2),  # 16  e2g  ring stretch
    (1178.0, 2),  # 17  e2g  CH bend
    (606.0, 2),  # 18  e2g  ring deformation
    (975.0, 2),  # 19  e2u  CH bend
    (410.0, 2),  # 20  e2u  ring deformation
)


def einstein_function(x):
    """Heat capacity of one harmonic oscillator, in units of k_B, at x = Theta/T.

        f_E(x) = x^2 e^x / (e^x - 1)^2 = (x/2)^2 / sinh^2(x/2)

    The sinh form is used because the first subtracts 1 from a large
    exponential. f_E is a smooth switch turning the mode's classical k_B on as
    T rises past its characteristic temperature Theta = h c nu / k_B:

        x -> 0  (T >> Theta):  f_E -> 1, equipartition (1/2 k_B kinetic +
                               1/2 k_B potential)
        x -> inf (T << Theta): f_E -> x^2 e^-x, exponentially frozen out

    with the crossover at x ~ 1. Accepts a scalar or any array shape.

    Note this is the Einstein *function*, which is exact for an independent
    harmonic oscillator -- not the Einstein *model* of a solid, whose single
    shared frequency famously misses the T^3 law that acoustic branches give.
    Molecular vibrations really are independent oscillators at fixed measured
    frequencies, so no Einstein-model approximation is involved here.
    """
    x = np.asarray(x, dtype=float)
    # Clipping keeps sinh finite at both ends; f_E underflows to 0 long before
    # x = 700, and is 1 to ~1e-16 below x = 1e-8, so neither bound is physical.
    xc = np.clip(x, 1e-8, 700.0)
    f = (xc / (2.0 * np.sinh(xc / 2.0))) ** 2
    return np.where(x < 1e-8, 1.0, f)


def vibrational_heat_capacity(temperature, fundamentals=BENZENE_FUNDAMENTALS):
    """Intramolecular vibrational heat capacity per molecule, in eV/K.

        C_vib(T) = k_B sum_j g_j f_E(Theta_j / T)

    over the distinct fundamentals j with degeneracies g_j. The modes are
    independent, so ln q_vib is a sum over modes and every extensive quantity
    decomposes mode by mode; the zero-point term is T-independent and so
    contributes to U and H but exactly nothing here.

    This is the piece a rigid-body MC trajectory cannot contain. The sampler
    gives each molecule 6 degrees of freedom (3 translational + 3 rotational),
    which ``HeatCapacity`` already accounts for via its 3 k_B kinetic term plus
    the configurational fluctuation. Real benzene has 3 x 12 = 36, so what is
    missing is exactly the 30 intramolecular vibrations -- and only those. Do
    not add a q_rot or q_trans on top; the MC samples both.

    The additivity relies on the intramolecular Hamiltonian being separable
    from the intermolecular one and independent of V. That independence is what
    lets q_vib factor out of the NPT volume integral, so this same term adds to
    C_p in NPT exactly as it adds to C_v in NVT (no P V work is done by an
    intramolecular mode). It holds well for benzene's stiff ring and CH modes;
    it is the lattice modes, which the MC does sample, that carry the large
    volume dependence.

    ``temperature`` may be a scalar or an array (the result matches its shape).
    Multiply by ``EV_PER_K_TO_J_PER_MOL_K`` to compare against calorimetry.
    """
    temp = np.asarray(temperature, dtype=float)
    nu = np.array([f[0] for f in fundamentals], dtype=float)
    degeneracy = np.array([f[1] for f in fundamentals], dtype=float)
    theta = HC_OVER_K * nu
    x = theta[:, None] / temp.reshape(1, -1)
    c = BOLTZCONST * (degeneracy[:, None] * einstein_function(x)).sum(axis=0)
    return c.reshape(temp.shape)


class HeatCapacity(Measurement):
    """Per-particle heat capacity from energy/enthalpy fluctuations.

    ``pressure=None`` (the default) gives the constant-volume C_v from the
    canonical fluctuation of the potential energy U,

        C_v = 3 k_B + Var(U) / (N k_B T^2),

    which is the right quantity for an NVT (fixed-box) trajectory. Passing a
    ``pressure`` switches to the isobaric C_p, built from the instantaneous
    enthalpy H = U + P V with V read per frame, so volume fluctuations and the
    U-V covariance are both included,

        C_p = 3 k_B + Var(U + P V) / (N k_B T^2),

    the right quantity for an NPT trajectory (P in eV/A^3, V in A^3, matching
    the codebase's eV/A units).

    The leading 3 k_B is the equipartition (kinetic) term added by hand -- MC
    samples no momenta -- assuming 6 quadratic momentum DOF per particle (3
    translational + 3 rotational, i.e. a rigid 3-D rotor). It is the same
    constant in both ensembles (the ideal-gas C_p - C_v = k_B shift is not
    applied) and only sets the baseline, not the transition peak.

    Those 6 DOF are all a rigid-body sampler has. Passing ``vibrational_modes``
    (e.g. ``BENZENE_FUNDAMENTALS``) adds the analytic intramolecular term
    ``vibrational_heat_capacity`` on top, which is what any comparison against
    a real calorimetry curve needs. It is left off by default because it is a
    property of the molecule, not of the trajectory, and because it is only
    half the quantum correction: the librational lattice modes the MC *does*
    sample are treated classically, which biases the result the other way at
    low T (h c nu / k_B T ~ 1.4 for a 100 cm^-1 mode at 100 K).
    """

    def __init__(
        self, temperature, num_particles, pressure=None, vibrational_modes=None
    ):
        super().__init__()
        self.enthalpies = []
        self.temp = temperature
        self.num_particles = num_particles
        self.pressure = pressure
        self.vibrational_modes = vibrational_modes

    def compute(self, frame, scalar_data, array_data):
        h = scalar_data["total_energy"]
        if self.pressure is not None:
            # NPT: instantaneous enthalpy U + P V (the box floats per frame)
            h += self.pressure * frame.get_volume()
        self.enthalpies.append(h)

    def finalize(self):
        var = np.var(self.enthalpies)
        cv_ideal = 3 * BOLTZCONST
        cv_excess = var / (BOLTZCONST * self.num_particles * self.temp**2)
        total = cv_ideal + cv_excess
        if self.vibrational_modes is not None:
            total += float(vibrational_heat_capacity(self.temp, self.vibrational_modes))
        return total


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau of a 1-D series.

        tau = 1 + 2 * sum_{k>=1} rho_k

    where rho_k is the normalised autocorrelation at lag k. The autocovariance
    is computed via FFT (O(M log M)); the sum is truncated at the first
    non-positive rho_k (Geyer's "initial positive sequence"), which stops the
    noisy long-lag tail from accumulating. tau >= 1 always; tau = 1 for an
    uncorrelated series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if n < 2 or np.dot(x, x) == 0.0:
        return 1.0
    # autocovariance via the Wiener-Khinchin theorem (zero-pad to avoid wrap)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf /= acf[0]  # normalise so rho_0 = 1
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        tau += 2.0 * acf[k]
    return tau


class EffectiveSampleSize(Measurement):
    """Effective number of independent samples for a per-frame scalar.

    MCMC frames are serially correlated, so M stored frames are worth fewer than
    M independent draws. With the integrated autocorrelation time tau (see
    ``integrated_autocorr_time``),

        ESS = M / tau

    ``observable`` selects the scalar tracked each frame: pass a string to read
    ``scalar_data[observable]`` (e.g. ``"total_energy"``, ``"vol"``), or a
    callable ``(frame, scalar_data, array_data) -> float`` for derived
    quantities (e.g. the per-frame nematic S). finalize() returns a dict:

        ess          M / tau, the effective sample size
        tau          integrated autocorrelation time
        num_samples  M, the number of frames seen
        mean, std    sample mean and standard deviation of the observable
        sem          standard error of the mean, std / sqrt(ess); a 95% CI on
                     ``mean`` is mean +/- 1.96 * sem

    SEM uses ESS. Valid for a *stationary* series and
    for the mean itself; for nonlinear observables (heat capacity, ratios) block
    bootstrap/jackknife the trajectory instead.
    """

    def __init__(self, observable):
        super().__init__()
        self.observable = observable
        self.values = []

    def compute(self, frame, scalar_data, array_data):
        if callable(self.observable):
            v = self.observable(frame, scalar_data, array_data)
        else:
            v = scalar_data[self.observable]
        self.values.append(float(v))

    def finalize(self):
        x = np.asarray(self.values, dtype=float)
        m = len(x)
        tau = integrated_autocorr_time(x)
        ess = m / tau
        std = float(x.std()) if m else float("nan")
        return {
            "ess": ess,
            "tau": tau,
            "num_samples": m,
            "mean": float(x.mean()) if m else float("nan"),
            "std": std,
            "sem": std / np.sqrt(ess) if m else float("nan"),
        }


class TrajectoryAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.measurements = {}

    def add_measurement(self, name, measurement_obj):
        self.measurements[name] = measurement_obj

    def run_analysis(self, progress=True):
        with connect(self.db_path) as db:
            total_frames = db.count()

            with tqdm(
                total=total_frames, desc="Analyzing Trajectory", disable=not progress
            ) as pbar:
                for row in db.select():
                    frame = row.toatoms()
                    scalar_data = row.key_value_pairs
                    array_data = row.data

                    for meas in self.measurements.values():
                        meas.compute(frame, scalar_data, array_data)

                    pbar.update(1)

            final_results = {}
            for name, meas in self.measurements.items():
                final_results[name] = meas.finalize()

        return final_results
