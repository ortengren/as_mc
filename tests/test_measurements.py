import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ase
import pytest
from measurements import (
    AverageEnergy,
    RadialDistributionFunction,
    OrientationalCorrelationFunction,
    NematicOrderParameter,
    nematic_q_tensor,
    HeatCapacity,
    BOLTZCONST,
)
from metropolis import npt_decide_accept
from potentials import calc_total_energy

# --- AverageEnergy ---


def test_average_energy_known_values():
    m = AverageEnergy()
    for e in [1.0, 2.0, 3.0]:
        m.compute(None, {"total_energy": e}, None)
    mean, var = m.finalize()
    np.testing.assert_allclose(mean, 2.0, atol=1e-12)
    np.testing.assert_allclose(var, 2.0 / 3.0, atol=1e-12)


def test_average_energy_recompute_matches_direct():
    """recompute=True reads the frame, not the (here bogus) tracked scalar, and
    reproduces calc_total_energy exactly; identical frames give zero variance."""
    frame = _make_uniform_frame(n=8, box=20.0)
    nl_radius = 15.0
    expected = calc_total_energy(frame, [nl_radius] * len(frame), "GB")
    m = AverageEnergy(recompute=True, nl_radius=nl_radius)
    for _ in range(3):
        m.compute(frame, {"total_energy": 999.0}, None)  # bogus scalar must be ignored
    mean, var = m.finalize()
    np.testing.assert_allclose(mean, expected, rtol=1e-12)
    np.testing.assert_allclose(var, 0.0, atol=1e-12)


def test_average_energy_recompute_requires_nl_radius():
    """recompute=True without a neighbour-list radius is a usage error."""
    with pytest.raises(ValueError):
        AverageEnergy(recompute=True)


def test_average_energy_recompute_restores_or_vec_from_array_data():
    """After a db round-trip the frame has lost or_vec; recompute restores it
    from array_data and still reproduces calc_total_energy."""
    frame = _make_uniform_frame(n=8, box=20.0)
    or_vec = frame.arrays["or_vec"].copy()
    expected = calc_total_energy(frame, [15.0] * len(frame), "GB")
    bare = frame.copy()
    bare.set_array("or_vec", None)  # mimic row.toatoms(), which drops custom arrays
    m = AverageEnergy(recompute=True, nl_radius=15.0)
    m.compute(bare, {"total_energy": 999.0}, {"or_vec": or_vec})
    mean, _ = m.finalize()
    np.testing.assert_allclose(mean, expected, rtol=1e-12)


# --- RadialDistributionFunction ---


def test_rdf_bin_centers():
    """Bin centres should span [dr/2, r_max - dr/2]."""
    r_max, n_bins = 20.0, 10
    m = RadialDistributionFunction(r_max, n_bins)
    dr = r_max / n_bins
    np.testing.assert_allclose(m.bin_centers[0], dr / 2, atol=1e-12)
    np.testing.assert_allclose(m.bin_centers[-1], r_max - dr / 2, atol=1e-12)


def _make_uniform_frame(n=64, box=40.0):
    """Return an ASE Atoms with particles on a regular grid inside a cubic box."""
    side = int(round(n ** (1 / 3)))
    assert side**3 == n, "n must be a perfect cube"
    spacing = box / side
    positions = np.array(
        [
            [i * spacing, j * spacing, k * spacing]
            for i in range(side)
            for j in range(side)
            for k in range(side)
        ]
    )
    frame = ase.Atoms(
        symbols="H" * n,
        positions=positions,
        cell=np.diag([box, box, box]),
        pbc=True,
    )
    frame.new_array("c_q", np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))
    frame.new_array("or_vec", np.tile([0.0, 0.0, 1.0], (n, 1)))
    return frame


def test_rdf_normalization_ideal_gas():
    """For random (ideal-gas-like) positions, mean g(r) in the bulk should be ≈ 1."""
    rng = np.random.default_rng(42)
    box, n = 40.0, 64
    m = RadialDistributionFunction(r_max=15.0, num_bins=30)
    for _ in range(20):
        positions = rng.uniform(0, box, size=(n, 3))
        frame = ase.Atoms(
            symbols="H" * n,
            positions=positions,
            cell=np.diag([box, box, box]),
            pbc=True,
        )
        frame.new_array("c_q", np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))
        frame.new_array("or_vec", np.tile([0.0, 0.0, 1.0], (n, 1)))
        m.compute(frame, {}, {})
    result = m.finalize()
    r, g_r = result["r"], result["g_r"]
    bulk = g_r[(r > 4) & (r < 14)]
    assert len(bulk) > 0
    np.testing.assert_allclose(np.mean(bulk), 1.0, atol=0.2)


# --- OrientationalCorrelationFunction ---


def test_ocf_parallel_particles():
    """All particles aligned → P2(cos θ=1) = 1 in every populated bin."""
    frame = _make_uniform_frame(n=8, box=20.0)
    # all or_vecs already point along z
    m = OrientationalCorrelationFunction(r_max=15.0, num_bins=20)
    m.compute(frame, {}, {"or_vec": frame.arrays["or_vec"]})
    result = m.finalize()
    populated = result["s2_r"][result["s2_r"] != 0]
    np.testing.assert_allclose(populated, 1.0, atol=1e-10)


def test_ocf_antiparallel_still_gives_p2_one():
    """Anti-parallel orientation (cos θ = -1) also gives P2 = 1 since P2(x) = (3x²-1)/2."""
    frame = _make_uniform_frame(n=8, box=20.0)
    or_vecs = frame.arrays["or_vec"].copy()
    # flip every other particle
    or_vecs[1::2] = -or_vecs[1::2]
    m = OrientationalCorrelationFunction(r_max=15.0, num_bins=20)
    m.compute(frame, {}, {"or_vec": or_vecs})
    result = m.finalize()
    populated = result["s2_r"][result["s2_r"] != 0]
    np.testing.assert_allclose(populated, 1.0, atol=1e-10)


def test_ocf_perpendicular_particles():
    """Two particles with perpendicular orientations give P2 = -0.5."""
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cell = np.diag([60.0, 60.0, 60.0])
    frame = ase.Atoms(symbols="HH", positions=positions, cell=cell, pbc=True)
    frame.new_array("c_q", np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    or_vecs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # z ⊥ x
    frame.new_array("or_vec", or_vecs)

    m = OrientationalCorrelationFunction(r_max=15.0, num_bins=20)
    m.compute(frame, {}, {"or_vec": or_vecs})
    result = m.finalize()
    populated = result["s2_r"][result["s2_r"] != 0]
    np.testing.assert_allclose(populated, -0.5, atol=1e-10)


# --- nematic_q_tensor (helper) ---


def test_q_tensor_is_symmetric_and_traceless():
    """Q is symmetric and traceless for any set of unit axes."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=(37, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    q = nematic_q_tensor(u)
    np.testing.assert_allclose(q, q.T, atol=1e-12)
    np.testing.assert_allclose(np.trace(q), 0.0, atol=1e-12)


def test_q_tensor_aligned_gives_unit_eigenvalue():
    """All axes along z → Q = diag(-1/2, -1/2, 1), so the top eigenvalue (S) = 1."""
    u = np.tile([0.0, 0.0, 1.0], (20, 1))
    q = nematic_q_tensor(u)
    np.testing.assert_allclose(np.diag(q), [-0.5, -0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(q)[-1], 1.0, atol=1e-12)


def test_q_tensor_orthogonal_axes_vanish():
    """Equal numbers of x-, y-, z-axes are isotropic → Q = 0 exactly."""
    u = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    q = nematic_q_tensor(u)
    np.testing.assert_allclose(q, np.zeros((3, 3)), atol=1e-12)


def test_q_tensor_isotropic_residual_is_small():
    """Many random axes → only the finite-size residual order remains."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4000, 3))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    s = np.linalg.eigvalsh(nematic_q_tensor(u))[-1]
    assert 0.0 <= s < 0.15


def test_q_tensor_eigenvalue_in_valid_range():
    """S stays within its physical bounds for an arbitrary set of axes."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    s = np.linalg.eigvalsh(nematic_q_tensor(u))[-1]
    assert -0.5 <= s <= 1.0


# --- NematicOrderParameter ---


def test_nematic_aligned_gives_s_one():
    """Perfectly aligned axes over several frames → S = S_lab = 1, no spread."""
    u = np.tile([0.0, 0.0, 1.0], (16, 1))
    m = NematicOrderParameter()
    for _ in range(5):
        m.compute(None, None, {"or_vec": u})
    r = m.finalize()
    np.testing.assert_allclose(r["S"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["S_lab"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["S_std"], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(r["director"]), [0.0, 0.0, 1.0], atol=1e-12)


def test_nematic_sign_invariant():
    """S depends on u u^T, so flipping half the axes anti-parallel leaves S = 1."""
    u = np.tile([0.0, 0.0, 1.0], (16, 1))
    u[1::2] *= -1
    m = NematicOrderParameter()
    m.compute(None, None, {"or_vec": u})
    np.testing.assert_allclose(m.finalize()["S"], 1.0, atol=1e-12)


def test_nematic_isotropic_gives_zero():
    """Orthogonal x/y/z axes are isotropic → S = S_lab = 0."""
    u = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    m = NematicOrderParameter()
    for _ in range(4):
        m.compute(None, None, {"or_vec": u})
    r = m.finalize()
    np.testing.assert_allclose(r["S"], 0.0, atol=1e-12)
    np.testing.assert_allclose(r["S_lab"], 0.0, atol=1e-12)


def test_nematic_per_frame_differs_from_averaged_q():
    """Per-frame S (tracks the instantaneous director) differs from the
    eigenvalue of the frame-averaged <Q> when the director rotates.

    Frame 1: all axes along z (S=1).  Frame 2: all axes along x (S=1).
    => mean per-frame S = 1, but <Q> = (Q_z + Q_x)/2 = diag(1/4, -1/2, 1/4),
       whose largest eigenvalue S_lab = 1/4.
    """
    z = np.tile([0.0, 0.0, 1.0], (10, 1))
    x = np.tile([1.0, 0.0, 0.0], (10, 1))
    m = NematicOrderParameter()
    m.compute(None, None, {"or_vec": z})
    m.compute(None, None, {"or_vec": x})
    r = m.finalize()
    np.testing.assert_allclose(r["S"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["S_lab"], 0.25, atol=1e-12)
    np.testing.assert_allclose(np.diag(r["Q_mean"]), [0.25, -0.5, 0.25], atol=1e-12)


# --- HeatCapacity ---


def test_heat_capacity_zero_variance():
    """Constant energy → zero excess Cv → total Cv = 3 k_B."""
    T, N = 300.0, 10
    m = HeatCapacity(temperature=T, num_particles=N)
    for _ in range(50):
        m.compute(None, {"total_energy": 1.0}, None)
    cv = m.finalize()
    np.testing.assert_allclose(cv, 3 * BOLTZCONST, rtol=1e-10)


def test_heat_capacity_known_variance():
    """Known energy variance → Cv = 3 k_B + σ²/(k_B * N * T²)."""
    T, N = 300.0, 5
    energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = HeatCapacity(temperature=T, num_particles=N)
    for e in energies:
        m.compute(None, {"total_energy": e}, None)
    cv = m.finalize()
    sigma2 = np.var(energies)
    expected = 3 * BOLTZCONST + sigma2 / (BOLTZCONST * N * T**2)
    np.testing.assert_allclose(cv, expected, rtol=1e-10)


# --- decide_accept ---


def test_decide_accept_lower_energy_always_accepted():
    """A move that lowers energy should always be accepted."""
    beta, P, N, vol = 1.0, 0.0, 2, 1000.0
    for _ in range(200):
        assert npt_decide_accept(1.0, 0.5, vol, vol, beta, P, N)


def test_decide_accept_same_energy_always_accepted():
    """Same energy, same volume → Boltzmann factor = 1, always accepted."""
    beta, P, N, vol = 1.0, 0.0, 2, 1000.0
    for _ in range(200):
        assert npt_decide_accept(1.0, 1.0, vol, vol, beta, P, N)


def test_decide_accept_high_temp_mostly_accepted():
    """At very high temperature (low beta) most moves should be accepted."""
    beta = 1e-10  # effectively infinite T
    P, N, vol = 0.0, 2, 1000.0
    decisions = [
        npt_decide_accept(0.0, 100.0, vol, vol, beta, P, N) for _ in range(1000)
    ]
    assert np.mean(decisions) > 0.99


def test_decide_accept_no_exp_overflow():
    """Extremely favorable move (large negative ΔE) must not raise OverflowWarning."""
    import warnings

    beta, P, N, vol = 1.0, 0.0, 2, 1000.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn warnings into exceptions
        # delta_E = -1e308 would overflow exp; the fix short-circuits before exp
        npt_decide_accept(1e308, 0.0, vol, vol, beta, P, N)


def test_decide_accept_very_favorable_always_accepted():
    """A move that improves energy by a huge amount must always be accepted."""
    beta, P, N, vol = 1.0, 0.0, 2, 1000.0
    for _ in range(200):
        assert npt_decide_accept(1e300, 0.0, vol, vol, beta, P, N)


def test_decide_accept_negative_vol_rejected():
    """new_vol < 0 (from a bad volume move) must be rejected without raising."""
    beta, P, N = 1.0, 0.0, 2
    assert not npt_decide_accept(0.0, 0.0, 1000.0, -1.0, beta, P, N)


def test_decide_accept_zero_vol_rejected():
    """new_vol == 0 must be rejected without raising."""
    beta, P, N = 1.0, 0.0, 2
    assert not npt_decide_accept(0.0, 0.0, 1000.0, 0.0, beta, P, N)
