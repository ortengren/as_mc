import numpy as np
import pytest

from asmcmc.potentials import CACELLI_POTENTIAL, GBQPotential
from asmcmc.validation import (
    DimerBenchmark,
    dimer_benchmark,
    dimer_scan,
    load_cacelli_dimers,
)


# A frozen copy of the condensed-phase GB+Q refit (data/my_fitted_gbq_params.json
# as of 2026-07): fits the PBE-D3 crystal energies well (test RMSE ~3 kcal/mol)
# yet is repulsive at the cofacial stacking distance and anti-correlated with
# the ab initio dimer wells. Kept inline so the test still documents the
# failure mode this benchmark exists to catch even after the default fit is
# replaced by something better.
CONDENSED_REFIT = GBQPotential(
    name="condensed_phase_refit_2026_07",
    sigma0=7.070314502548505,
    eps0=0.007880572307778944,
    kappa=0.5782634293212777,
    kappa_prime=0.4206023022125707,
    mu=-1.682824932973503,
    nu=3.972392556925173,
    xi=1.0,
    Q=-3.59172701770117,
)


@pytest.fixture(scope="module")
def data():
    return load_cacelli_dimers()


@pytest.fixture(scope="module")
def cacelli_bench(data):
    return dimer_benchmark(CACELLI_POTENTIAL, data)


@pytest.fixture(scope="module")
def refit_bench(data):
    return dimer_benchmark(CONDENSED_REFIT, data)


# --- loader & geometry construction ---

def test_load_shapes_and_units(data):
    assert len(data) == 197
    for arr in (data.uhat1, data.uhat2, data.r, data.euler_deg):
        assert len(arr) == 197
    # normals are unit vectors
    assert np.allclose(np.linalg.norm(data.uhat1, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(data.uhat2, axis=1), 1.0)
    # molecule A's ring lies in the xz-plane -> normal +y
    assert np.allclose(data.uhat1, [0.0, 1.0, 0.0])
    # interaction energies span the wall and the wells (kcal/mol)
    assert data.energy_kcal.min() == pytest.approx(-2.60, abs=0.05)
    assert data.energy_kcal.max() > 20.0


def test_angle_zero_rows_keep_parallel_normals(data):
    ang0 = np.all(data.euler_deg == 0.0, axis=1)
    assert ang0.sum() == 144
    assert np.allclose(data.uhat2[ang0], [0.0, 1.0, 0.0])


def test_euler_convention_maps_t_family_to_z(data):
    """(beta=90, gamma=90) is the T-shaped family: A-normal y -> B-normal z."""
    t = (
        (data.euler_deg[:, 0] == 0.0)
        & (data.euler_deg[:, 1] == 90.0)
        & (data.euler_deg[:, 2] == 90.0)
    )
    assert t.sum() > 0
    assert np.allclose(np.abs(data.uhat2[t] @ [0.0, 0.0, 1.0]), 1.0, atol=1e-12)


def test_dimer_scan_matches_pair_energy(data):
    """dimer_scan along the cofacial ray reproduces row-wise pair_energy."""
    d = np.array([3.9, 5.0])
    curve = dimer_scan(CACELLI_POTENTIAL, [0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0], d)
    direct = CACELLI_POTENTIAL.pair_energy(
        np.tile([0, 1.0, 0], (2, 1)),
        np.tile([0, 1.0, 0], (2, 1)),
        d[:, None] * np.array([0, 1.0, 0]),
    ) * 23.060541945329334
    assert np.allclose(curve, direct)


# --- the benchmark: Cacelli (fit to this data) must pass ---

def test_cacelli_reproduces_its_training_wells(cacelli_bench):
    b = cacelli_bench
    assert b.well_pearson_r > 0.95
    assert b.well_rmse_kcal < 0.3
    assert b.full_pearson_r > 0.85
    assert b.stacking_bound
    # cofacial well: ab initio -1.72 @ 3.9 A; model within 0.5 kcal/mol & 0.5 A
    cof = b.wells["cofacial"]
    assert cof.model_depth == pytest.approx(cof.ab_depth, abs=0.5)
    assert cof.model_r == pytest.approx(cof.ab_r, abs=0.5)
    # slipped-parallel and T-shaped wells present and comparably deep
    for fam in ("parallel_displaced", "t_shaped"):
        w = b.wells[fam]
        assert w.model_depth == pytest.approx(w.ab_depth, abs=0.5)
        assert w.model_at_ab_min < -1.0


def test_family_minima_anchor_correctly(cacelli_bench):
    """The ab initio family minima are the known Cacelli 2004 values."""
    wells = cacelli_bench.wells
    assert wells["cofacial"].ab_depth == pytest.approx(-1.722, abs=0.01)
    assert wells["cofacial"].ab_r == pytest.approx(3.9, abs=0.01)
    # PD holds the dataset's global minimum: the slipped-parallel row
    # (0, 3.5, 1.6), i.e. stack height 3.5 A with 1.6 A lateral slip
    assert wells["parallel_displaced"].ab_depth == pytest.approx(-2.600, abs=0.01)
    assert wells["parallel_displaced"].ab_r == pytest.approx(3.85, abs=0.01)
    assert wells["t_shaped"].ab_depth == pytest.approx(-2.280, abs=0.01)
    assert wells["t_shaped"].ab_r == pytest.approx(5.0, abs=0.01)


# --- the benchmark: the condensed-phase refit must fail ---

def test_condensed_refit_fails_benchmark(refit_bench):
    b = refit_bench
    # repulsive where real benzene stacks: the single fatal check
    assert not b.stacking_bound
    assert b.stacking_energy_kcal > 1.0
    # anti-correlated with the true wells despite its good crystal parity
    assert b.well_pearson_r < 0.5
    # no well anywhere near the ab initio depth
    deepest = min(w.model_depth for w in b.wells.values())
    assert deepest > -1.0


def test_benchmark_discriminates(cacelli_bench, refit_bench):
    """The benchmark orders the two reference potentials correctly."""
    assert cacelli_bench.well_rmse_kcal < refit_bench.well_rmse_kcal / 5
    assert cacelli_bench.well_pearson_r > refit_bench.well_pearson_r + 0.5


# --- reporting ---

def test_summary_is_printable(cacelli_bench):
    s = cacelli_bench.summary()
    assert isinstance(cacelli_bench, DimerBenchmark)
    for token in ("cofacial", "parallel_displaced", "t_shaped", "RMSE", "bound"):
        assert token in s