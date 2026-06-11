import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random

import numpy as np
import pytest

from nvt_scan import tstar_to_kelvin, nematic_order_parameter, run_state_point
from metropolis import MetropolisCalculator, BOLTZCONST
from potentials import GB_PARAMS
from initialize import generate_random_config


# ---------------------------------------------------------------------------
# Reduced-unit conversion
# ---------------------------------------------------------------------------

def test_tstar_to_kelvin_roundtrip():
    eps0 = GB_PARAMS["eps0"]
    # T* = 1  <=>  kT = eps0
    assert np.isclose(tstar_to_kelvin(1.0), eps0 / BOLTZCONST)
    # kB * T[K] must recover T* * eps0
    for t_star in (0.2, 0.5, 1.3):
        assert np.isclose(BOLTZCONST * tstar_to_kelvin(t_star), t_star * eps0)


# ---------------------------------------------------------------------------
# Nematic order parameter
# ---------------------------------------------------------------------------

def test_nematic_perfectly_aligned_is_one():
    u = np.tile([0.0, 0.0, 1.0], (50, 1))
    assert np.isclose(nematic_order_parameter(u), 1.0)


def test_nematic_isotropic_is_small():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4000, 3))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert 0.0 <= nematic_order_parameter(u) < 0.15  # finite-size residual only


def test_nematic_stays_in_valid_range():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert -0.5 <= nematic_order_parameter(u) <= 1.0


# ---------------------------------------------------------------------------
# NVT mode of the sampler
# ---------------------------------------------------------------------------

def test_nvt_holds_box_fixed(tmp_path):
    frame = generate_random_config(n_particles=27, density=0.3, seed=2)
    metro = MetropolisCalculator(
        temp=300, pressure=0.0, init_frame=frame,
        volume_moves=False, output_dir=str(tmp_path / "s"),
    )
    v0 = np.linalg.det(metro.current_frame.cell)
    for _ in range(400):
        metro.step()
    v1 = np.linalg.det(metro.current_frame.cell)
    assert np.isclose(v0, v1)              # box unchanged
    assert len(metro.vol_decisions) == 0   # no volume moves attempted


def test_npt_default_still_attempts_volume_moves(tmp_path):
    frame = generate_random_config(n_particles=27, density=0.3, seed=3)
    metro = MetropolisCalculator(
        temp=300, pressure=0.0, init_frame=frame,
        output_dir=str(tmp_path / "s"),
    )
    assert metro.volume_moves is True
    for _ in range(400):
        metro.step()
    assert len(metro.vol_decisions) > 0    # NPT default is unchanged


def test_nvt_equilibrate_tunes_deltas_and_holds_box(tmp_path):
    """equilibrate() in NVT mode adapts step sizes and keeps the box fixed."""
    frame = generate_random_config(n_particles=64, density=0.3, seed=5)
    metro = MetropolisCalculator(
        temp=300, pressure=0.0, init_frame=frame,
        volume_moves=False, output_dir=str(tmp_path / "s"),
    )
    v0 = np.linalg.det(metro.current_frame.cell)
    assert metro.pos_delt == 0.15  # timid default
    # runs block_update with dynamic_delta in NVT (no vol-decisions nan crash)
    metro.equilibrate(20 * 64, block_size=4 * 64, dynamic_delta=True,
                      buffer_size=10_000, progress=False)
    v1 = np.linalg.det(metro.current_frame.cell)
    assert np.isclose(v0, v1)              # NVT: box held fixed
    assert len(metro.vol_decisions) == 0   # no volume moves attempted
    assert metro.pos_delt > 0.15           # adaptive deltas grew the timid default


def test_larger_max_scale_converges_faster(tmp_path):
    """A bigger max_scale tunes the over-accepting moves faster per block."""
    def equil_pos_delt(max_scale):
        random.seed(1)
        np.random.seed(1)
        frame = generate_random_config(n_particles=64, density=0.3, seed=5)
        metro = MetropolisCalculator(
            temp=300, pressure=0.0, init_frame=frame,
            volume_moves=False, output_dir=str(tmp_path / f"s{max_scale}"),
        )
        metro.equilibrate(20 * 64, block_size=4 * 64, dynamic_delta=True,
                          buffer_size=10_000, progress=False,
                          max_scale=max_scale, min_scale=1.0 / max_scale)
        return metro.pos_delt

    gentle = equil_pos_delt(1.1)
    aggressive = equil_pos_delt(2.0)
    assert aggressive > gentle


# ---------------------------------------------------------------------------
# run_state_point end-to-end smoke
# ---------------------------------------------------------------------------

def test_run_state_point_returns_sane_observables(tmp_path):
    res = run_state_point(
        t_star=0.8, rho_star=0.3, n_particles=27,
        n_equil_sweeps=2, n_prod_sweeps=3, sample_every=1,
        seed=4, scratch_dir=str(tmp_path / "scan"),
    )
    assert set(res) >= {"T_star", "rho_star", "E_star_per_N",
                        "Cv_over_kB", "S", "pos_acc", "or_acc"}
    assert res["T_star"] == 0.8 and res["rho_star"] == 0.3
    assert np.isfinite(res["E_star_per_N"])
    assert res["Cv_over_kB"] >= 0.0          # variance is non-negative
    assert 0.0 <= res["S"] <= 1.0
    assert 0.0 <= res["pos_acc"] <= 1.0
    assert 0.0 <= res["or_acc"] <= 1.0
