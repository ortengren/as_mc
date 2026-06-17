import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ase
import pytest
from ase.db import connect

from asmcmc.metropolis import MetropolisCalculator
from asmcmc.potentials import calc_total_energy
from asmcmc.measurements import TrajectoryAnalyzer, AverageEnergy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def four_particle_frame():
    """2×2 grid of particles 10 Å apart, cubic 30 Å box, all oriented along z."""
    positions = np.array([
        [ 0.,  0., 0.],
        [10.,  0., 0.],
        [ 0., 10., 0.],
        [10., 10., 0.],
    ])
    cell = np.diag([30., 30., 30.])
    frame = ase.Atoms(symbols="HHHH", positions=positions, cell=cell, pbc=True)
    frame.new_array("c_q",    np.tile([1., 0., 0., 0.], (4, 1)))
    frame.new_array("or_vec", np.tile([0., 0., 1.],     (4, 1)))
    return frame


def make_metro(frame, tmp_path, **kwargs):
    return MetropolisCalculator(
        temp=300,
        pressure=0.0,
        init_frame=frame,
        output_dir=str(tmp_path / "sim"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simulation_runs_to_completion(four_particle_frame, tmp_path):
    """calculate_trajectory exits cleanly and step_count equals num_steps."""
    metro = make_metro(four_particle_frame, tmp_path)
    num_steps = 200
    metro.calculate_trajectory(
        num_steps=num_steps,
        block_size=50,
        num_eq_steps=100,
        buffer_size=50,
    )
    assert metro.step_count == num_steps


def test_calculate_trajectory_clears_equilibration_decisions(four_particle_frame, tmp_path):
    """Acceptance decisions after a run cover production only, not equilibration."""
    metro = make_metro(four_particle_frame, tmp_path)
    num_steps = 200
    metro.calculate_trajectory(
        num_steps=num_steps,
        block_size=50,
        num_eq_steps=100,
        buffer_size=50,
    )
    # step() appends exactly one decision per step; equilibration's were dropped,
    # so the three lists together hold exactly the production steps.
    total = (
        len(metro.pos_decisions)
        + len(metro.or_decisions)
        + len(metro.vol_decisions)
    )
    assert total == num_steps


def test_simulation_db_created(four_particle_frame, tmp_path):
    """simulation.db is created and contains one row per block."""
    block_size = 50
    num_steps = 200
    metro = make_metro(four_particle_frame, tmp_path)
    metro.calculate_trajectory(
        num_steps=num_steps,
        block_size=block_size,
        num_eq_steps=0,
        buffer_size=1,  # flush every block so counts are exact
    )
    db_path = str(tmp_path / "sim" / "simulation.db")
    assert os.path.exists(db_path), "simulation.db was not created"
    with connect(db_path) as db:
        rows = list(db.select())
    assert len(rows) == num_steps // block_size
    # all recorded energies must be finite
    for row in rows:
        assert np.isfinite(row.key_value_pairs["total_energy"])


def test_equilibration_marks_equilibrated_and_keeps_step_count(four_particle_frame, tmp_path):
    """After equilibrate(): equilibrated=True, step_count == num_steps (re-entrant), db written."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=300, block_size=100, buffer_size=50)

    assert metro.equilibrated is True
    assert metro.step_count == 300

    db_path = str(tmp_path / "sim" / "equilibration.db")
    assert os.path.exists(db_path), "equilibration.db was not created"
    with connect(db_path) as db:
        assert db.count() > 0


def test_energy_tracking_consistent(four_particle_frame, tmp_path):
    """Incremental energy tracking must stay consistent with full recalculation."""
    metro = make_metro(four_particle_frame, tmp_path)
    for _ in range(500):
        metro.step()

    recomputed = calc_total_energy(
        metro.current_frame, metro.nl_cutoffs, potential=metro.potential
    )
    np.testing.assert_allclose(
        metro.current_energy, recomputed, rtol=1e-4,
        err_msg="Incremental energy tracker drifted from calc_total_energy",
    )


def test_volume_tracking_consistent(four_particle_frame, tmp_path):
    """self.current_vol must equal det(cell) after accepted volume moves."""
    metro = make_metro(four_particle_frame, tmp_path)
    for _ in range(2000):
        metro.step()

    if not any(metro.vol_decisions):
        pytest.skip("No volume moves were accepted in this run; cannot test vol tracking")

    actual_vol = np.linalg.det(metro.current_frame.get_cell())
    np.testing.assert_allclose(
        metro.current_vol, actual_vol, rtol=1e-6,
        err_msg="self.current_vol is stale — not updated after accepted volume moves",
    )


def test_trajectory_analyzer_on_output(four_particle_frame, tmp_path):
    """TrajectoryAnalyzer returns a finite AverageEnergy result."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.calculate_trajectory(
        num_steps=200,
        block_size=50,
        num_eq_steps=0,
        buffer_size=1,
    )

    db_path = str(tmp_path / "sim" / "simulation.db")
    analyzer = TrajectoryAnalyzer(db_path)
    analyzer.add_measurement("energy", AverageEnergy())
    results = analyzer.run_analysis()

    mean_e, var_e = results["energy"]
    assert np.isfinite(mean_e), f"Mean energy is not finite: {mean_e}"
    assert np.isfinite(var_e),  f"Energy variance is not finite: {var_e}"


# ---------------------------------------------------------------------------
# Resume from equilibration (RunConfig + from_equilibration)
# ---------------------------------------------------------------------------

def test_from_equilibration_restores_state(four_particle_frame, tmp_path):
    """from_equilibration rebuilds the static definition + the last db frame/deltas/step."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=200, block_size=50, buffer_size=10)

    out = str(tmp_path / "sim")
    db = connect(out + "/equilibration.db")
    row = db.get(db.count())

    resumed = MetropolisCalculator.from_equilibration(out)

    # static run definition (from run_config.json)
    assert resumed.temp == metro.temp
    assert resumed.pressure == metro.pressure
    assert resumed.npt_ensemble == metro.npt_ensemble
    assert resumed.nl_radius == metro.nl_radius
    assert resumed.nl_skin == metro.nl_skin
    assert resumed.potential == metro.potential

    # evolving state (from the last db row)
    np.testing.assert_allclose(resumed.current_frame.positions, row.toatoms().positions)
    np.testing.assert_allclose(resumed.current_frame.arrays["c_q"], np.asarray(row.data["c_q"]))
    np.testing.assert_allclose(
        resumed.current_frame.arrays["or_vec"], np.asarray(row.data["or_vec"])
    )
    assert (resumed.pos_delt, resumed.or_delt, resumed.vol_delt) == (
        row.pos_delta, row.or_delta, row.vol_delta
    )
    assert resumed.step_count == row.step


def test_from_equilibration_appends_and_continues(four_particle_frame, tmp_path):
    """Continuing a resumed run appends to the same db with a monotonic step axis."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=200, block_size=50, buffer_size=10)

    db_path = str(tmp_path / "sim" / "equilibration.db")
    db = connect(db_path)
    before = db.count()
    last_step = db.get(before).step

    resumed = MetropolisCalculator.from_equilibration(str(tmp_path / "sim"))
    resumed.equilibrate(num_steps=last_step + 200, block_size=50, buffer_size=10)

    steps = [r.step for r in connect(db_path).select()]
    assert len(steps) > before            # rows were appended, not replaced
    assert max(steps) == last_step + 200  # continued past the original target
    assert steps == sorted(steps)         # continuous, monotonic step axis


def test_from_equilibration_preserves_run_config(four_particle_frame, tmp_path):
    """Resuming must not clobber the original run's run_config.json (write-once)."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=100, block_size=50, buffer_size=10)
    cfg_path = tmp_path / "sim" / "run_config.json"
    original = cfg_path.read_text()

    resumed = MetropolisCalculator.from_equilibration(str(tmp_path / "sim"))
    resumed.equilibrate(num_steps=300, block_size=50, buffer_size=10)

    assert cfg_path.read_text() == original


def test_equilibrate_is_reentrant(four_particle_frame, tmp_path):
    """A bare equilibrate leaves step_count == num_steps and continues on re-call."""
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=100, block_size=50, buffer_size=10)
    assert metro.step_count == 100
    metro.equilibrate(num_steps=200, block_size=50, buffer_size=10)  # 100 more, not 200
    assert metro.step_count == 200
