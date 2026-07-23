import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ase
import pytest
from ase.db import connect

from asmcmc.metropolis import (
    MetropolisCalculator,
    MIN_VOL_DELT,
    MAX_VOL_DELT,
    MAX_OR_DELT,
    BOLTZCONST,
    npt_decide_accept,
)
from asmcmc.potentials import calc_total_energy
from asmcmc.trial_moves import calculate_vol_move
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
    # atol, not just rtol: this 4-particle system's total energy is near zero
    # (~-0.01 eV), so a pure relative tolerance is ill-conditioned. Benign drift
    # (near-cutoff pairs differing between the skinned NL and a fresh rebuild,
    # growing with move width) stays below ~1e-5 eV over 500 steps; a real
    # bookkeeping bug (e.g. a missed current_energy update) shows up at >=1e-2.
    np.testing.assert_allclose(
        metro.current_energy, recomputed, rtol=1e-4, atol=1e-4,
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


def test_isotropic_vol_only_scales_box_uniformly(four_particle_frame, tmp_path):
    """aniso_vol=False keeps the box an exact uniform scaling of the start: the
    three axis lengths stay in a constant ratio no matter which volume moves are
    accepted (isotropic scaling can only change size, never shape)."""
    metro = make_metro(four_particle_frame, tmp_path, aniso_vol=False)
    orig = metro.current_frame.cell.lengths().copy()
    for _ in range(2000):
        metro.step()
    ratios = metro.current_frame.cell.lengths() / orig
    np.testing.assert_allclose(
        ratios, ratios[0], rtol=1e-10,
        err_msg="isotropic volume moves must scale all three axes by one factor",
    )


def test_anisotropic_vol_changes_box_shape(four_particle_frame, tmp_path):
    """aniso_vol=True (the default) lets the aspect ratio change: after at least one
    accepted single-axis volume move the three axis lengths are no longer in the
    starting ratio."""
    metro = make_metro(four_particle_frame, tmp_path)  # default: anisotropic
    assert metro.aniso_vol is True
    orig = metro.current_frame.cell.lengths().copy()
    for _ in range(3000):
        metro.step()
    if not any(metro.vol_decisions):
        pytest.skip("no volume moves accepted in this run; cannot observe a shape change")
    ratios = metro.current_frame.cell.lengths() / orig
    assert not np.allclose(ratios, ratios[0]), (
        "anisotropic volume moves should break the uniform box aspect ratio"
    )


def test_from_equilibration_restores_aniso_vol(four_particle_frame, tmp_path):
    """The volume-move geometry (recorded in run_config.json) survives a resume;
    the non-default value proves it is read back, not just re-defaulted."""
    metro = make_metro(four_particle_frame, tmp_path, aniso_vol=False)
    metro.equilibrate(num_steps=100, block_size=50, buffer_size=10)
    resumed = MetropolisCalculator.from_equilibration(str(tmp_path / "sim"))
    assert resumed.aniso_vol is False


def test_recorded_energy_matches_recompute(four_particle_frame, tmp_path):
    """Every recorded total_energy must equal a fresh recompute of its frame.

    block_update re-syncs current_energy to calc_total_energy, so the energy
    written to the db is exact and never carries accumulated incremental drift.
    This is what keeps a resumed run (which recomputes from scratch) from
    showing an energy discontinuity at the resume step.
    """
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=1000, block_size=50, buffer_size=1, progress=False)

    db_path = str(tmp_path / "sim" / "equilibration.db")
    with connect(db_path) as db:
        rows = list(db.select())
    assert len(rows) > 0

    for row in rows:
        frame = row.toatoms()
        frame.new_array("c_q", np.asarray(row.data["c_q"]))
        frame.new_array("or_vec", np.asarray(row.data["or_vec"]))
        recomputed = calc_total_energy(
            frame, [metro.nl_radius] * len(frame), potential=metro.potential
        )
        np.testing.assert_allclose(
            row.total_energy, recomputed, rtol=1e-6, atol=1e-9,
            err_msg=f"recorded energy at step {row.step} drifted from recompute",
        )


def test_resume_has_no_energy_jump(four_particle_frame, tmp_path):
    """The last energy before a resume matches the first one after it.

    With the per-block resync, from_equilibration's fresh recompute agrees with
    the stored value, so the trajectory is continuous across the resume.
    """
    metro = make_metro(four_particle_frame, tmp_path)
    metro.equilibrate(num_steps=500, block_size=50, buffer_size=1, progress=False)
    last_step = metro.step_count

    resumed = MetropolisCalculator.from_equilibration(str(tmp_path / "sim"))
    np.testing.assert_allclose(
        resumed.current_energy, metro.current_energy, rtol=1e-6, atol=1e-9,
        err_msg="energy jumped across resume — incremental drift was not re-synced",
    )
    assert resumed.step_count == last_step


def test_vol_delt_floored_on_rejected_volume_moves(four_particle_frame, tmp_path):
    """A run of rejected volume moves must not shrink vol_delt below the floor."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=MIN_VOL_DELT)
    # all moves rejected ⇒ acc rate 0 ⇒ dynamic_delta tries to shrink vol_delt
    window = 50
    metro.pos_decisions = [0] * (window + 1)
    metro.or_decisions = [0] * (window + 1)

    db_file = str(tmp_path / "sim" / "equilibration.db")
    for _ in range(20):
        # feed a fresh window of rejected volume moves each block so the gated
        # tuner fires and repeatedly tries to shrink vol_delt
        metro.vol_decisions += [0] * window
        metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)

    assert metro.vol_delt >= MIN_VOL_DELT


# ---------------------------------------------------------------------------
# Volume-move (delta) tuning: log-uniform proposal + gated adaptation
# ---------------------------------------------------------------------------

def test_acceptance_decreases_with_vol_delta(four_particle_frame, tmp_path):
    """The log-uniform proposal paired with the (N+1) criterion gives a
    monotonically decreasing acceptance(delta) — i.e. a unique well-defined
    optimum for the tuner to converge to. (The old uniform-in-V proposal made
    this flat/rising, so the tuner had no fixed point.)"""
    metro = make_metro(four_particle_frame, tmp_path)
    frame = metro.current_frame
    nl, pot = metro.nl_cutoffs, metro.potential
    beta = 1.0 / (BOLTZCONST * 300)
    n = len(frame)
    old_vol = frame.get_volume()
    old_en = calc_total_energy(frame, nl, potential=pot)

    def mean_acceptance(delta, trials=400):
        accepts = 0
        for _ in range(trials):
            new_cell, new_vol = calculate_vol_move(frame.get_cell(), old_vol, delta)
            cand = frame.copy()
            cand.set_cell(new_cell, scale_atoms=True)
            new_en = calc_total_energy(cand, nl, potential=pot)
            if npt_decide_accept(old_en, new_en, old_vol, new_vol, beta, 0.0, n):
                accepts += 1
        return accepts / trials

    acc_small = mean_acceptance(0.02)
    acc_large = mean_acceptance(0.8)
    assert 0.0 < acc_large < acc_small <= 1.0


def test_vol_delt_tuning_is_gated_on_fresh_window(four_particle_frame, tmp_path):
    """vol_delt updates only once `window` fresh volume moves accrue, on a
    non-overlapping window, and not at all without fresh moves."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=0.05)
    window = 10
    db_file = str(tmp_path / "sim" / "equilibration.db")
    metro.pos_decisions = [1] * window  # so pos/or means are defined
    metro.or_decisions = [1] * window

    # fewer than `window` volume moves -> no tuning
    metro.vol_decisions = [1] * (window - 1)  # all accepted: would grow if tuned
    before = metro.vol_delt
    metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)
    assert metro._vol_tune_idx == 0
    assert metro.vol_delt == before

    # top up to a full fresh window -> tunes once, idx advances, delta grows
    metro.vol_decisions.append(1)
    metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)
    assert metro._vol_tune_idx == window
    assert metro.vol_delt > before

    # no new fresh moves -> no further tuning
    grown = metro.vol_delt
    metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)
    assert metro._vol_tune_idx == window
    assert metro.vol_delt == grown


def test_vol_delt_capped_on_accepted_volume_moves(four_particle_frame, tmp_path):
    """A run of accepted volume moves must not grow vol_delt past the cap."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=MAX_VOL_DELT)
    window = 10
    metro.pos_decisions = [1] * window
    metro.or_decisions = [1] * window
    db_file = str(tmp_path / "sim" / "equilibration.db")
    for _ in range(20):
        # fresh accepted volume moves each block ⇒ tuner repeatedly tries to grow
        metro.vol_decisions += [1] * window
        metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)

    assert metro.vol_delt <= MAX_VOL_DELT


def test_vol_delt_slew_bounded_per_update(four_particle_frame, tmp_path):
    """vol_max_scale caps how much a single tuning update may grow vol_delt, even
    when acceptance is high enough to warrant a larger jump; None falls back to the
    shared max_scale (current behavior)."""
    window = 10
    db_file = str(tmp_path / "sim" / "equilibration.db")

    def grow_factor(vol_max_scale):
        metro = make_metro(four_particle_frame, tmp_path, vol_delt=0.05)
        metro.pos_decisions = [1] * window
        metro.or_decisions = [1] * window
        metro.vol_decisions = [1] * window  # all accepted ⇒ tuner wants to grow
        before = metro.vol_delt
        metro.block_update(
            window, [], db_file, dynamic_delta=True,
            buffer_size=1, vol_max_scale=vol_max_scale,
        )
        return metro.vol_delt / before

    # fresh_acc=1.0 ⇒ raw ratio 1/TARGET_ACC_RATE ≈ 3.6, clamped to the slew bound
    assert grow_factor(1.02) == pytest.approx(1.02)
    assert grow_factor(None) == pytest.approx(1.1)  # shared max_scale default


def test_vol_delt_slew_limits_rate_not_ceiling(four_particle_frame, tmp_path):
    """With a tight slew bound, vol_delt never jumps by more than the bound per
    update, yet still climbs past its start toward its natural ceiling over many
    updates — the *rate* is limited, the endpoint is not capped."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=0.05)
    window = 10
    metro.pos_decisions = [1] * window
    metro.or_decisions = [1] * window
    db_file = str(tmp_path / "sim" / "equilibration.db")
    start = metro.vol_delt
    prev = start
    for _ in range(200):
        metro.vol_decisions += [1] * window  # all accepted every block
        metro.block_update(
            window, [], db_file, dynamic_delta=True,
            buffer_size=1, vol_max_scale=1.02,
        )
        assert metro.vol_delt <= prev * 1.02 + 1e-12  # never jumps past the bound
        prev = metro.vol_delt
    assert metro.vol_delt > start  # not pinned at the start value
    assert metro.vol_delt == pytest.approx(MAX_VOL_DELT)  # walked up to its ceiling


def test_or_delt_capped_at_geometric_ceiling(four_particle_frame, tmp_path):
    """The adapted rotation width never grows past MAX_OR_DELT (π): or_delt is a
    rotation angle, so larger values only re-parameterize the same move while the
    tuner chases an acceptance target a flat orientational landscape can't give."""
    metro = make_metro(four_particle_frame, tmp_path, or_delt=MAX_OR_DELT)
    window = 10
    metro.pos_decisions = [1] * window
    db_file = str(tmp_path / "sim" / "equilibration.db")
    for _ in range(20):
        # every rotation accepted ⇒ tuner repeatedly tries to grow or_delt
        metro.or_decisions += [1] * window
        metro.block_update(window, [], db_file, dynamic_delta=True, buffer_size=1)

    assert metro.or_delt <= MAX_OR_DELT


def test_max_or_delt_caps_rotation_width(four_particle_frame, tmp_path):
    """An explicit max_or_delt clamps or_delt at that value — the guard that keeps
    a crystal start from being orientationally melted by near-randomizing
    rotations — while leaving pos_delt tuning untouched."""
    cap = 0.25
    metro = make_metro(four_particle_frame, tmp_path, or_delt=0.2, pos_delt=0.1)
    window = 10
    db_file = str(tmp_path / "sim" / "equilibration.db")
    for _ in range(20):
        # everything accepted ⇒ tuner wants to grow both widths every block
        metro.pos_decisions += [1] * window
        metro.or_decisions += [1] * window
        metro.block_update(
            window, [], db_file, dynamic_delta=True, buffer_size=1, max_or_delt=cap
        )

    assert metro.or_delt == pytest.approx(cap)
    assert metro.pos_delt > 0.1  # pos_delt kept tuning independently


def test_equilibrate_forwards_max_or_delt(four_particle_frame, tmp_path):
    """equilibrate threads max_or_delt through to block_update and records it in
    the run config, so a capped run is reproducible from its config."""
    import json

    cap = 0.05  # tight enough that a short free-rotor run must hit it
    metro = make_metro(four_particle_frame, tmp_path, or_delt=cap)
    metro.equilibrate(
        num_steps=300, block_size=20, buffer_size=50, progress=False, max_or_delt=cap
    )
    assert metro.or_delt <= cap
    with open(os.path.join(metro.output_dir, "run_config.json")) as f:
        assert json.load(f)["run"]["max_or_delt"] == cap


def test_vol_delt_tunes_during_equilibration(four_particle_frame, tmp_path):
    """A real (short) equilibration advances the tune index yet keeps vol_delt
    clamped within bounds."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=0.05)
    metro.equilibrate(num_steps=300, block_size=20, buffer_size=50, progress=False)
    assert metro._vol_tune_idx >= 10  # window = block_size // 2; tuning happened
    assert MIN_VOL_DELT <= metro.vol_delt <= MAX_VOL_DELT


def test_production_does_not_tune_volume(four_particle_frame, tmp_path):
    """dynamic_delta=False leaves vol_delt and the tune index untouched even
    with a full window of decisions, but still records vol_acc_rate."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=0.05)
    window = 10
    metro.pos_decisions = [1] * window
    metro.or_decisions = [1] * window
    metro.vol_decisions = [1, 0] * window  # 20 moves, acc 0.5
    before = metro.vol_delt
    db_file = str(tmp_path / "sim" / "equilibration.db")
    metro.block_update(window, [], db_file, dynamic_delta=False, buffer_size=1)
    assert metro.vol_delt == before
    assert metro._vol_tune_idx == 0
    # recording is decoupled from tuning: vol_acc_rate is still logged
    row = connect(db_file).get(1)
    assert row.vol_acc_rate == pytest.approx(0.5)


def test_nvt_leaves_vol_delt_untouched(four_particle_frame, tmp_path):
    """NVT attempts no volume moves, so vol_delt never changes and no error."""
    metro = make_metro(four_particle_frame, tmp_path, npt_ensemble=False, vol_delt=0.05)
    before = metro.vol_delt
    metro.equilibrate(num_steps=300, block_size=50, buffer_size=50, progress=False)
    assert metro.vol_delt == before
    assert metro._vol_tune_idx == 0


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


def test_from_equilibration_resets_vol_delt(four_particle_frame, tmp_path):
    """vol_delt=X overrides the carried db value; the rest of the state is unchanged."""
    metro = make_metro(four_particle_frame, tmp_path, vol_delt=MAX_VOL_DELT)
    metro.equilibrate(num_steps=200, block_size=50, buffer_size=10)

    out = str(tmp_path / "sim")
    row = connect(out + "/equilibration.db").get(connect(out + "/equilibration.db").count())

    reset = MetropolisCalculator.from_equilibration(out, vol_delt=0.05)
    assert reset.vol_delt == 0.05                 # overridden, not the carried value
    assert reset.pos_delt == row.pos_delta        # other deltas still from the db
    assert reset.or_delt == row.or_delta
    assert reset.step_count == row.step           # still resumes in place

    kept = MetropolisCalculator.from_equilibration(out)  # default keeps tuned value
    assert kept.vol_delt == row.vol_delta


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
