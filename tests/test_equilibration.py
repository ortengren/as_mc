import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ase.db import connect

from asmcmc.config import RunConfig
from asmcmc.equilibration import pressure_ramp
from asmcmc.metropolis import MetropolisCalculator
from asmcmc.initialize import RandomLatticeInitializer


def _init(seed=1):
    return RandomLatticeInitializer(n_particles=27, density=0.3, seed=seed)


def _stage_pressure(stage_dir):
    return RunConfig.load(os.path.join(stage_dir, "run_config.json")).pressure


def _last_step(stage_dir):
    with connect(os.path.join(stage_dir, "equilibration.db")) as db:
        return db.get(db.count()).step


# ---------------------------------------------------------------------------
# pressure_ramp: stage artifacts + schedule
# ---------------------------------------------------------------------------


def test_pressure_ramp_writes_a_resumable_dir_per_stage(tmp_path):
    """Each pressure stage gets its own run dir holding the equilibration db, a
    write-once run_config stamped with that stage's pressure, and diagnostics."""
    pressures = [1e-6, 2e-6, 3e-6]
    out = str(tmp_path / "ramp")

    dirs = pressure_ramp(300.0, pressures, 2 * 27, _init(), out, block_size=27)

    assert len(dirs) == len(pressures)
    for d, p in zip(dirs, pressures):
        assert os.path.exists(os.path.join(d, "equilibration.db"))
        assert os.path.exists(os.path.join(d, "run_config.json"))
        assert os.path.exists(os.path.join(d, "equilibration_diagnostics.png"))
        # each stage's config carries its own pressure, in ascending order
        assert _stage_pressure(d) == pytest.approx(p)


def test_pressure_ramp_sorts_pressures_ascending(tmp_path):
    """A shuffled schedule is walked low -> high (a pressurization)."""
    out = str(tmp_path / "ramp")
    dirs = pressure_ramp(300.0, [3e-6, 1e-6, 2e-6], 2 * 27, _init(), out, block_size=27)
    pressures = [_stage_pressure(d) for d in dirs]
    assert pressures == sorted(pressures)


def test_pressure_ramp_per_stage_step_budget(tmp_path):
    """`num_steps` is per-stage: each stage's db advances to its own budget from 0,
    and a per-stage sequence sizes the stages individually."""
    out = str(tmp_path / "ramp")
    dirs = pressure_ramp(
        300.0, [1e-6, 2e-6], [2 * 27, 4 * 27], _init(), out, block_size=27
    )
    assert _last_step(dirs[0]) == pytest.approx(2 * 27, abs=27)
    assert _last_step(dirs[1]) == pytest.approx(4 * 27, abs=27)


def test_pressure_ramp_stages_are_resumable(tmp_path):
    """The final stage is a normal equilibration dir: from_equilibration rebuilds
    a sampler at that stage's pressure and step count."""
    out = str(tmp_path / "ramp")
    dirs = pressure_ramp(300.0, [1e-6, 5e-6], 2 * 27, _init(), out, block_size=27)
    metro = MetropolisCalculator.from_equilibration(dirs[-1])
    assert metro.pressure == pytest.approx(5e-6)
    assert metro.step_count > 0
    assert len(metro.current_frame) == 27


def test_pressure_ramp_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        pressure_ramp(
            300.0, [1e-6, 2e-6], [2 * 27], _init(), str(tmp_path / "r"), block_size=27
        )


def test_pressure_ramp_deterministic(tmp_path):
    """Same seed + schedule -> identical final frame across two runs."""
    finals = []
    for _ in range(2):
        out = str(tmp_path / f"ramp_{_}")
        dirs = pressure_ramp(
            300.0, [1e-6, 2e-6], 2 * 27, _init(seed=7), out, seed=7, block_size=27
        )
        with connect(os.path.join(dirs[-1], "equilibration.db")) as db:
            finals.append(db.get(db.count()).toatoms().get_volume())
    assert finals[0] == pytest.approx(finals[1])
