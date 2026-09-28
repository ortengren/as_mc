"""scripts/run_herringbone.py reproduces the per-temperature scripts it replaced."""

import importlib.util
import json
import pickle
from pathlib import Path

import pytest

from asmcmc.mc.potentials import CACELLI_POTENTIAL
from asmcmc.mc.run_config import RunConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_herringbone.py"


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("run_herringbone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_points_match_the_replaced_scripts(driver):
    """(T, seed, run dir) as in the old single_state_run_hb_1atm*.py scripts."""
    expected = {
        100.0: (45, "100.0_6.324209e-07/herringbone_jittered_2"),
        150.0: (313, "150.0_6.324209e-07/herringbone_jittered_0"),
        200.0: (312, "200.0_6.324209e-07/herringbone_jittered_0"),
        273.15: (311, "273.15_6.324209e-07/herringbone_jittered_0"),
        300.0: (123219, "300.0_6.324209e-07/herringbone_jittered_2"),
    }
    assert set(driver.STATE_POINTS) == set(expected)
    for temp, (seed, rel) in expected.items():
        assert driver.STATE_POINTS[temp][0] == seed
        assert driver.run_dir(temp) == driver.REPO_ROOT / "results" / "validation" / rel


def test_sampler_matches_the_recorded_protocol(driver, tmp_path, monkeypatch):
    """The static settings the old scripts wrote into each run_config.json."""
    monkeypatch.setattr(driver, "run_dir", lambda temp: tmp_path / "run")
    cfg = RunConfig.from_sampler(driver.build_sampler(150.0))

    assert (cfg.temp, cfg.pressure) == (150.0, 6.324209e-07)
    assert cfg.npt_ensemble and cfg.aniso_vol
    assert (cfg.nl_radius, cfg.nl_skin) == (6.8, 1.0)
    assert (cfg.pos_delt, cfg.or_delt, cfg.vol_delt) == (0.45, 0.6, 0.025)
    assert cfg.potential == CACELLI_POTENTIAL.to_dict()
    assert cfg.init == {
        "init_n_particles": 400,
        "init_density": pytest.approx(1.5144510965996862),
        "init_seed": 313,
        "init_sigma0": 5.72,
        "init_kappa": 0.542,
        "init_packing": "herringbone",
        "init_pos_jitter": 0.15,
        "init_or_jitter": 0.15,
    }


def test_stages_run_end_to_end(driver, tmp_path, monkeypatch):
    """equilibrate -> resume -> produce -> measure, with short step budgets."""
    run = tmp_path / "run"
    monkeypatch.setattr(driver, "run_dir", lambda temp: run)
    monkeypatch.setattr(driver, "EQUILIBRATION_STEPS", 800)
    monkeypatch.setattr(driver, "PRODUCTION_STEPS", 800)

    driver.equilibrate(150.0)
    recorded = json.loads((run / "run_config.json").read_text())["run"]
    assert recorded["num_steps"] == 800
    assert recorded["block_size"] == 400 and recorded["buffer_size"] == 500
    assert recorded["max_or_delt"] == 0.25 and recorded["dynamic_delta"] is True

    driver.resume(150.0, extra_steps=400)
    driver.produce(150.0)
    driver.measure(150.0)
    with open(run / "measurements.pkl", "rb") as f:
        results = pickle.load(f)
    assert set(results) == {"rdf", "ocf", "nematic", "enthalpy", "heat_capacity"}

    # a second start or production would append to (and corrupt) the existing db
    with pytest.raises(SystemExit, match="use `resume`"):
        driver.equilibrate(150.0)
    with pytest.raises(SystemExit, match="already has a simulation.db"):
        driver.produce(150.0)
