import json

import pytest

from asmcmc.base.config import RunConfig
from asmcmc.base.potentials import DEFAULT_POTENTIAL, potential_from_dict
from asmcmc.base.metropolis import MetropolisCalculator


def _default_config(**overrides):
    base = dict(
        temp=300.0,
        pressure=0.0,
        npt_ensemble=True,
        nl_radius=15.0,
        nl_skin=1.0,
        potential=DEFAULT_POTENTIAL.to_dict(),
        pos_delt=0.15,
        or_delt=0.05,
        vol_delt=0.05,
        init={"init_n_particles": 8, "init_density": 0.3, "init_seed": 0},
        run={"kind": "equilibration", "num_steps": 100},
    )
    base.update(overrides)
    return RunConfig(**base)


def test_potential_dict_round_trip():
    """GBQPotential.to_dict / potential_from_dict are inverses."""
    d = DEFAULT_POTENTIAL.to_dict()
    assert d["type"] == "GBQPotential"
    assert potential_from_dict(d) == DEFAULT_POTENTIAL


def test_runconfig_save_load_round_trip(tmp_path):
    """RunConfig survives a save -> load round trip, potential included."""
    cfg = _default_config()
    path = tmp_path / "run_config.json"
    cfg.save(path)
    assert RunConfig.load(path) == cfg
    assert cfg.build_potential() == DEFAULT_POTENTIAL


def test_runconfig_aniso_vol_round_trip(tmp_path):
    """aniso_vol survives a save -> load round trip."""
    path = tmp_path / "run_config.json"
    _default_config(aniso_vol=True).save(path)
    assert RunConfig.load(path).aniso_vol is True


def test_runconfig_aniso_vol_defaults_false_for_legacy(tmp_path):
    """A run_config.json predating the flag (no aniso_vol key) loads as isotropic
    (False) — faithful to the moves those runs actually used."""
    path = tmp_path / "run_config.json"
    _default_config(aniso_vol=True).save(path)
    data = json.loads(path.read_text())
    del data["aniso_vol"]  # simulate a config written before the flag existed
    path.write_text(json.dumps(data))
    assert RunConfig.load(path).aniso_vol is False


def test_runconfig_written_as_plain_json(tmp_path):
    """The config on disk is human-readable JSON with the expected fields."""
    cfg = _default_config(npt_ensemble=False)
    path = tmp_path / "run_config.json"
    cfg.save(path)
    data = json.loads(path.read_text())
    assert data["temp"] == 300.0
    assert data["npt_ensemble"] is False
    assert data["potential"]["type"] == "GBQPotential"


def test_equilibrate_writes_run_config(two_particle_frame, tmp_path):
    """A run stamps run_config.json with the static run definition."""
    metro = MetropolisCalculator(
        temp=250.0,
        pressure=0.0,
        init_frame=two_particle_frame,
        output_dir=str(tmp_path / "sim"),
    )
    metro.equilibrate(num_steps=100, block_size=50, buffer_size=10)

    cfg = RunConfig.load(tmp_path / "sim" / "run_config.json")
    assert cfg.temp == 250.0
    assert cfg.pressure == 0.0
    assert cfg.npt_ensemble is True
    assert cfg.nl_radius == metro.nl_radius
    assert cfg.nl_skin == metro.nl_skin
    assert cfg.build_potential() == metro.potential
    assert cfg.run["kind"] == "equilibration"
    # the sampler's live volume-move geometry is stamped into the config; the
    # default sampler is anisotropic
    assert metro.aniso_vol is True
    assert cfg.aniso_vol is True


def test_write_config_is_write_once(two_particle_frame, tmp_path):
    """A second write attempt (e.g. on resume) must not clobber the stamped config."""
    metro = MetropolisCalculator(
        temp=250.0,
        pressure=0.0,
        init_frame=two_particle_frame,
        output_dir=str(tmp_path / "sim"),
    )
    metro.equilibrate(num_steps=100, block_size=50, buffer_size=10)
    original = (tmp_path / "sim" / "run_config.json").read_text()

    metro._write_config(run={"kind": "should-not-appear"})
    assert (tmp_path / "sim" / "run_config.json").read_text() == original
