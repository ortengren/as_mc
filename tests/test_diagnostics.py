"""Run-directory diagnostics: the loader's reductions and the four figures."""

import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest

from ase.db import connect

from asmcmc.diagnostics import (
    PLOTS,
    TAIL_FRACTION,
    TAIL_MAX_FRAMES,
    load_run,
    render,
)
from asmcmc.measurements import RadialDistributionFunction, nematic_q_tensor

from asmcmc.npt_equilibration import _evaluate_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point(out_dir, num_steps=12 * 27):
    """One equilibrated run dir, built the way the scan builds them."""
    cfg = {
        "n_particles": 27,
        "density": 0.3,
        "num_steps": num_steps,
        "block_size": 27,
        "buffer_size": 100,
        "seed0": 100,
        "out_dir": out_dir,
    }
    _, d = _evaluate_point(0, 300.0, 0.0, cfg)
    return d


# ---------------------------------------------------------------------------
# load_run: one pass, correct reductions
# ---------------------------------------------------------------------------


def test_load_run_shapes_and_monotonic_steps(tmp_path):
    trace = load_run(_point(str(tmp_path / "scan")))
    n = len(trace.steps)
    assert n > 1
    for arr in (trace.cycles, trace.energy, trace.volume, trace.density,
                trace.nematic_s, trace.pos_acc, trace.or_acc, trace.vol_acc):
        assert len(arr) == n
    assert np.all(np.diff(trace.steps) > 0)
    assert trace.n_particles == 27
    # cycles is steps expressed in sweeps
    assert trace.cycles == pytest.approx(trace.steps / 27)


def test_load_run_density_and_nematic_match_the_db(tmp_path):
    """The reductions must be the db's own numbers, not a re-simulation."""
    d = _point(str(tmp_path / "scan"))
    trace = load_run(d)

    with connect(os.path.join(d, "equilibration.db")) as db:
        rows = sorted(db.select(), key=lambda r: r.step)

    assert trace.density[-1] == pytest.approx(
        rows[-1].num_particles / rows[-1].vol
    )
    assert trace.energy[-1] == pytest.approx(rows[-1].total_energy)
    assert trace.pos_acc[-1] == pytest.approx(rows[-1].pos_acc_rate)

    direct = float(
        np.linalg.eigvalsh(nematic_q_tensor(np.asarray(rows[-1].data["or_vec"])))[-1]
    )
    assert trace.nematic_s[-1] == pytest.approx(direct)


def test_load_run_rejects_empty_db(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with connect(str(empty / "equilibration.db")) as db:
        pass
    with pytest.raises(ValueError, match="no recorded frames"):
        load_run(str(empty))


# ---------------------------------------------------------------------------
# Tail averaging: a real window, capped, and actually averaged
# ---------------------------------------------------------------------------


def test_tail_covers_the_requested_window(tmp_path):
    trace = load_run(_point(str(tmp_path / "scan")))
    n = len(trace.steps)
    expected = min(max(1, math.ceil(TAIL_FRACTION * n)), TAIL_MAX_FRAMES)
    assert trace.tail_frames == expected
    assert trace.rdf["g_r"].shape == trace.rdf["r"].shape
    assert trace.ocf["s2_r"].shape == trace.ocf["r"].shape


def test_tail_average_differs_from_a_single_frame(tmp_path):
    """Averaging must be real -- a tail g(r) is not the last frame's g(r).

    This is the whole reason the tail exists: one frame at these particle counts
    is too noisy to read a phase off.
    """
    d = _point(str(tmp_path / "scan"), num_steps=40 * 27)
    trace = load_run(d)
    assert trace.tail_frames > 1, "need a multi-frame tail for this to mean anything"

    with connect(os.path.join(d, "equilibration.db")) as db:
        last = sorted(db.select(), key=lambda r: r.step)[-1]
    single = RadialDistributionFunction(12, 120)
    single.compute(last.toatoms(), last.key_value_pairs, last.data)
    one_frame = single.finalize()["g_r"]

    assert not np.allclose(trace.rdf["g_r"], one_frame)


# ---------------------------------------------------------------------------
# render: writes exactly what was asked for
# ---------------------------------------------------------------------------


def test_render_writes_every_figure(tmp_path):
    d = _point(str(tmp_path / "scan"))
    written = render(d)
    assert set(written) == set(PLOTS)
    for name, path in written.items():
        assert os.path.exists(path), name
        assert os.path.getsize(path) > 0, name
        assert os.path.basename(path) == f"equilibration_{name}.png"


def test_render_honours_the_selection(tmp_path):
    d = _point(str(tmp_path / "scan"))
    written = render(d, which=["phase"])
    assert set(written) == {"phase"}
    assert os.path.exists(os.path.join(d, "equilibration_phase.png"))
    # nothing else got drawn
    for other in set(PLOTS) - {"phase"}:
        assert not os.path.exists(os.path.join(d, f"equilibration_{other}.png"))


def test_render_rejects_unknown_plot(tmp_path):
    d = _point(str(tmp_path / "scan"))
    with pytest.raises(ValueError, match="unknown plot"):
        render(d, which=["phase", "nonsense"])


def test_render_db_stem_prefixes_output(tmp_path):
    """Rendering a production db must not clobber the equilibration figures."""
    d = _point(str(tmp_path / "scan"))
    render(d, which=["energy"])
    # produce a simulation.db alongside, then render that
    from asmcmc.npt_production import produce_point

    produce_point(d, num_steps=4 * 27, block_size=27)
    written = render(d, which=["energy"], db_name="simulation.db")
    assert os.path.basename(written["energy"]) == "simulation_energy.png"
    assert os.path.exists(os.path.join(d, "equilibration_energy.png"))


def test_render_out_dir_redirects_output(tmp_path):
    d = _point(str(tmp_path / "scan"))
    elsewhere = tmp_path / "figs"
    written = render(d, which=["energy"], out_dir=str(elsewhere))
    assert os.path.exists(written["energy"])
    assert str(elsewhere) in written["energy"]
    assert not os.path.exists(os.path.join(d, "equilibration_energy.png"))
