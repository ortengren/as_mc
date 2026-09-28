import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from asmcmc.mc.potentials import CACELLI_POTENTIAL, DEFAULT_POTENTIAL, GBQPotential
from asmcmc.delta_learning.dimer_benchmark import (
    DEFAULT_REFERENCE,
    UMA_DIMER_PATH,
    DimerBenchmark,
    benzene_monomer,
    cacelli_dimer_frames,
    cg_scan,
    dimer_benchmark,
    dimer_scan,
    family_labels,
    load_cacelli_dimers,
    load_reference_dimers,
    load_uma_dimers,
    score_energies,
)
from asmcmc.mc.coarse_graining import disc_normal
from asmcmc.units import EV_TO_KCAL


# The GB+Q refit to the DFT crystal energies, frozen inline. It fits those energies
# well (test RMSE ~3 kcal/mol) yet is repulsive at the cofacial stacking distance
# and anti-correlated with the dimer wells: the failure this benchmark exists to
# catch.
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


def test_condensed_refit_is_the_tracked_uniform_fit():
    """CONDENSED_REFIT is results/fitting's uniform seed-0 fit, frozen inline."""
    tracked = GBQPotential.from_json(
        Path(__file__).resolve().parents[1]
        / "results/fitting/multiseed/uniform/seed_0/uniform/params.json"
    )
    assert tracked.name == "multiseed/uniform/seed_0/uniform"
    assert tracked.gb_params_dict() == CONDENSED_REFIT.gb_params_dict()
    assert tracked.Q == CONDENSED_REFIT.Q


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


def test_default_potential_clears_the_gate(data):
    """``DEFAULT_POTENTIAL``, which every ``MetropolisSampler`` gets when no
    ``potential=`` is passed, must itself clear the gate: a default with
    ``CONDENSED_REFIT``'s failure mode would silently corrupt every run.
    """
    bench = dimer_benchmark(DEFAULT_POTENTIAL, data)
    assert bench.stacking_bound
    assert bench.stacking_energy_kcal < 0.0
    assert bench.well_rmse_kcal < 1.0


# --- reporting ---

def test_summary_is_printable(cacelli_bench):
    s = cacelli_bench.summary()
    assert isinstance(cacelli_bench, DimerBenchmark)
    for token in ("cofacial", "parallel_displaced", "t_shaped", "RMSE", "bound"):
        assert token in s

# --- UMA as ground truth ---
# The project scores corrections against UMA, not MP2: GBQIII was *fitted* to
# these MP2 rows, so scoring a correction against them rewards standing still.
# The geometries survive as a structural probe; see the module docstring.

@pytest.fixture(scope="module")
def uma_data():
    return load_uma_dimers()


def test_default_reference_is_uma():
    assert DEFAULT_REFERENCE == "uma"
    assert load_reference_dimers().reference == "uma"


def test_uma_shares_the_cacelli_geometry(data, uma_data):
    """Same 197 dimers, different energies -- the point of the swap."""
    assert len(uma_data) == len(data)
    assert uma_data.reference == "uma" and data.reference == "mp2"
    for attr in ("uhat1", "uhat2", "r", "euler_deg"):
        np.testing.assert_allclose(getattr(uma_data, attr), getattr(data, attr))
    assert not np.allclose(uma_data.energy_kcal, data.energy_kcal)


def test_reference_dispatch(data, uma_data):
    np.testing.assert_allclose(
        load_reference_dimers("mp2").energy_kcal, data.energy_kcal
    )
    np.testing.assert_allclose(
        load_reference_dimers("uma").energy_kcal, uma_data.energy_kcal
    )
    with pytest.raises(ValueError, match="unknown reference"):
        load_reference_dimers("ccsdt")


def test_geometry_mismatch_raises(tmp_path):
    """A reference file that is not these geometries must not score silently."""
    rows = UMA_DIMER_PATH.read_text().splitlines()
    header, first = rows[0], rows[1].split(",")
    first[1] = f"{float(first[1]) + 1.0:.4f}"  # shift one x by 1 A
    bad = tmp_path / "dimer_energies.csv"
    bad.write_text("\n".join([header, ",".join(first), *rows[2:]]))
    with pytest.raises(ValueError, match="centre offsets disagree"):
        load_uma_dimers(bad)


def test_missing_reference_names_the_regeneration_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="uma_cacelli_dimers"):
        load_uma_dimers(tmp_path / "absent.csv")


def test_baseline_against_itself_shows_no_gain(uma_data):
    """Scoring the baseline as the candidate must report exactly zero gain."""
    b = dimer_benchmark(CACELLI_POTENTIAL, uma_data, baseline=CACELLI_POTENTIAL)
    assert b.reference == "uma"
    assert b.well_rmse_kcal == pytest.approx(b.baseline_well_rmse_kcal)
    assert b.well_rmse_gain_kcal == pytest.approx(0.0, abs=1e-12)
    assert b.improves_on_baseline is False


def test_baseline_may_be_skipped(uma_data):
    b = dimer_benchmark(CACELLI_POTENTIAL, uma_data, baseline=None)
    assert b.baseline_name == ""
    assert np.isnan(b.baseline_well_rmse_kcal)


def test_uma_wants_every_well_deeper_than_gbq(uma_data):
    """Why the reference swap changes the sign of the verdict.

    Under UMA all three families bind deeper than GB+Q does, so the correct
    correction *deepens* every well -- exactly the move the MP2-referenced
    score recorded as 'over-binding the cofacial stack'.
    """
    b = dimer_benchmark(CACELLI_POTENTIAL, uma_data)
    for family, well in b.wells.items():
        assert well.ab_depth < well.model_at_ab_min, family


def test_summary_names_the_reference_and_baseline(uma_data):
    s = dimer_benchmark(CACELLI_POTENTIAL, uma_data).summary()
    assert "reference: uma" in s
    assert "vs baseline" in s


# --- atomistic reconstruction (geometry only; no MLIP needed) ---


@pytest.fixture(scope="module")
def frames(data):
    return cacelli_dimer_frames(data)


def test_monomer_matches_supplement_frame():
    """README: ring in the xz-plane, two C-H bonds on the z-axis."""
    mol = benzene_monomer()
    assert mol.get_chemical_formula() == "C6H6"
    pos = mol.get_positions()
    assert np.allclose(mol.get_center_of_mass(), 0.0, atol=1e-9)
    assert np.abs(pos[:, 1]).max() < 1e-9  # planar in xz
    on_z = np.abs(pos[:, [0, 1]]).max(axis=1) < 1e-9
    assert on_z[:6].sum() == 2 and on_z[6:].sum() == 2  # two C-H bonds on z
    assert np.allclose(np.abs(disc_normal(pos)), [0.0, 1.0, 0.0], atol=1e-9)
    assert mol.info["charge"] == 0 and mol.info["spin"] == 1


def test_frames_are_two_rigid_benzenes(frames, data):
    assert len(frames) == len(data) == 197
    for f in frames[:5]:
        assert len(f) == 24
        assert np.bincount(f.arrays["molecule_id"]).tolist() == [12, 12]
        assert not f.pbc.any()
        assert f.info["charge"] == 0 and f.info["spin"] == 1
    ref = benzene_monomer().get_all_distances()
    for f in frames[::40]:
        for half in (slice(0, 12), slice(12, 24)):
            assert np.allclose(f[half].get_all_distances(), ref, atol=1e-9)


def test_frames_agree_with_coarse_grained_data(frames, data):
    """B's centre and disc normal reproduce ``data.r`` and ``data.uhat2``, so UMA is
    scored on the same geometries as the pair potentials."""
    for k, f in enumerate(frames):
        b = f[12:]
        assert np.allclose(b.get_center_of_mass(), data.r[k], atol=1e-8)
        assert abs(abs(disc_normal(b.get_positions()) @ data.uhat2[k]) - 1) < 1e-6


def test_zero_angle_rows_are_pure_translations(frames, data):
    """144 of 197 rows carry no angles, so no Euler convention is involved."""
    ang0 = np.all(data.euler_deg == 0.0, axis=1)
    assert ang0.sum() == 144
    for k in np.where(ang0)[0]:
        pos = frames[k].get_positions()
        assert np.allclose(pos[12:] - pos[:12], data.r[k], atol=1e-9)


def test_euler_seq_threads_through_to_geometry(data):
    default = cacelli_dimer_frames(data)
    other = cacelli_dimer_frames(data, euler_seq="xyz")
    ang0 = np.all(data.euler_deg == 0.0, axis=1)
    for k in np.where(ang0)[0][:5]:
        assert np.allclose(default[k].get_positions(), other[k].get_positions())
    moved = sum(
        not np.allclose(default[k].get_positions(), other[k].get_positions())
        for k in np.where(~ang0)[0]
    )
    assert moved > 0


# --- the Euler convention, pinned by the supplement's own energies ---


def _contacts(frame):
    """Sorted A-B atom-atom distances, per element pair: equal for congruent dimers."""
    pos = frame.get_positions()
    sym = np.array(frame.get_chemical_symbols())
    a, b = slice(0, 12), slice(12, 24)
    out = []
    for pair in (("C", "C"), ("C", "H"), ("H", "H")):
        d = []
        for s1, s2 in {pair, pair[::-1]}:
            pa, pb = pos[a][sym[a] == s1], pos[b][sym[b] == s2]
            d.append(np.linalg.norm(pa[:, None] - pb[None], axis=-1).ravel())
        out.append(np.sort(np.concatenate(d)))
    return np.concatenate(out)


def _row(data, euler, r):
    match = np.all(data.euler_deg == euler, axis=1) & np.all(np.isclose(data.r, r), axis=1)
    assert match.sum() == 1
    return int(np.argmax(match))


def test_repeated_t_shape_energies_rebuild_the_same_dimer(frames, data):
    """The supplement gives (90, 90, 0) along y the energies of (0, 90, 90) along z
    (-2.27956 kcal/mol at 5 A in both), so the two must be one dimer seen from two
    frames. Only proper Euler sequences such as z-y-z make them congruent."""
    for dist in (4.5, 5.0, 6.0, 7.0):
        i = _row(data, [0, 90, 90], [0, 0, dist])
        j = _row(data, [90, 90, 0], [0, dist, 0])
        assert data.energy_kcal[i] == pytest.approx(data.energy_kcal[j], abs=2e-3)
        assert np.allclose(_contacts(frames[i]), _contacts(frames[j]), atol=1e-6)


def test_gamma_turns_b_about_the_c_h_bond_it_points_at_a(frames, data):
    """(0, 0, 90) along z keeps the head-on H...H contact of the in-plane (0, 0, 0)
    row at the same distance: MP2 puts both on a repulsive wall at 6.5 A."""
    turned = _row(data, [0, 0, 90], [0, 0, 6.5])
    in_plane = _row(data, [0, 0, 0], [0, 0, 6.5])
    assert data.energy_kcal[turned] > 3.0 and data.energy_kcal[in_plane] > 3.0
    assert _contacts(frames[turned]).min() == pytest.approx(_contacts(frames[in_plane]).min())


def test_no_rebuilt_dimer_has_an_atom_clash(frames):
    """The closest contact in the set is that head-on H...H (1.54 A). Readings that
    go below it put atoms 1.3 A apart in rows MP2 calls only +25 kcal/mol."""
    closest = min(_contacts(f).min() for f in frames)
    assert closest == pytest.approx(1.535, abs=1e-3)


def test_uma_reference_was_built_on_the_current_geometry(uma_data):
    """The tracked CSV's GBQIII column matches GBQIII on the loader's normals, so the
    reference was generated under the current ``EULER_SEQ``."""
    with open(UMA_DIMER_PATH, newline="") as fh:
        csv_gbq = np.array([float(row["e_gbq_kcal"]) for row in csv.DictReader(fh)])
    gbq = (
        CACELLI_POTENTIAL.pair_energy(uma_data.uhat1, uma_data.uhat2, uma_data.r)
        * EV_TO_KCAL
    )
    assert np.allclose(csv_gbq, gbq, atol=1e-5)


# --- scoring precomputed energies ---


def test_score_energies_reproduces_dimer_benchmark(data, cacelli_bench):
    """The generic scorer and the pair-potential entry point agree exactly."""
    model = CACELLI_POTENTIAL.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL
    b = score_energies(
        model, data, name=CACELLI_POTENTIAL.name, scan_fn=cg_scan(CACELLI_POTENTIAL)
    )
    assert b.full_pearson_r == cacelli_bench.full_pearson_r
    assert b.full_rmse_kcal == cacelli_bench.full_rmse_kcal
    assert b.well_pearson_r == cacelli_bench.well_pearson_r
    assert b.well_rmse_kcal == cacelli_bench.well_rmse_kcal
    assert b.wells == cacelli_bench.wells


def test_score_without_scan_falls_back_to_data_rows(data, cacelli_bench):
    """scan_fn=None still finds each family's well, from the rows themselves."""
    model = CACELLI_POTENTIAL.pair_energy(data.uhat1, data.uhat2, data.r) * EV_TO_KCAL
    b = score_energies(model, data, name="rows-only")
    for fam, well in b.wells.items():
        assert well.model_depth <= cacelli_bench.wells[fam].model_at_ab_min + 1e-9
        assert well.model_depth == pytest.approx(cacelli_bench.wells[fam].model_depth, abs=0.3)


def test_family_labels_partition_the_rows(data):
    labels = family_labels(data)
    assert len(labels) == len(data)
    counts = {name: int((labels == name).sum()) for name in set(labels)}
    for fam in ("cofacial", "parallel_displaced", "t_shaped"):
        assert counts[fam] > 0
    assert sum(counts.values()) == len(data)


# --- the regeneration script, with UMA replaced by a stub ---


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regeneration_script_writes_a_reference_the_benchmark_accepts(
    tmp_path, monkeypatch, make_stub_calculator
):
    """``scripts/uma_cacelli_dimers.py`` end to end, without fairchem."""

    class PairwiseStub(make_stub_calculator):
        def get_potential_energy(self, atoms=None):
            self.n_calls += 1
            d = atoms.get_all_distances()
            return -float(np.sum(1.0 / d[np.triu_indices(len(atoms), k=1)]))

    script = _load_script("uma_cacelli_dimers")
    monkeypatch.setattr(script, "load_uma_calculator", lambda *a, **k: PairwiseStub())
    monkeypatch.setattr(
        "sys.argv", ["uma_cacelli_dimers.py", "--out-dir", str(tmp_path), "--scan-points", "3"]
    )
    script.main()

    reference = load_uma_dimers(tmp_path / "dimer_energies.csv")
    assert len(reference) == 197 and reference.reference == "uma"
    assert np.all(np.isfinite(reference.energy_kcal))
    assert (tmp_path / "family_curves.csv").exists()
