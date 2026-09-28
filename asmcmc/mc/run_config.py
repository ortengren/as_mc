"""``run_config.json``: the static definition of a run, written once when it starts."""

import json
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from asmcmc.mc.potentials import potential_from_dict


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to rebuild a sampler, apart from its evolving state.

    The last db row supplies the evolving state (frame, tuned move widths, step);
    together they let ``MetropolisSampler.from_equilibration`` resume a run.
    """

    temp: float
    pressure: float
    npt_ensemble: bool
    nl_radius: float
    nl_skin: float
    potential: dict  # potential.to_dict() — self-contained, incl. name + params
    pos_delt: float  # initial deltas (run provenance; tuned values live in the db)
    or_delt: float
    vol_delt: float
    init: dict  # initializer.provenance()  (already JSON-ready)
    # anisotropic (single-axis) volume moves? Defaults False so a run_config.json
    # written before this flag existed loads as the isotropic moves that run used;
    # from_sampler stamps the live sampler's value for new runs.
    aniso_vol: bool = False
    run: dict | None = (
        None  # call-time knobs (num_steps, block_size, …) — provenance only
    )
    version: int = 1

    @classmethod
    def from_sampler(cls, sampler, run=None):
        return cls(
            temp=sampler.temp,
            pressure=sampler.pressure,
            npt_ensemble=sampler.npt_ensemble,
            nl_radius=sampler.nl_radius,
            nl_skin=sampler.nl_skin,
            potential=sampler.potential.to_dict(),
            pos_delt=sampler.pos_delt,
            or_delt=sampler.or_delt,
            vol_delt=sampler.vol_delt,
            init=sampler.initializer.provenance(),
            aniso_vol=sampler.aniso_vol,
            run=run,
        )

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            # An old run_config.json can carry fields RunConfig no longer has.
            # Drop them rather than fail the resume, but say so: it loses
            # provenance.
            warnings.warn(
                f"{path}: dropping unknown RunConfig field(s) {sorted(unknown)}",
                stacklevel=2,
            )
            raw = {k: v for k, v in raw.items() if k in known}
        return cls(**raw)

    def build_potential(self):
        return potential_from_dict(self.potential)
