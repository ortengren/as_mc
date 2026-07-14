from dataclasses import asdict, dataclass
from asmcmc.potentials import potential_from_dict

import json
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
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
    # from_calculator stamps the live sampler's value for new runs.
    aniso_vol: bool = False
    run: dict | None = (
        None  # call-time knobs (num_steps, block_size, …) — provenance only
    )
    version: int = 1

    @classmethod
    def from_calculator(cls, metro, run=None):
        return cls(
            temp=metro.temp,
            pressure=metro.pressure,
            npt_ensemble=metro.npt_ensemble,
            nl_radius=metro.nl_radius,
            nl_skin=metro.nl_skin,
            potential=metro.potential.to_dict(),
            pos_delt=metro.pos_delt,
            or_delt=metro.or_delt,
            vol_delt=metro.vol_delt,
            init=metro.initializer.provenance(),
            aniso_vol=metro.aniso_vol,
            run=run,
        )

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def build_potential(self):
        return potential_from_dict(self.potential)
