"""A thin wrapper around UMA, Meta FAIR Chemistry's machine-learned potential.

``fairchem-core`` is a heavy optional dependency (it pulls in PyTorch), and the
UMA checkpoints are gated on Hugging Face, so fairchem is only imported inside
:func:`load_uma_calculator`. Importing ``asmcmc`` needs neither.

The calculator returned here is an ordinary ASE calculator. The OMol task expects
``charge`` and ``spin`` in ``Atoms.info`` (0 and 1 for neutral benzene).
"""

import inspect

# fairchem 2.12 ships uma-s-1, uma-s-1p1 and uma-m-1p1, and newer releases add
# 1p2. The default is one that 2.12 has; callers can ask for another.
DEFAULT_UMA_MODEL = "uma-s-1p1"

_INSTALL_HINT = (
    "fairchem-core is required for UMA energies. Install it in a suitable "
    "PyTorch environment:\n"
    "  pip install fairchem-core ase numpy\n"
    "Then authenticate for the gated UMA checkpoint:\n"
    "  huggingface-cli login"
)


def load_uma_calculator(
    model=DEFAULT_UMA_MODEL, device="cpu", task_name="omol", seed=None
):
    """Build a ``FAIRChemCalculator`` for the pretrained ``model``.

    ``seed`` is only passed on if the installed fairchem accepts it. Some
    releases have that argument and others, including 2.12, don't.
    """
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc

    kwargs = {}
    if seed is not None and (
        "seed" in inspect.signature(pretrained_mlip.get_predict_unit).parameters
    ):
        kwargs["seed"] = int(seed)
    predictor = pretrained_mlip.get_predict_unit(model, device=device, **kwargs)
    return FAIRChemCalculator(predictor, task_name=task_name)


def frame_energy(atoms, calculator, charge=0, spin=1):
    """Potential energy (eV) of a copy of ``atoms``, with the charge and spin OMol needs.

    It works on a copy so the caller's frame keeps whatever calculator it had (or
    none). Attaching a live MLIP to frames that are about to be written out
    leaves stale calculators in trajectories.
    """
    at = atoms.copy()
    at.set_pbc(atoms.pbc)
    at.info.setdefault("charge", charge)
    at.info.setdefault("spin", spin)
    at.calc = calculator
    return float(at.get_potential_energy())
