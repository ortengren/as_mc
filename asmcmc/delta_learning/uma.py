"""Thin wrapper around Meta FAIR Chemistry's OMol-trained UMA MLIP.

`fairchem-core` is a heavy optional dependency (it pulls in PyTorch) and the
UMA checkpoints are gated on Hugging Face, so the import happens inside
:func:`load_uma_calculator` -- importing ``asmcmc`` must not require either.

The calculator returned here is an ordinary ASE calculator; the OMol task
expects ``charge`` and ``spin`` in ``Atoms.info`` (neutral benzene: 0 and 1).
"""

import inspect

# fairchem 2.12 ships uma-s-1{,p1} and uma-m-1p1; newer releases add 1p2. Ask
# for what is actually registered here and let callers override.
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

    ``seed`` is forwarded only when the installed fairchem accepts it -- the
    kwarg exists in some releases and not others (2.12 has no ``seed``).
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
    """Potential energy (eV) of a copy of ``atoms``, OMol bookkeeping applied.

    Works on a copy so the caller's frame keeps whatever calculator (or none)
    it already had -- attaching a live MLIP to frames that are about to be
    written out is how stale calculators end up in trajectories.
    """
    at = atoms.copy()
    at.set_pbc(atoms.pbc)
    at.info.setdefault("charge", charge)
    at.info.setdefault("spin", spin)
    at.calc = calculator
    return float(at.get_potential_energy())
