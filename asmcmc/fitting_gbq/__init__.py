"""Fitting Gay-Berne + quadrupole parameters to the DFT benzene crystal energies.

The refit reproduces those energies well but fails the dimer benchmark (its
cofacial stack is repulsive), so it is not used for MC; see docs/findings.md.
``python -m asmcmc.fitting_gbq.run`` fits, and ``python -m asmcmc.fitting_gbq.summary``
redraws the campaign figures in ``results/fitting/summary``.
"""
