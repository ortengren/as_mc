"""Monte Carlo of coarse-grained benzene, and an AniSOAP correction to its potential.

Subpackages:
    mc              the Metropolis sampler and tools for analysing its runs
    delta_learning  UMA-labelled dimers, AniSOAP descriptors and the Delta-model
    fitting_gbq     fitting Gay-Berne + quadrupole parameters to DFT crystal energies

``delta_learning`` and ``fitting_gbq`` build on ``mc``; ``mc`` imports neither.
"""
