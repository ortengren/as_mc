"""AniSOAP Delta-learning fit: representation, hyperparameter sweep, physics gate.

Fits ``Delta = E_UMA - E_GBQ`` on the UMA-labelled cluster campaigns written by
:mod:`asmcmc.data_preparation.cluster_dataset`, using AniSOAP ellipsoid
descriptors of the coarse-grained clusters.

Model-specific by design, which is why the descriptor code lives here rather
than in ``data_preparation/``: an AniSOAP power spectrum is meaningless outside
an AniSOAP fit, exactly as ``fitting_gbq.data``'s ``(r, a_i, a_j, b_ij)``
invariants are meaningless outside the GB+Q closed form.

Layering: may import ``base/`` and ``utils/``; nothing in either imports this.
"""
