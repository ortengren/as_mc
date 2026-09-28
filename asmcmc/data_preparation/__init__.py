"""Dataset generation for potential fitting: labelled configurations and features.

Model-agnostic inputs to a fit, kept out of the fit packages themselves.
``cluster_dataset`` emits UMA-labelled benzene clusters (carrying a
``gbq_interaction_energy`` baseline, so a GBQ refit and a Delta-learning fit
consume the same frames). Model-*specific* featurisation belongs to its fit
package instead -- see ``fitting_gbq.data``, whose ``(r, a_i, a_j, b_ij)``
invariants are meaningless outside the GB+Q closed form, and
``fitting_anisoap.data``, which owns the AniSOAP descriptors.

Depends on ``base``/``utils``; nothing in either may import this.
"""
