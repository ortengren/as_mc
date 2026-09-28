"""An AniSOAP correction to the GB+quadrupole potential, learned from UMA dimers.

``dataset`` generates UMA-labelled benzene dimers, ``descriptors`` turns them into
AniSOAP features, ``model`` fits ``Delta = E_UMA - E_GBQ`` by ridge regression,
``sweep`` scans the descriptor hyperparameters, and ``dimer_benchmark`` is the
physics gate every candidate potential must pass before it is used in MC.
"""
