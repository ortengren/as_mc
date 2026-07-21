Beyond a `requirements.txt`, this folder contains the following files:

Data files:
- benzenes.xyz: an ase-type xyz file containing 6,826 configurations of benzene crystals and their corresponding energetics computed  using QuantumEspresso v7.046 using Perdew–Burke–Ernzerhof (PBE) pseudopotentials and cutoff parameters reported by Prandini et al.,  Grimme D3-dispersion correction, and a 3 × 3 Monkhorst–Pack k-point grid. Data was initially managed by `signac`, for which there are `signac` hashes found in the frame info (this makes it easier to correlate configurations with the file below).
- ellipsoids.xyz: an ase-type xyz file containing 6,826 configurations of benzene crystals as represented by ellipsoidal bodies. These configurations directly correlate with those in `benzenes.xyz`, and can be identified out-of-order using the `signac` hash code.

In `hyperparamter_tuning.py`, there are the following scripts to 1) run a Bayesian-optimization loop and grid search of the AniSOAP parameters that best approximate the atomistic interactions (as defined by the Global Feature Reconstruction Error), and 2) plot the  optimization results.

To generate the figures, the following scripts must be run in order:
- make_descriptors.py: This code reads the frames in `benzenes.xyz` and `ellipsoids.xyz` and generates the corresponding SOAP and AniSOAP descriptors, saving them in a folder labeled `representations` (which should be created before running this file).
- make_parity_plots.py: This code pulls the saved descriptors and tests their efficacy in learning first-principles energetics using regularized linear regression.
- make_learning_curve.py: This code generates a learning curve for the AniSOAP representation.
- make_chemiscope.py: This code generates a chemiscope-type visualization file of PCA and PCovR mappings and the corresponding configurations. This includes a mapping of the error in the AniSOAP approximation.
