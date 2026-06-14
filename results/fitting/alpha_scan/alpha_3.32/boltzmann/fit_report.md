# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.0020013118617665483

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.61718 | Angstrom |
| eps0 | 0.0262399 | eV |
| kappa | 0.586547 | dimensionless |
| kappa_prime | 2.55643 | dimensionless |
| mu | 0.111587 | dimensionless |
| nu | 1 | dimensionless |
| Q | -3.51374 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.38 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2454 | 0.1148 | 4.713 | 0.8766 |
| test | 1365 | 0.2568 | 0.1183 | 3.389 | 0.8736 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3775
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2402 (benzene sublimation ref ~ -0.456)
