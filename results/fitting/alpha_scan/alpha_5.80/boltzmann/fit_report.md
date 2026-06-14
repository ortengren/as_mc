# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.0010921215028727

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.78546 | Angstrom |
| eps0 | 0.0165304 | eV |
| kappa | 0.598622 | dimensionless |
| kappa_prime | 3.57777 | dimensionless |
| mu | 0.164248 | dimensionless |
| nu | 0.956481 | dimensionless |
| Q | -3.28506 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.46 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2946 | 0.1401 | 3.491 | 0.8221 |
| test | 1365 | 0.3049 | 0.1431 | 3.052 | 0.8219 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.4567
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.1540 (benzene sublimation ref ~ -0.456)
