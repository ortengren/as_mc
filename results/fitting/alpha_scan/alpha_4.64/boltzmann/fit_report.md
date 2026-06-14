# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.001399341392244563

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.71181 | Angstrom |
| eps0 | 0.0202549 | eV |
| kappa | 0.592776 | dimensionless |
| kappa_prime | 2.95507 | dimensionless |
| mu | 0.135826 | dimensionless |
| nu | 0.987103 | dimensionless |
| Q | -3.32404 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.43 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2708 | 0.1269 | 4.022 | 0.8497 |
| test | 1365 | 0.2814 | 0.1302 | 2.896 | 0.8483 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.4252
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.1882 (benzene sublimation ref ~ -0.456)
