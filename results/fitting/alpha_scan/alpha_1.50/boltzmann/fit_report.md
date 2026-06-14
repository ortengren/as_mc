# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.004248060185775612

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.58547 | Angstrom |
| eps0 | 0.0260354 | eV |
| kappa | 0.580579 | dimensionless |
| kappa_prime | 1.0952 | dimensionless |
| mu | -1.16979 | dimensionless |
| nu | 2.11423 | dimensionless |
| Q | -3.75555 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.34 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.1963 | 0.1028 | 4.142 | 0.921 |
| test | 1365 | 0.2175 | 0.1107 | 2.984 | 0.9094 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3378
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2810 (benzene sublimation ref ~ -0.456)
