# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: uniform
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 1
- **objective**: 0.01487051534227465

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 7.07726 | Angstrom |
| eps0 | 0.00778526 | eV |
| kappa | 0.578682 | dimensionless |
| kappa_prime | 0.421098 | dimensionless |
| mu | -1.64837 | dimensionless |
| nu | 3.97488 | dimensionless |
| Q | -3.6121 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.45 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.1725 | 0.105 | 2.09 | 0.9391 |
| test | 1365 | 0.1888 | 0.111 | 1.416 | 0.9317 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.4539
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.1535 (benzene sublimation ref ~ -0.456)
