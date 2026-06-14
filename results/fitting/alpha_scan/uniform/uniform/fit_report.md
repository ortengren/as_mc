# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: uniform
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.01654061448049711

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 7.21106 | Angstrom |
| eps0 | 0.00792761 | eV |
| kappa | 0.531389 | dimensionless |
| kappa_prime | 0.161984 | dimensionless |
| mu | 0.687331 | dimensionless |
| nu | 1 | dimensionless |
| Q | -4.32917 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.4 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.1819 | 0.1087 | 3.739 | 0.9322 |
| test | 1365 | 0.1947 | 0.1141 | 2.704 | 0.9274 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3956
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2259 (benzene sublimation ref ~ -0.456)
