# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: uniform
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.014870568514841424

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 7.07031 | Angstrom |
| eps0 | 0.00788057 | eV |
| kappa | 0.578263 | dimensionless |
| kappa_prime | 0.420602 | dimensionless |
| mu | -1.68282 | dimensionless |
| nu | 3.97239 | dimensionless |
| Q | -3.59173 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.45 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.1725 | 0.105 | 2.092 | 0.9391 |
| test | 1365 | 0.1888 | 0.1111 | 1.417 | 0.9317 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.4516
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.1555 (benzene sublimation ref ~ -0.456)
