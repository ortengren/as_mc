# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.0032697585997454154

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.57972 | Angstrom |
| eps0 | 0.0270637 | eV |
| kappa | 0.58029 | dimensionless |
| kappa_prime | 1.57219 | dimensionless |
| mu | 0.0472275 | dimensionless |
| nu | 1.83919 | dimensionless |
| Q | -3.63175 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.34 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2114 | 0.106 | 4.372 | 0.9084 |
| test | 1365 | 0.2276 | 0.1113 | 3.147 | 0.9008 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3400
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2814 (benzene sublimation ref ~ -0.456)
