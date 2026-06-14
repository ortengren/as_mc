# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.0017028934661313937

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.63268 | Angstrom |
| eps0 | 0.0247852 | eV |
| kappa | 0.58803 | dimensionless |
| kappa_prime | 2.76837 | dimensionless |
| mu | 0.122164 | dimensionless |
| nu | 0.994565 | dimensionless |
| Q | -3.41857 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.39 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2551 | 0.119 | 4.47 | 0.8666 |
| test | 1365 | 0.266 | 0.1225 | 3.215 | 0.8645 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3894
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2267 (benzene sublimation ref ~ -0.456)
