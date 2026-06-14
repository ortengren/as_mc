# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 0
- **objective**: 0.0022865491277049036

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.61573 | Angstrom |
| eps0 | 0.0247058 | eV |
| kappa | 0.583849 | dimensionless |
| kappa_prime | 2.07631 | dimensionless |
| mu | 0.0772192 | dimensionless |
| nu | 1.72015 | dimensionless |
| Q | -3.50959 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.37 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2297 | 0.1111 | 4.178 | 0.8919 |
| test | 1365 | 0.2433 | 0.1151 | 3.007 | 0.8866 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3683
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2523 (benzene sublimation ref ~ -0.456)
