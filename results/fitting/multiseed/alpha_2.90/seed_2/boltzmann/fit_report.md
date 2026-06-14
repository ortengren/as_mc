# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 2
- **objective**: 0.002286558971458903

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.6108 | Angstrom |
| eps0 | 0.0249236 | eV |
| kappa | 0.583673 | dimensionless |
| kappa_prime | 2.06919 | dimensionless |
| mu | 0.077004 | dimensionless |
| nu | 1.72877 | dimensionless |
| Q | -3.50455 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.37 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2294 | 0.111 | 4.173 | 0.8922 |
| test | 1365 | 0.2431 | 0.115 | 3.003 | 0.8868 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3661
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2544 (benzene sublimation ref ~ -0.456)
