# GBQ fit report

## Run
- **dataset**: data/xyz_files/ellipsoids_with_axes_and_energies.xyz
- **weighting**: boltzmann
- **cutoff_A**: 15.0
- **n_frames**: 6826
- **n_train**: 5461
- **n_test**: 1365
- **seed**: 1
- **objective**: 0.0022865288663728115

## Parameters
| parameter | value | unit |
| --- | --- | --- |
| sigma0 | 6.60963 | Angstrom |
| eps0 | 0.025006 | eV |
| kappa | 0.583745 | dimensionless |
| kappa_prime | 2.08105 | dimensionless |
| mu | 0.0766232 | dimensionless |
| nu | 1.71397 | dimensionless |
| Q | -3.51183 | (eV*Angstrom^5)^0.5 |
| E_intra | -1601.37 | eV/molecule |

## Metrics (eV/molecule)
| partition | n | rmse | mae | max_abs_err | r2 |
| --- | --- | --- | --- | --- | --- |
| train | 5461 | 0.2293 | 0.111 | 4.182 | 0.8923 |
| test | 1365 | 0.2429 | 0.115 | 3.009 | 0.8869 |

## Sanity checks (eV/molecule)
- inferred E_intra (isolated-molecule ref): -1601.3659
- mean target: -1601.0727
- min lattice energy (pred - E_intra): -0.2547 (benzene sublimation ref ~ -0.456)
