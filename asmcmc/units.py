"""Physical constants and unit conversions. The code works in eV, Å and K."""

# Boltzmann constant, eV/K. Rounded (CODATA: 8.617333e-5); every existing run
# used this value, so it is kept for comparability.
BOLTZCONST = 8.617e-5

# The sampler takes pressure in eV/Å^3.
ATM_TO_EV_PER_A3 = 6.324209e-7

# Energies per molecule.
EV_TO_KCAL = 23.060541945329334  # eV -> kcal/mol
KJ_PER_MOL_IN_EV = 0.01036410  # 1 kJ/mol in eV
EV_PER_K_TO_J_PER_MOL_K = 96485.33  # eV/K per molecule -> J/(mol K)

# h c / k_B in cm K: converts a vibrational wavenumber to a temperature.
HC_OVER_K = 1.438777
