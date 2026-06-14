#!/usr/bin/env bash
#
# Fit the GB+quadrupole potential at a list of Boltzmann weight scales alpha
# (alpha = 1/(k_B*T), in 1/eV), passed as positional arguments. The special
# value 0 is the unweighted (uniform) reference -- exp(-0*E) = 1 for every
# frame. Each run writes to its own results/fitting/alpha_scan/ subdirectory in
# the shared output format, all sharing one built-dataset cache. Pass only the
# alphas you still need; existing runs are not touched, so nothing already on
# disk is recomputed.
#
# Usage:
#   ./scripts/run_fits.sh [ALPHA ...]
# Examples:
#   ./scripts/run_fits.sh                # default sweep (below) + uniform
#   ./scripts/run_fits.sh 1.5 2.0        # just two new weighted fits
#   ./scripts/run_fits.sh 0              # just the uniform reference
#
# alpha <-> T (LOWER alpha is HIGHER T): 5.80/4.64/3.87/3.32/2.90 ~
# 2000/2500/3000/3500/4000 K; 2.0/1.5 ~ 5800/7700 K resolve the knee in the
# unsampled gap below 2.90.
#
# Run with the `asmcmc` env active, or point $PYTHON at its interpreter:
#   PYTHON=~/.local/share/mamba/envs/asmcmc/bin/python ./scripts/run_fits.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"

# Default to the full canonical sweep + uniform (0) when no alphas are given.
if [ "$#" -gt 0 ]; then
    ALPHAS=("$@")
else
    ALPHAS=(5.80 4.64 3.87 3.32 2.90 0)
fi

# Shared differential_evolution settings (see README "Fit the GB+quadrupole
# potential"). -1 workers uses every core.
DE_OPTS=(--workers -1 --popsize 22 --maxiter 250 --tol 1e-3)

# All sweep runs share one built-dataset cache (keyed by file/cutoff/mtime, so
# independent of alpha/weighting): the slow neighbour-list build happens once.
SCAN_DIR="results/fitting/alpha_scan"
CACHE_DIR="results/fitting/cache"

for a in "${ALPHAS[@]}"; do
    if [ "$(awk -v a="$a" 'BEGIN{print (a==0)?1:0}')" -eq 1 ]; then
        # Unweighted reference: uniform weighting (alpha is ignored here).
        echo "=== uniform (unweighted) fit ==="
        "$PYTHON" -m asmcmc.fitting.run \
            --weighting uniform \
            "${DE_OPTS[@]}" \
            --cache-dir "$CACHE_DIR" \
            --out-dir "${SCAN_DIR}/uniform"
    else
        echo "=== boltzmann fit, alpha=${a} ==="
        "$PYTHON" -m asmcmc.fitting.run \
            --weighting boltzmann --alpha "$a" \
            "${DE_OPTS[@]}" \
            --cache-dir "$CACHE_DIR" \
            --out-dir "${SCAN_DIR}/alpha_${a}"
    fi
done

echo "Done. Compare the per-run comparison.csv / metrics.json under ${SCAN_DIR}/."
