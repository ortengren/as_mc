#!/usr/bin/env bash
#
# Finalizing ("production") fit of the GB+quadrupole potential at the chosen
# weighting, repeated at three differential_evolution seeds. The train/test
# split is held FIXED (--split-seed 0) so all three runs optimize against the
# same partition -- agreement across the seeds is then a clean reproducibility
# check on the located minimum, not an artifact of different held-out frames.
#
# The weighting campaign is selected by environment variables:
#   WEIGHTING=uniform                (default) -> multiseed/uniform/seed_{n}
#   WEIGHTING=boltzmann ALPHA=2.90             -> multiseed/alpha_2.90/seed_{n}
# All runs share the one built-dataset cache under results/fitting/cache (keyed
# by file/cutoff/mtime, independent of weighting/alpha/seed), so the slow
# neighbour-list build is never repeated.
#
# Run with the `asmcmc` env active, or point $PYTHON at its interpreter:
#   PYTHON=~/.local/share/mamba/envs/asmcmc/bin/python ./scripts/run_fit_seeds.sh
#   WEIGHTING=boltzmann ALPHA=2.90 ./scripts/run_fit_seeds.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
WEIGHTING="${WEIGHTING:-uniform}"
ALPHA="${ALPHA:-2.90}"

# differential_evolution seeds to repeat the fit at (only --fit-seed varies).
FIT_SEEDS=(0 1 2)

# Production differential_evolution settings: a larger population for broader
# exploration, a lower maxiter ceiling, and a tighter convergence tolerance than
# the alpha sweep used. -1 workers uses every core.
DE_OPTS=(--workers -1 --popsize 60 --maxiter 300 --tol 1e-4)

CACHE_DIR="results/fitting/cache"

# Per-weighting campaign dir + the weighting-specific run flags.
if [ "$WEIGHTING" = "uniform" ]; then
    CAMPAIGN="uniform"
    WEIGHT_OPTS=(--weighting uniform)
    echo "=== uniform finalisation, 3 seeds ==="
else
    CAMPAIGN="alpha_${ALPHA}"
    WEIGHT_OPTS=(--weighting boltzmann --alpha "$ALPHA")
    echo "=== boltzmann (alpha=${ALPHA}) finalisation, 3 seeds ==="
fi
CAMPAIGN_DIR="results/fitting/multiseed/${CAMPAIGN}"

for s in "${FIT_SEEDS[@]}"; do
    out_dir="${CAMPAIGN_DIR}/seed_${s}"
    echo "--- ${WEIGHTING} fit, fit-seed=${s} -> ${out_dir} ---"
    "$PYTHON" -m asmcmc.fitting.run \
        "${WEIGHT_OPTS[@]}" \
        --split-seed 0 --fit-seed "$s" \
        "${DE_OPTS[@]}" \
        --cache-dir "$CACHE_DIR" \
        --out-dir "$out_dir"
done

echo "Done. Compare the per-run params/metrics under ${CAMPAIGN_DIR}/seed_{0,1,2}/."
