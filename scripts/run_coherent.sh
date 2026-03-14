#!/bin/bash
# =============================================================================
# Run COHERENT experiments across all branches
# Usage: ./scripts/run_coherent.sh
# Runs all envs (env0-env4) × all models (gpt-5, gpt-5-mini, gpt-5-nano)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-COHERENT"

BRANCHES=(baseline agenticcache parallel speculative)

echo "========================================"
echo "COHERENT — Full Experiment Run"
echo "Branches: ${BRANCHES[*]}"
echo "========================================"

ORIGINAL_BRANCH=$(cd "$SUBMODULE_DIR" && git rev-parse --abbrev-ref HEAD)

for branch in "${BRANCHES[@]}"; do
    echo ""
    echo "========================================"
    echo "Branch: $branch"
    echo "========================================"

    cd "$SUBMODULE_DIR"
    git checkout "$branch"
    cd src/experiment/PEFA

    conda run -n coherent --no-banner \
        bash scripts/run_all.sh

    echo "Branch $branch completed."
done

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "All COHERENT experiments completed!"
echo "========================================"
