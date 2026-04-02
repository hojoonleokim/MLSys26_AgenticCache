#!/bin/bash
# =============================================================================
# Run CoELA experiments across all branches
# Requires: Xorg running on DISPLAY :1
# Usage: ./scripts/run_coela.sh
# Cache episodes (excluded from eval): 1 2 3 4
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-CoELA"

BRANCHES=(baseline agenticcache parallel speculative)

echo "========================================"
echo "CoELA — Full Experiment Run"
echo "Branches: ${BRANCHES[*]}"
echo "Requires: DISPLAY=:1 (Xorg)"
echo "========================================"

# Verify Xorg is running
if ! pgrep -x Xorg > /dev/null; then
    echo "ERROR: Xorg is not running."
    echo "Start it with: sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &"
    exit 1
fi

ORIGINAL_BRANCH=$(cd "$SUBMODULE_DIR" && git rev-parse --abbrev-ref HEAD)

for branch in "${BRANCHES[@]}"; do
    echo ""
    echo "========================================"
    echo "Branch: $branch"
    echo "========================================"

    cd "$SUBMODULE_DIR"
    git checkout "$branch"
    cd tdw_mat

    eval "$(conda shell.bash hook)"
    conda activate coela
    bash scripts/test_2_LMs-gpt-5.sh
    conda deactivate

    echo "Branch $branch completed."
done

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "All CoELA experiments completed!"
echo "========================================"
