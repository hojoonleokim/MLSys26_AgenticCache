#!/bin/bash
# =============================================================================
# Run COMBO experiments across all branches
# Requires: Xorg running on DISPLAY :1
# Usage: ./scripts/run_combo.sh [TASK]
#   TASK : cook | game | all  (default: all)
# Cache episodes (excluded): cook 0-1, game 0
# Eval episodes: cook 2-17 (16 episodes), game 1-8 (8 episodes)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-COMBO"

TASK=${1:-all}
START_ID_COOK=2
NUM_RUNS_COOK=16
START_ID_GAME=1
NUM_RUNS_GAME=8

BRANCHES=(baseline agenticcache parallel speculative)

echo "========================================"
echo "COMBO — Full Experiment Run"
echo "Task: $TASK"
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
    cd tdw_maco

    if [ "$TASK" == "cook" ] || [ "$TASK" == "all" ]; then
        echo "  Running cook task..."
        conda run -n combo --no-banner \
            bash scripts/run_gpt5_all.sh cook "$START_ID_COOK" "$NUM_RUNS_COOK"
        echo "  Done: cook"
    fi

    if [ "$TASK" == "game" ] || [ "$TASK" == "all" ]; then
        echo "  Running game task..."
        conda run -n combo --no-banner \
            bash scripts/run_gpt5_all.sh game "$START_ID_GAME" "$NUM_RUNS_GAME"
        echo "  Done: game"
    fi

    echo "Branch $branch completed."
done

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "All COMBO experiments completed!"
echo "========================================"
