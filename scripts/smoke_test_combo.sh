#!/bin/bash
# =============================================================================
# COMBO — Smoke Test (minimal validation)
# Runs 1 branch (baseline) × 1 model (gpt-5) × 1 cook episode to verify setup.
# Requires: Xorg running on DISPLAY :1, OPENAI_API_KEY set
# Usage: ./scripts/smoke_test_combo.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-COMBO"

BRANCH=agenticcache
START_ID=2   # first eval episode (cache = cook 0-1)
NUM_RUNS=1   # single episode

echo "========================================"
echo "COMBO — Smoke Test"
echo "Branch: $BRANCH | Model: gpt-5 | Task: cook | Episode: $START_ID"
echo "========================================"

# Check prerequisites
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

if ! pgrep -x Xorg > /dev/null; then
    echo "ERROR: Xorg is not running."
    echo "Start it with: sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &"
    exit 1
fi

ORIGINAL_BRANCH=$(cd "$SUBMODULE_DIR" && git rev-parse --abbrev-ref HEAD)

cd "$SUBMODULE_DIR"
git checkout "$BRANCH"
cd tdw_maco

port=12077
pkill -f -9 "port\ $port" 2>/dev/null || true
sleep 1

echo "Running COMBO smoke test..."
eval "$(conda shell.bash hook)"
conda activate combo
python3 challenge.py \
    --port $port \
    --experiment_name smoke_test \
    --task cook \
    --run_id gpt5-smoke \
    --data_prefix dataset/ \
    --data_path test.json \
    --agents_algo combo_agent combo_agent \
    --screen_size 336 \
    --start_id $START_ID \
    --num_runs $NUM_RUNS \
    --max_steps 60 \
    --only_propose \
    --lm_source openai \
    --proposer_lm_id gpt-5-2025-08-07
conda deactivate

pkill -f -9 "port\ $port" 2>/dev/null || true

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "COMBO smoke test completed successfully!"
echo "========================================"
