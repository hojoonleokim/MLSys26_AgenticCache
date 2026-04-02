#!/bin/bash
# =============================================================================
# CoELA — Smoke Test (minimal validation)
# Runs 1 branch (baseline) × 1 model (gpt-5) × 1 episode to verify setup.
# Requires: Xorg running on DISPLAY :1, OPENAI_API_KEY set
# Usage: ./scripts/smoke_test_coela.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-CoELA"

BRANCH=agenticcache
EPISODE=5  # episode outside cache set (cache = 1 2 3 4)

echo "========================================"
echo "CoELA — Smoke Test"
echo "Branch: $BRANCH | Model: gpt-5 | Episode: $EPISODE"
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
cd tdw_mat

port=1071
pkill -f -9 "port $port" 2>/dev/null || true
sleep 1

echo "Running CoELA smoke test..."
eval "$(conda shell.bash hook)"
conda activate coela
python3 tdw-gym/challenge.py \
    --output_dir results/smoke_test \
    --lm_id gpt-5-2025-08-07 \
    --experiment_name smoke \
    --run_id run_smoke \
    --port $port \
    --agents lm_agent lm_agent \
    --communication \
    --prompt_template_path LLM/prompt_com.txt \
    --max_tokens 256 \
    --cot \
    --data_prefix dataset/dataset_test/ \
    --eval_episodes $EPISODE \
    --screen_size 256 \
    --no_save_img
conda deactivate

pkill -f -9 "port $port" 2>/dev/null || true

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "CoELA smoke test completed successfully!"
echo "========================================"
