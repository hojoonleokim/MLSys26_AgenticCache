#!/bin/bash
# =============================================================================
# COHERENT — Smoke Test (minimal validation)
# Runs 1 branch (agenticcache) × 1 model (gpt-5) × 1 env × 1 task to verify setup.
# Requires: OPENAI_API_KEY set (no Xorg needed)
# Usage: ./scripts/smoke_test_coherent.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-COHERENT"

BRANCH=agenticcache
ENV=env0
TASK=2  # single task outside cache set

echo "========================================"
echo "COHERENT — Smoke Test"
echo "Branch: $BRANCH | Model: gpt-5 | Env: $ENV | Task: $TASK"
echo "========================================"

# Check prerequisites
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

ORIGINAL_BRANCH=$(cd "$SUBMODULE_DIR" && git rev-parse --abbrev-ref HEAD)

cd "$SUBMODULE_DIR"
git checkout "$BRANCH"
cd src/experiment/PEFA

echo "Running COHERENT smoke test..."
eval "$(conda shell.bash hook)"
conda activate coherent
python main.py \
    --env "$ENV" \
    --task $TASK \
    --lm_id gpt-5-2025-08-07 \
    --branch smoke_test \
    --source openai
conda deactivate

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "COHERENT smoke test completed successfully!"
echo "========================================"
