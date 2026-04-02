#!/bin/bash
# =============================================================================
# Run COMBO training pipeline (prerequisite for evaluation)
# Requires: Xorg running on DISPLAY :1
# Usage: ./scripts/run_combo_train.sh [--step N]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/MLSys26_AgenticCache-COMBO"

echo "========================================"
echo "COMBO — Training Pipeline"
echo "Branch: training-code"
echo "Requires: DISPLAY=:1 (Xorg)"
echo "========================================"

# Verify Xorg is running
if ! pgrep -x Xorg > /dev/null; then
    echo "ERROR: Xorg is not running."
    echo "Start it with: sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf &"
    exit 1
fi

ORIGINAL_BRANCH=$(cd "$SUBMODULE_DIR" && git rev-parse --abbrev-ref HEAD)

cd "$SUBMODULE_DIR"
git checkout training-code
cd AVDC/flowdiffusion

echo ""
echo "Starting training pipeline..."
echo ""

eval "$(conda shell.bash hook)"
conda activate combo
bash train_all.sh "$@"
conda deactivate

echo ""
echo "Training completed."

# Restore original branch
cd "$SUBMODULE_DIR"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "========================================"
echo "COMBO training pipeline completed!"
echo "Output: AVDC/results/tdw_maco_inpainting/modl-100.pt"
echo "========================================"
