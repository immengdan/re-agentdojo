#!/bin/bash
# Run AgentDojo baseline WITHOUT attacks (benign utility)
# Usage: bash scripts/run_baseline.sh
set -e
MODEL="GEMINI_2_5_FLASH"
#MODEL="GEMINI_1_5_FLASH"
SUITE="workspace"
LOGDIR="./results/baseline_no_attack"

# If no -ut/--user-task flags are provided, benchmark.py runs all available
# user tasks for the selected suite.
echo "=== AgentDojo Baseline (No Attack) ==="
echo "Model:  $MODEL"
echo "Suite:  $SUITE"
echo "Logdir: $LOGDIR"
echo ""
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m agentdojo.scripts.benchmark \
    -s "$SUITE" \
    --model "$MODEL" \
    --logdir "$LOGDIR"
echo ""
echo "=== Done! Results saved to $LOGDIR ==="
