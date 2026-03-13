#!/bin/bash
# Run AgentDojo baseline WITHOUT attacks (benign utility)
# Usage: bash scripts/run_baseline.sh
set -e
MODEL="gpt-4o-mini-2024-07-18"
SUITE="workspace"
LOGDIR="./results/baseline_no_attack"
TASKS="-ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_3 -ut user_task_4 -ut user_task_5 -ut user_task_6 -ut user_task_7 -ut user_task_8 -ut user_task_9"
echo "=== AgentDojo Baseline (No Attack) ==="
echo "Model:  $MODEL"
echo "Suite:  $SUITE"
echo "Logdir: $LOGDIR"
echo ""
python -m agentdojo.scripts.benchmark \
    -s "$SUITE" \
    $TASKS \
    --model "$MODEL" \
    --logdir "$LOGDIR"
echo ""
echo "=== Done! Results saved to $LOGDIR ==="
