#!/bin/bash
# Run AgentDojo baseline WITH tool_knowledge attack (security evaluation)
# Usage: bash scripts/run_attack.sh
set -e
MODEL="gemini-1.5-flash"
SUITE="workspace"
ATTACK="tool_knowledge"
LOGDIR="./results/baseline_with_attack"
TASKS="-ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_3 -ut user_task_4 -ut user_task_5 -ut user_task_6 -ut user_task_7 -ut user_task_8 -ut user_task_9"
echo "=== AgentDojo Baseline (With Attack: $ATTACK) ==="
echo "Model:  $MODEL"
echo "Suite:  $SUITE"
echo "Attack: $ATTACK"
echo "Logdir: $LOGDIR"
echo ""
python -m agentdojo.scripts.benchmark \
    -s "$SUITE" \
    $TASKS \
    --model "$MODEL" \
    --attack "$ATTACK" \
    --logdir "$LOGDIR"
echo ""
echo "=== Done! Results saved to $LOGDIR ==="
