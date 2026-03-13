#!/bin/bash
# Run AgentDojo with Sanitization defense AND tool_knowledge attack
# Usage: bash scripts/run_config_4.sh
set -e
MODEL="gemini-1.5-flash-001"
SUITE="workspace"
ATTACK="tool_knowledge"
DEFENSE="sanitization"
LOGDIR="./results/config_4_sanitization"
TASKS="-ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_3 -ut user_task_4 -ut user_task_5 -ut user_task_6 -ut user_task_7 -ut user_task_8 -ut user_task_9"
echo "=== AgentDojo Config 4 (Defense: $DEFENSE, Attack: $ATTACK) ==="
echo "Model:   $MODEL"
echo "Suite:   $SUITE"
echo "Defense: $DEFENSE"
echo "Attack:  $ATTACK"
echo "Logdir:  $LOGDIR"
echo ""

python -m agentdojo.scripts.benchmark \
    -s "$SUITE" \
    $TASKS \
    --model "$MODEL" \
    --attack "$ATTACK" \
    --defense "$DEFENSE" \
    --logdir "$LOGDIR"
echo ""
echo "=== Done! Results saved to $LOGDIR ==="
