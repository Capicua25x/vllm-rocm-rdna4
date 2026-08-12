#!/bin/bash
# Apples-to-apples sweep for the Gemma dev env (matches prod knobs). $1=label, $2=model(gemma|qwen)
set -uo pipefail
LABEL="${1:-gemma}"; MODEL="${2:-gemma}"
BENCH="${BENCH:-<path>/throughput_sweep.sh}"
acc() { curl -s localhost:8011/metrics | grep -E "vllm:spec_decode_num_(draft|accepted)_tokens_total\{" | awk '{print $2}' | paste -sd'/' ; }
echo "############ A2A MEASURE $LABEL ($MODEL) — $(date) ############"
echo "=== correctness ==="
curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23?\"}],\"max_tokens\":400,\"temperature\":0}" \
  | jq -r '.choices[0].message | (.content // .reasoning_content // "null")' | tail -2
echo "=== acc before ==="; acc
echo "=== NORMAL sweep (1 16 32 64) ==="
"$BENCH" --model "$MODEL" --levels "1 16 32 64" 2>&1 | sed -n '/users | per-user/,/Practical/p'
echo "=== acc mid ==="; acc
echo "=== 6K sweep (1 16 32 64) ==="
"$BENCH" --model "$MODEL" --prompt-tokens 6000 --levels "1 16 32 64" 2>&1 | sed -n '/users | per-user/,/Practical/p'
echo "=== acc final ==="; acc
echo "############ A2A MEASURE $LABEL DONE — $(date) ############"
