#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export GENERATOR_BASE_URL="https://openrouter.ai/api/v1"
export GENERATOR_API_KEY="${GENERATOR_API_KEY:?Set GENERATOR_API_KEY env variable}"
export JUDGE_MODEL="openai/gpt-5.4"
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="$GENERATOR_API_KEY"
export JUDGE_ENABLED="true"

OUTPUT_DIR="results_multiframe"
BATCH_SIZE=20
MAX_TURNS=10
MAX_TOKENS=64000
LOG_DIR="logs/benchmark_runs"
PARALLEL=5

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

MODELS=(
    "qwen/qwen3-8b"
    "qwen/qwen3-14b"
    "google/gemini-2.5-flash"
    "deepseek/deepseek-v3.2"
    "qwen/qwen3-coder-30b-a3b-instruct"
    "qwen/qwen3-235b-a22b-2507"
    "x-ai/grok-4.1-fast"
    "google/gemini-3-flash-preview"
    "z-ai/glm-5"
    "moonshotai/kimi-k2.5"
    "anthropic/claude-sonnet-4.6"
    "openai/gpt-5.2-codex"
    "anthropic/claude-sonnet-4.5"
    "openai/gpt-5.4"
    "anthropic/claude-opus-4.6"
)

run_model() {
    local model_id="$1"
    local safe_name="${model_id//\//_}"
    local log="${LOG_DIR}/${safe_name}.log"

    echo "[$(date '+%H:%M:%S')] START $model_id" | tee -a "$log"

    echo "=== single-shot ===" >> "$log"
    python run_single_shot.py \
        --model "$model_id" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --save-trajectories \
        --max-tokens "$MAX_TOKENS" \
        >> "$log" 2>&1

    echo "=== agentic ===" >> "$log"
    python run_agentic.py \
        --model "$model_id" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --save-trajectories \
        --max-turns "$MAX_TURNS" \
        --max-tokens "$MAX_TOKENS" \
        >> "$log" 2>&1

    echo "[$(date '+%H:%M:%S')] DONE  $model_id" | tee -a "$log"
}

echo "================================================"
echo " QuantCode-Bench: Full Benchmark (${#MODELS[@]} models)"
echo " Start: $(date)"
echo " Output: $OUTPUT_DIR"
echo " Judge:  $JUDGE_MODEL"
echo " Batch:  $BATCH_SIZE  |  Max turns: $MAX_TURNS"
echo " Max tokens: $MAX_TOKENS"
echo " Parallel: $PARALLEL models at a time"
echo "================================================"

running=0
pids=()
model_for_pid=()

for model in "${MODELS[@]}"; do
    run_model "$model" &
    pids+=($!)
    model_for_pid+=("$model")
    ((running++)) || true

    if (( running >= PARALLEL )); then
        # Wait for any one to finish
        wait -n "${pids[@]}" 2>/dev/null || true
        # Reap finished
        new_pids=()
        new_models=()
        for i in "${!pids[@]}"; do
            if kill -0 "${pids[$i]}" 2>/dev/null; then
                new_pids+=("${pids[$i]}")
                new_models+=("${model_for_pid[$i]}")
            else
                echo "[$(date '+%H:%M:%S')] REAPED ${model_for_pid[$i]} (pid ${pids[$i]})"
            fi
        done
        pids=("${new_pids[@]}")
        model_for_pid=("${new_models[@]}")
        running=${#pids[@]}
    fi
done

# Wait for remaining
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] REAPED ${model_for_pid[$i]} (pid ${pids[$i]})"
done

echo ""
echo "================================================"
echo " ALL MODELS COMPLETE"
echo " End: $(date)"
echo "================================================"
