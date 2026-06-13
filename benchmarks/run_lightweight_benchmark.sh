#!/usr/bin/env bash
# Lightweight Euclidean-vs-hyperbolic benchmark harness.
# This script intentionally wraps the existing train/test entrypoints without
# modifying them, so timing and memory can be collected non-invasively.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python}"
TORCHRUN="${TORCHRUN:-/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/torchrun}"

if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="torchrun"
fi

ZERO_DATASET="${ZERO_DATASET:-Liver}"
FEW_DATASET="${FEW_DATASET:-Retina_OCT2017}"
DATA_PATH="${DATA_PATH:-./data/}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-999}"
SHOT="${SHOT:-4}"
ITERATE="${ITERATE:-0}"
TEMPERATURE_FEW="${TEMPERATURE_FEW:-1.0}"
SEEDS="${SEEDS:-111 222 333}"
NGPU="${NGPU:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MEM_INTERVAL="${MEM_INTERVAL:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
RUN_ID="${RUN_ID:-lightweight_$(date +%Y%m%d_%H%M%S)}"

RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}"
LOG_DIR="${RESULT_DIR}/logs"
MEM_DIR="${RESULT_DIR}/memory"
MANIFEST="${RESULT_DIR}/manifest.tsv"

mkdir -p "$LOG_DIR" "$MEM_DIR"

cd "$REPO_ROOT"

echo -e "task\tmethod\tseed\tphase\tdataset\ttag\tgpu_ids\tlog_path\tmemory_path\tstart_epoch\tend_epoch\tstatus" > "$MANIFEST"

timestamp_stream() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date -Is)" "$line"
    done
}

start_memory_monitor() {
    local gpu_ids="$1"
    local memory_path="$2"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "timestamp,index,memory.used [MiB]" > "$memory_path"
        echo "nvidia-smi not found" >> "$memory_path"
        return 0
    fi

    {
        echo "timestamp,index,memory.used [MiB]"
        while true; do
            nvidia-smi \
                --id="$gpu_ids" \
                --query-gpu=timestamp,index,memory.used \
                --format=csv,noheader,nounits || true
            sleep "$MEM_INTERVAL"
        done
    } >> "$memory_path" &
    echo "$!"
}

stop_memory_monitor() {
    local pid="${1:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    fi
}

append_manifest() {
    local task="$1"
    local method="$2"
    local seed="$3"
    local phase="$4"
    local dataset="$5"
    local tag="$6"
    local gpu_ids="$7"
    local log_path="$8"
    local memory_path="$9"
    local start_epoch="${10}"
    local end_epoch="${11}"
    local status="${12}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$task" "$method" "$seed" "$phase" "$dataset" "$tag" "$gpu_ids" \
        "$log_path" "$memory_path" "$start_epoch" "$end_epoch" "$status" >> "$MANIFEST"
}

run_logged() {
    local task="$1"
    local method="$2"
    local seed="$3"
    local phase="$4"
    local dataset="$5"
    local tag="$6"
    local gpu_ids="$7"
    local log_path="$8"
    local memory_path="$9"
    shift 9

    local start_epoch end_epoch status monitor_pid
    start_epoch="$(date +%s)"
    monitor_pid="$(start_memory_monitor "$gpu_ids" "$memory_path")"

    {
        echo "BENCH_RUN_START task=${task} method=${method} seed=${seed} phase=${phase} dataset=${dataset} tag=${tag} gpu_ids=${gpu_ids} start_epoch=${start_epoch}"
        echo "BENCH_COMMAND $*"
    } | timestamp_stream | tee "$log_path"

    set +e
    "$@" 2>&1 | timestamp_stream | tee -a "$log_path"
    status="${PIPESTATUS[0]}"
    set -e

    end_epoch="$(date +%s)"
    stop_memory_monitor "$monitor_pid"

    {
        echo "BENCH_RUN_END task=${task} method=${method} seed=${seed} phase=${phase} dataset=${dataset} tag=${tag} gpu_ids=${gpu_ids} end_epoch=${end_epoch} status=${status}"
    } | timestamp_stream | tee -a "$log_path"

    append_manifest "$task" "$method" "$seed" "$phase" "$dataset" "$tag" "$gpu_ids" "$log_path" "$memory_path" "$start_epoch" "$end_epoch" "$status"
    return "$status"
}

method_flag() {
    local method="$1"
    if [[ "$method" == "hyper" ]]; then
        echo "--use_hyperbolic"
    fi
}

run_zero_pair() {
    local method="$1"
    local seed="$2"
    local port="$3"
    local extra_flag
    extra_flag="$(method_flag "$method")"
    local tag="bench_zero_${method}_seed${seed}_${RUN_ID}"

    if ! run_logged "zero" "$method" "$seed" "train" "$ZERO_DATASET" "$tag" "$GPU_IDS" \
        "${LOG_DIR}/zero_${method}_seed${seed}_train.log" \
        "${MEM_DIR}/zero_${method}_seed${seed}_train.csv" \
        env CUDA_VISIBLE_DEVICES="$GPU_IDS" "$TORCHRUN" \
            --nproc_per_node="$NGPU" \
            --master_port="$port" \
            train_zero.py \
            --obj "$ZERO_DATASET" \
            --seed "$seed" \
            --data_path "$DATA_PATH" \
            --tag "$tag" \
            --epoch "$EPOCHS" \
            --patience "$PATIENCE" \
            $extra_flag; then
        echo "Zero-shot train failed for method=${method} seed=${seed}; skipping matching test."
        return 0
    fi

    run_logged "zero" "$method" "$seed" "test" "$ZERO_DATASET" "$tag" "0" \
        "${LOG_DIR}/zero_${method}_seed${seed}_test.log" \
        "${MEM_DIR}/zero_${method}_seed${seed}_test.csv" \
        env CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
            test_zero.py \
            --obj "$ZERO_DATASET" \
            --seed "$seed" \
            --data_path "$DATA_PATH" \
            --tag "$tag" \
            $extra_flag || true
}

run_few_pair() {
    local method="$1"
    local seed="$2"
    local gpu="$3"
    local extra_flag
    extra_flag="$(method_flag "$method")"
    local tag="bench_few_${method}_seed${seed}_${RUN_ID}"

    if ! run_logged "few" "$method" "$seed" "train" "$FEW_DATASET" "$tag" "$gpu" \
        "${LOG_DIR}/few_${method}_seed${seed}_train.log" \
        "${MEM_DIR}/few_${method}_seed${seed}_train.csv" \
        env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
            train_few.py \
            --obj "$FEW_DATASET" \
            --seed "$seed" \
            --data_path "$DATA_PATH" \
            --tag "$tag" \
            --epoch "$EPOCHS" \
            --patience "$PATIENCE" \
            --shot "$SHOT" \
            --iterate "$ITERATE" \
            --temperature "$TEMPERATURE_FEW" \
            $extra_flag; then
        echo "Few-shot train failed for method=${method} seed=${seed}; skipping matching test."
        return 0
    fi

    run_logged "few" "$method" "$seed" "test" "$FEW_DATASET" "$tag" "$gpu" \
        "${LOG_DIR}/few_${method}_seed${seed}_test.log" \
        "${MEM_DIR}/few_${method}_seed${seed}_test.csv" \
        env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
            test_few.py \
            --obj "$FEW_DATASET" \
            --seed "$seed" \
            --data_path "$DATA_PATH" \
            --tag "$tag" \
            --shot "$SHOT" \
            --iterate "$ITERATE" \
            --temperature "$TEMPERATURE_FEW" \
            $extra_flag || true
}

echo "===== Lightweight benchmark start: $(date -Is) ====="
echo "RESULT_DIR=${RESULT_DIR}"
echo "ZERO_DATASET=${ZERO_DATASET} FEW_DATASET=${FEW_DATASET} EPOCHS=${EPOCHS} SEEDS=${SEEDS}"

run_index=0
for method in euclid hyper; do
    for seed in $SEEDS; do
        port=$((MASTER_PORT_BASE + run_index))
        echo "===== Zero-shot ${method} seed=${seed} port=${port} ====="
        run_zero_pair "$method" "$seed" "$port"
        run_index=$((run_index + 1))
    done
done

echo "===== Few-shot runs start: $(date -Is) ====="
active=0
gpu_array=(${GPU_IDS//,/ })
gpu_count="${#gpu_array[@]}"

for method in euclid hyper; do
    for seed in $SEEDS; do
        gpu="${gpu_array[$((active % gpu_count))]}"
        echo "===== Few-shot ${method} seed=${seed} gpu=${gpu} ====="
        run_few_pair "$method" "$seed" "$gpu" &
        active=$((active + 1))
        if (( active % gpu_count == 0 )); then
            wait || true
        fi
    done
done
wait || true

echo "===== Collecting benchmark summary: $(date -Is) ====="
"$PYTHON" "${SCRIPT_DIR}/collect_lightweight_benchmark.py" "$RESULT_DIR"

echo "===== Lightweight benchmark done: $(date -Is) ====="
echo "Summary: ${RESULT_DIR}/benchmark_summary.md"
