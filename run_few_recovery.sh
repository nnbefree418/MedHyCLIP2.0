#!/bin/bash
set -euo pipefail

PYTHON="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python"
TAG="run_20260522"
COMMON="--use_hyperbolic --seed 111 --data_path ./data/ --tag $TAG --epoch 50 --patience 10 --shot 4 --iterate 0 --temperature 1.0"

cd /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP
mkdir -p logs ckpt/few-shot-hyper

run_one() {
  local gpu=$1
  local obj=$2
  echo "[$(date '+%H:%M:%S')] [RECOVERY][GPU${gpu}] start ${obj}"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON train_few.py --obj "$obj" $COMMON 2>&1 | tee "logs/few_train_${obj}_${TAG}.log"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON test_few.py --obj "$obj" $COMMON 2>&1 | tee "logs/few_test_${obj}_${TAG}.log"
  echo "[$(date '+%H:%M:%S')] [RECOVERY][GPU${gpu}] done ${obj}"
}

echo "===== Few-shot recovery start: $(date) ====="

# 两卡并行补跑失败数据集中的两个
run_one 2 Retina_RESC > logs/few_recovery_gpu2.log 2>&1 &
PID2=$!
run_one 3 Brain > logs/few_recovery_gpu3.log 2>&1 &
PID3=$!
wait $PID2
wait $PID3

# 再补跑剩余的 Liver（用 GPU2）
run_one 2 Liver | tee -a logs/few_recovery_gpu2.log

echo "===== Few-shot recovery done: $(date) ====="
