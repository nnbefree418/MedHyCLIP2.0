#!/bin/bash
# =============================================================================
# 全量实验一键运行脚本（DDP 版本）
#
# 策略：
#   Zero-shot：每个数据集串行，每次用全部 4 张 GPU 做 DDP 训练（4× 加速）
#   Few-shot ：每个数据集串行，在 GPU0 单卡运行（训练数据量小，单卡足够）
#
# 使用方式：
#   chmod +x run_all.sh
#   bash run_all.sh
# =============================================================================

set -euo pipefail

PYTHON="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python"
TORCHRUN="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/torchrun"
TAG="run_$(date +%Y%m%d)"
PATIENCE=10
MAX_EPOCHS=50
SHOT=4
SEED=111
DATA_PATH="./data/"
NGPU=4                              # DDP 使用的 GPU 数量
MASTER_PORT=29500                   # DDP 通信端口（同时只有一个 DDP 任务，端口不冲突）

DATASETS_ALL="Brain Liver Retina_RESC Retina_OCT2017 Chest Histopathology"

mkdir -p logs ckpt/zero-shot-hyper ckpt/few-shot-hyper

COMMON_HYPER="--use_hyperbolic --seed $SEED --data_path $DATA_PATH --tag $TAG"
COMMON_ZERO="$COMMON_HYPER --epoch $MAX_EPOCHS --patience $PATIENCE"
COMMON_FEW="$COMMON_HYPER --epoch $MAX_EPOCHS --patience $PATIENCE --shot $SHOT --iterate 0 --temperature 1.0"

# =============================================================================
# Zero-shot：每个数据集用 4-GPU DDP 训练，串行执行
# =============================================================================
echo "===== 实验开始：$(date) ====="
echo "TAG=$TAG  PATIENCE=$PATIENCE  MAX_EPOCHS=$MAX_EPOCHS  NGPU=$NGPU"

echo ""
echo "--- Zero-shot 阶段（DDP 4-GPU 串行）---"
for obj in $DATASETS_ALL; do
    echo "[$(date '+%H:%M:%S')] Zero-shot train (4-GPU DDP): $obj"
    CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN \
        --nproc_per_node=$NGPU \
        --master_port=$MASTER_PORT \
        train_zero.py --obj "$obj" $COMMON_ZERO \
        2>&1 | tee "logs/zero_train_${obj}_${TAG}.log"

    echo "[$(date '+%H:%M:%S')] Zero-shot test: $obj"
    CUDA_VISIBLE_DEVICES=0 $PYTHON test_zero.py \
        --obj "$obj" $COMMON_HYPER \
        2>&1 | tee "logs/zero_test_${obj}_${TAG}.log"
done
echo "[OK] Zero-shot 全部完成"

# =============================================================================
# Few-shot：数据量小，单卡串行即可
# =============================================================================
echo ""
echo "--- Few-shot 阶段（单卡 GPU0 串行）---"
for obj in $DATASETS_ALL; do
    echo "[$(date '+%H:%M:%S')] Few-shot train: $obj"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_few.py \
        --obj "$obj" $COMMON_FEW \
        2>&1 | tee "logs/few_train_${obj}_${TAG}.log"

    echo "[$(date '+%H:%M:%S')] Few-shot test: $obj"
    CUDA_VISIBLE_DEVICES=0 $PYTHON test_few.py \
        --obj "$obj" $COMMON_HYPER \
        --shot $SHOT --iterate 0 --temperature 1.0 \
        2>&1 | tee "logs/few_test_${obj}_${TAG}.log"
done
echo "[OK] Few-shot 全部完成"

echo ""
echo "===== 实验结束：$(date) ====="

# =============================================================================
# 自动汇总所有 AUC 指标到 logs/summary.txt
# =============================================================================
echo ""
echo "===== 生成指标汇总 ====="
SUMMARY="logs/summary_${TAG}.txt"
{
    echo "===== 指标汇总 TAG=${TAG} $(date) ====="
    echo ""
    echo "--- Zero-shot 结果 ---"
    for obj in $DATASETS_ALL; do
        logf="logs/zero_test_${obj}_${TAG}.log"
        if [ -f "$logf" ]; then
            echo "[$obj]"
            grep -E "AUC|pAUC" "$logf" | tail -5 || echo "  (无结果)"
        else
            echo "[$obj] 日志不存在: $logf"
        fi
    done
    echo ""
    echo "--- Few-shot (k=${SHOT}) 结果 ---"
    for obj in $DATASETS_ALL; do
        logf="logs/few_test_${obj}_${TAG}.log"
        if [ -f "$logf" ]; then
            echo "[$obj]"
            grep -E "AUC|pAUC" "$logf" | tail -5 || echo "  (无结果)"
        else
            echo "[$obj] 日志不存在: $logf"
        fi
    done
} | tee "$SUMMARY"

echo ""
echo "汇总文件：$SUMMARY"
