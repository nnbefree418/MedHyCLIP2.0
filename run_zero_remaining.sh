#!/bin/bash
# =============================================================================
# 零样本训练脚本：依次完成 5 个数据集，每次使用 4 张 GPU (DDP)
# 顺序：Liver → Histopathology → Retina_RESC → Brain → Chest
# 每次训练结束后自动运行对应测试，全程无需人工干预
# =============================================================================

PYTHON="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python"
TORCHRUN="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/torchrun"
TAG="run_20260522"
ARGS="--use_hyperbolic --seed 111 --data_path ./data/ --tag $TAG --epoch 50 --patience 10"
NGPU=4
PORT=29500

cd /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP
mkdir -p logs ckpt/zero-shot-hyper

DATASETS="Liver Histopathology Retina_RESC Brain Chest"

echo "===== Zero-shot 全量训练开始：$(date) ====="
echo "数据集顺序：$DATASETS"
echo ""

for obj in $DATASETS; do
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] 开始训练：$obj (4-GPU DDP)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN \
        --nproc_per_node=$NGPU \
        --master_port=$PORT \
        train_zero.py --obj "$obj" $ARGS \
        2>&1 | tee "logs/zero_train_${obj}_${TAG}.log"

    TRAIN_EXIT=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[ERROR] $obj 训练失败 (exit code $TRAIN_EXIT)，跳过测试，继续下一个数据集"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] 开始测试：$obj"
    CUDA_VISIBLE_DEVICES=0 $PYTHON test_zero.py --obj "$obj" $ARGS \
        2>&1 | tee "logs/zero_test_${obj}_${TAG}.log"

    echo "[$(date '+%H:%M:%S')] [DONE] $obj 完成"
    echo ""
done

echo "===== Zero-shot 全量训练结束：$(date) ====="
echo ""
echo "===== AUC 汇总 ====="
for obj in $DATASETS; do
    echo -n "[$obj] "
    grep -E "AUC" "logs/zero_test_${obj}_${TAG}.log" 2>/dev/null | tail -2 || echo "(无结果)"
done
