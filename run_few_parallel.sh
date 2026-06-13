#!/bin/bash
# =============================================================================
# 少样本并行训练脚本
# 策略：4 张 GPU 各独立跑一个数据集（单卡单进程），并行完成 6 个数据集
#   第一批（并行）：Brain(GPU0) + Liver(GPU1) + Retina_RESC(GPU2) + Histopathology(GPU3)
#   第二批（并行）：Retina_OCT2017(GPU0) + Chest(GPU1)
# 错误自动重试（最多 3 次），失败后跳过继续下一个
# =============================================================================

PYTHON="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python"
TAG="run_20260522"
COMMON="--use_hyperbolic --seed 111 --data_path ./data/ --tag $TAG --epoch 50 --patience 10 --shot 4 --iterate 0 --temperature 1.0"

cd /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP
mkdir -p logs ckpt/few-shot-hyper

# =============================================================================
# 函数：在指定 GPU 上完成训练 + 测试，自动重试
# 用法：run_few GPU_ID DATASET_NAME
# =============================================================================
run_few() {
    local gpu=$1
    local obj=$2
    local max_retries=3
    local attempt=0

    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        echo "[$(date '+%H:%M:%S')] [GPU$gpu] [$obj] 开始训练 (attempt $attempt/$max_retries)"

        CUDA_VISIBLE_DEVICES=$gpu $PYTHON train_few.py --obj "$obj" $COMMON \
            2>&1 | tee "logs/few_train_${obj}_${TAG}.log"
        train_exit=${PIPESTATUS[0]}

        if [ $train_exit -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] [GPU$gpu] [$obj] 训练完成，开始测试"
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON test_few.py --obj "$obj" $COMMON \
                2>&1 | tee "logs/few_test_${obj}_${TAG}.log"
            echo "[$(date '+%H:%M:%S')] [GPU$gpu] [$obj] 全部完成"
            return 0
        else
            echo "[$(date '+%H:%M:%S')] [GPU$gpu] [$obj] 训练失败 (exit=$train_exit)，等待 10s 后重试..."
            sleep 10
        fi
    done

    echo "[$(date '+%H:%M:%S')] [GPU$gpu] [$obj] 已重试 $max_retries 次仍失败，跳过"
    return 1
}

echo "===== Few-shot 全量训练开始：$(date) ====="
echo ""

# =============================================================================
# 第一批：4 个数据集并行
# =============================================================================
echo "--- 第一批（并行）：Brain / Liver / Retina_RESC / Histopathology ---"
run_few 0 Brain        > logs/few_gpu0_batch1.log 2>&1 &
run_few 1 Liver        > logs/few_gpu1_batch1.log 2>&1 &
run_few 2 Retina_RESC  > logs/few_gpu2_batch1.log 2>&1 &
run_few 3 Histopathology > logs/few_gpu3_batch1.log 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] 第一批全部完成"
echo ""

# =============================================================================
# 第二批：2 个数据集并行
# =============================================================================
echo "--- 第二批（并行）：Retina_OCT2017 / Chest ---"
run_few 0 Retina_OCT2017 > logs/few_gpu0_batch2.log 2>&1 &
run_few 1 Chest          > logs/few_gpu1_batch2.log 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] 第二批全部完成"
echo ""

# =============================================================================
# 汇总结果
# =============================================================================
echo "===== Few-shot 全量训练结束：$(date) ====="
echo ""
echo "===== 性能汇总 ====="
for obj in Brain Liver Retina_RESC Histopathology Retina_OCT2017 Chest; do
    echo -n "[$obj] "
    grep -E "AUC|pAUC" "logs/few_test_${obj}_${TAG}.log" 2>/dev/null | tail -3 \
        || echo "(无结果)"
done
