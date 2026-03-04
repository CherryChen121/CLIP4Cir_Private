#!/bin/bash
# ===================================================
# 使用 5 个 RetiZero LoRA 微调模型在 UWF CIR 验证集上评估
# ===================================================

# ---- 路径配置 ----
BASE_WEIGHT="/data0/qrchen/projects/RetiZero/model/RetiZero.pth"
MODEL_DIR="/data0/qrchen/projects/RetiZero/Model_saved"
OUTPUT_CSV="/data0/qrchen/projects/CLIP4Cir/models/retizero_lora_val_results.csv"

# ---- 避免网络请求 ----
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- 自动收集 5 个 best 模型路径 ----
MODEL_PATHS=""
for run_dir in $(ls -d ${MODEL_DIR}/run_20260228_*_run{1,2,3,4,5} 2>/dev/null | sort); do
    best_pth=$(ls ${run_dir}/best_acc_*.pth 2>/dev/null | head -1)
    if [ -n "$best_pth" ]; then
        MODEL_PATHS="${MODEL_PATHS} ${best_pth}"
        echo "找到模型: ${best_pth}"
    fi
done

if [ -z "$MODEL_PATHS" ]; then
    echo "错误: 未在 ${MODEL_DIR} 中找到任何模型文件"
    exit 1
fi

echo ""
echo "共找到 $(echo $MODEL_PATHS | wc -w) 个模型，开始验证..."
echo "输出 CSV: ${OUTPUT_CSV}"
echo ""

# ---- 运行验证 ----
cd /data0/qrchen/projects/CLIP4Cir/src

conda run -n clip4cir python validate_retizero_lora.py \
    --model-paths ${MODEL_PATHS} \
    --base-weight-path ${BASE_WEIGHT} \
    --output-csv ${OUTPUT_CSV} \
    --target-ratio 1.25
