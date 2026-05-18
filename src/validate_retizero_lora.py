"""
RetiZero LoRA 微调模型在 UWF CIR 数据集上的验证脚本。

使用方法:
    python validate_retizero_lora.py \
        --model-paths /path/to/run1/best_acc_*.pth /path/to/run2/best_acc_*.pth ... \
        --base-weight-path /path/to/RetiZero.pth \
        --output-csv results.csv

输出格式与 clip_fine_tune.py 生成的 validation_metrics.csv 完全一致:
    epoch, CH_recall_at1, ..., UM_recall_at10, average_recall_at1, ..., average_recall
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import torch
import numpy as np
from pathlib import Path
from statistics import mean
from argparse import ArgumentParser

from retizero_adapter import RetiZeroAdapter
from data_utils import FashionIQDataset, targetpad_transform, list_fashioniq_categories
from utils import extract_index_features, element_wise_sum, device
from validate import compute_fiq_val_metrics

# 动态获取数据集类别
CATEGORIES = list_fashioniq_categories("train")
if not CATEGORIES:
    CATEGORIES = ['CH', 'CO', 'NM', 'RB', 'RCH', 'UM', 'IDRiD']

# 默认使用 TRANSFORMERS 离线模式，避免网络问题
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_retizero_with_lora(base_weight_path: str, lora_checkpoint_path: str) -> RetiZeroAdapter:
    """
    加载 RetiZeroAdapter（base 权重），然后替换 vision encoder 为 LoRA 微调后的权重。

    LoRA checkpoint 来自 RetiZero/Finetuning.py 保存的 Model_Finetuing 模型，
    其 state_dict 包含:
      - img_encoder.*  → 对应 CLIPRModel.vision_model.model.*
      - classifier.*   → 分类头（CIR 验证时不需要）
    """
    # 1. 加载 base RetiZero（包含 vision + text 完整模型）
    print(f"  加载 base RetiZero: {base_weight_path}")
    adapter = RetiZeroAdapter(base_weight_path)

    # 2. 加载 LoRA 微调 checkpoint
    print(f"  加载 LoRA checkpoint: {lora_checkpoint_path}")
    ckpt = torch.load(lora_checkpoint_path, map_location='cpu')
    lora_state_dict = ckpt['state_dict']
    ckpt_epoch = ckpt.get('epoch', -1)
    ckpt_acc = ckpt.get('mean_ACC', -1)
    print(f"  Checkpoint info: epoch={ckpt_epoch}, val_acc={ckpt_acc:.4f}")

    # 3. 提取并重映射 vision encoder 权重
    #    img_encoder.xxx → vision_model.model.xxx
    vision_state_dict = {}
    skipped_keys = []
    for k, v in lora_state_dict.items():
        if k.startswith('img_encoder.'):
            new_key = k.replace('img_encoder.', '')
            vision_state_dict[new_key] = v
        else:
            skipped_keys.append(k)

    if skipped_keys:
        print(f"  跳过非 vision 权重: {len(skipped_keys)} keys (classifier 等)")

    # 4. 替换 vision model 权重（text model 保持 base 不变）
    load_result = adapter.retizero.vision_model.model.load_state_dict(vision_state_dict, strict=True)
    print(f"  LoRA vision weights loaded: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")

    return adapter, ckpt_epoch, ckpt_acc


def validate_single_model(model, categories, preprocess, combining_function):
    """
    对单个模型在所有类别上运行 CIR 验证，返回指标字典。
    """
    model.eval().float().to(device)

    results = {}
    recalls_at1, recalls_at5, recalls_at10 = [], [], []

    for cat in categories:
        print(f"\n  === 类别: {cat} ===")

        # 构建 gallery 数据集，提取 index features
        classic_val_dataset = FashionIQDataset('val', [cat], 'classic', preprocess)
        index_features, index_names = extract_index_features(classic_val_dataset, model)

        # 构建 relative 查询数据集
        relative_val_dataset = FashionIQDataset('val', [cat], 'relative', preprocess)

        # 计算 R@1, R@5, R@10
        r1, r5, r10 = compute_fiq_val_metrics(
            relative_val_dataset, model, index_features, index_names, combining_function
        )

        results[f'{cat}_recall_at1'] = r1
        results[f'{cat}_recall_at5'] = r5
        results[f'{cat}_recall_at10'] = r10
        recalls_at1.append(r1)
        recalls_at5.append(r5)
        recalls_at10.append(r10)

        print(f"  {cat}: R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}")

    # 计算平均指标
    avg_r1 = mean(recalls_at1)
    avg_r5 = mean(recalls_at5)
    avg_r10 = mean(recalls_at10)

    results['average_recall_at1'] = avg_r1
    results['average_recall_at5'] = avg_r5
    results['average_recall_at10'] = avg_r10
    results['average_recall'] = (avg_r1 + avg_r5 + avg_r10) / 3

    return results


def main():
    parser = ArgumentParser(description="在 UWF CIR 数据集上验证 RetiZero LoRA 微调模型")
    parser.add_argument("--model-paths", nargs='+', required=True,
                        help="LoRA checkpoint 路径列表 (支持多个)")
    parser.add_argument("--base-weight-path",
                        default="/data0/qrchen/projects/RetiZero/model/RetiZero.pth",
                        help="Base RetiZero 权重路径")
    parser.add_argument("--output-csv", default="retizero_lora_val_results.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--target-ratio", default=1.25, type=float,
                        help="TargetPad target ratio")

    args = parser.parse_args()

    # 图像预处理 (与 clip_fine_tune.py 一致)
    input_dim = 224  # RetiZero 输入分辨率
    preprocess = targetpad_transform(args.target_ratio, input_dim)
    print(f"预处理: TargetPad (ratio={args.target_ratio}, dim={input_dim})")

    combining_function = element_wise_sum

    # CSV 列定义 (与 validation_metrics.csv 格式完全一致)
    fieldnames = ['epoch']
    for cat in CATEGORIES:
        fieldnames.extend([f'{cat}_recall_at1', f'{cat}_recall_at5', f'{cat}_recall_at10'])
    fieldnames.extend(['average_recall_at1', 'average_recall_at5', 'average_recall_at10', 'average_recall'])

    all_results = []

    for i, model_path in enumerate(args.model_paths, 1):
        print(f"\n{'=' * 70}")
        print(f" 模型 {i}/{len(args.model_paths)}: {model_path}")
        print(f"{'=' * 70}")

        # 加载模型
        model, ckpt_epoch, ckpt_acc = load_retizero_with_lora(args.base_weight_path, model_path)

        # 运行验证
        results = validate_single_model(model, CATEGORIES, preprocess, combining_function)

        # 用 checkpoint epoch 作为 epoch 列 (若不存在则用运行编号)
        results['epoch'] = ckpt_epoch if ckpt_epoch >= 0 else i

        all_results.append(results)

        # 打印汇总
        print(f"\n  ---- 模型 {i} 汇总 ----")
        print(f"  分类 val acc (LoRA): {ckpt_acc:.4f}")
        print(f"  CIR avg R@1:  {results['average_recall_at1']:.4f}")
        print(f"  CIR avg R@5:  {results['average_recall_at5']:.4f}")
        print(f"  CIR avg R@10: {results['average_recall_at10']:.4f}")
        print(f"  CIR avg recall: {results['average_recall']:.4f}")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 写入 CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\n{'=' * 70}")
    print(f" 结果已保存至: {args.output_csv}")
    print(f"{'=' * 70}")

    # 跨模型统计摘要
    if len(all_results) > 1:
        print(f"\n  ===== {len(all_results)} 次运行的统计摘要 =====")
        summary_metrics = ['average_recall_at1', 'average_recall_at5', 'average_recall_at10', 'average_recall']
        for cat in CATEGORIES:
            summary_metrics.extend([f'{cat}_recall_at1', f'{cat}_recall_at5', f'{cat}_recall_at10'])

        for metric in summary_metrics:
            values = [r[metric] for r in all_results]
            print(f"  {metric:30s}: {np.mean(values):7.4f} ± {np.std(values):7.4f}")


if __name__ == '__main__':
    main()
