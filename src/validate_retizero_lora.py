"""
RetiZero LoRA 微调模型在 FashionIQ 格式 CIR 数据集上的验证脚本。

使用方法:
    python validate_retizero_lora.py \
        --model-paths /path/to/run1/best_acc_*.pth /path/to/run2/best_acc_*.pth ... \
        --base-weight-path /path/to/RetiZero.pth \
        --fashioniq-root /path/to/dataset \
        --dress-types CH CO NM RB RCH UM \
        --output-dataset uwf

结果统一写入 outputs/<实际数据集>/evaluation/retizero-lora/<run-id>/。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
from statistics import mean
from argparse import ArgumentParser, ArgumentTypeError
import traceback

from retizero_adapter import RetiZeroAdapter
from data_utils import (
    FashionIQDataset,
    list_fashioniq_categories,
    resolve_fashioniq_root,
    targetpad_transform,
)
from dataset_identity import resolve_fashioniq_evaluation_identity
from evaluation_outputs import (
    create_evaluation_layout,
    discard_evaluation_metrics,
    finalize_evaluation,
    publish_evaluation_metrics,
    tee_evaluation_output,
    validate_metrics_csv_filename,
)
from utils import extract_index_features, element_wise_sum, device
from validate import compute_fiq_val_metrics

# 默认使用 TRANSFORMERS 离线模式，避免网络问题
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _parse_output_csv_filename(value):
    try:
        return validate_metrics_csv_filename(value)
    except ValueError as error:
        raise ArgumentTypeError(str(error)) from error


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


def validate_single_model(
    model,
    categories,
    preprocess,
    combining_function,
    *,
    split,
    dataset_root,
):
    """
    对单个模型在所有类别上运行 CIR 验证，返回指标字典。
    """
    model.eval().float().to(device)

    results = {}
    recalls_at1, recalls_at5, recalls_at10 = [], [], []

    for cat in categories:
        print(f"\n  === 类别: {cat} ===")

        # 构建 gallery 数据集，提取 index features
        classic_val_dataset = FashionIQDataset(
            split,
            [cat],
            'classic',
            preprocess,
            dataset_root=dataset_root,
        )
        index_features, index_names = extract_index_features(classic_val_dataset, model)

        # 构建 relative 查询数据集
        relative_val_dataset = FashionIQDataset(
            split,
            [cat],
            'relative',
            preprocess,
            dataset_root=dataset_root,
            return_target=(split == "test"),
        )

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


def build_parser():
    parser = ArgumentParser(
        description="在 FashionIQ 格式 CIR 数据集上验证 RetiZero LoRA 模型"
    )
    parser.add_argument("--model-paths", nargs='+', required=True,
                        help="LoRA checkpoint 路径列表 (支持多个)")
    parser.add_argument("--base-weight-path",
                        default="/data0/qrchen/projects/RetiZero/model/RetiZero.pth",
                        help="Base RetiZero 权重路径")
    parser.add_argument(
        "--output-csv",
        type=_parse_output_csv_filename,
        default="evaluation_metrics.csv",
        help="评估运行目录内的 CSV 文件名",
    )
    parser.add_argument("--target-ratio", default=1.25, type=float,
                        help="TargetPad target ratio")
    parser.add_argument("--fashioniq-root", default=None)
    parser.add_argument("--dress-types", nargs="+", default=None)
    parser.add_argument(
        "--fashioniq-split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-dataset", default=None)
    parser.add_argument(
        "--evaluation-name",
        default="retizero-lora-validation",
    )
    return parser


def _canonical_checkpoint_result(
    *,
    model_path,
    sequence,
    checkpoint_epoch,
    checkpoint_accuracy,
    categories,
    metrics,
):
    aggregate = {
        "average_recall_at1": metrics["average_recall_at1"],
        "average_recall_at5": metrics["average_recall_at5"],
        "average_recall_at10": metrics["average_recall_at10"],
        "average_recall": metrics["average_recall"],
    }
    return {
        "model_path": str(model_path),
        "epoch": (
            checkpoint_epoch
            if checkpoint_epoch >= 0
            else sequence
        ),
        "classification_val_accuracy": checkpoint_accuracy,
        "per_category": {
            category: {
                "recall_at1": metrics[f"{category}_recall_at1"],
                "recall_at5": metrics[f"{category}_recall_at5"],
                "recall_at10": metrics[f"{category}_recall_at10"],
            }
            for category in categories
        },
        "aggregate": aggregate,
    }


def _run_evaluation(args, categories):
    # 图像预处理 (与 clip_fine_tune.py 一致)
    input_dim = 224  # RetiZero 输入分辨率
    preprocess = targetpad_transform(args.target_ratio, input_dim)
    print(f"预处理: TargetPad (ratio={args.target_ratio}, dim={input_dim})")

    combining_function = element_wise_sum
    all_results = []

    for i, model_path in enumerate(args.model_paths, 1):
        print(f"\n{'=' * 70}")
        print(f" 模型 {i}/{len(args.model_paths)}: {model_path}")
        print(f"{'=' * 70}")

        # 加载模型
        model, ckpt_epoch, ckpt_acc = load_retizero_with_lora(args.base_weight_path, model_path)

        # 运行验证
        metrics = validate_single_model(
            model,
            categories,
            preprocess,
            combining_function,
            split=args.fashioniq_split,
            dataset_root=args.fashioniq_root,
        )
        result = _canonical_checkpoint_result(
            model_path=model_path,
            sequence=i,
            checkpoint_epoch=ckpt_epoch,
            checkpoint_accuracy=ckpt_acc,
            categories=categories,
            metrics=metrics,
        )
        aggregate = result["aggregate"]
        all_results.append(result)

        # 打印汇总
        print(f"\n  ---- 模型 {i} 汇总 ----")
        print(f"  分类 val acc (LoRA): {ckpt_acc:.4f}")
        print(f"  CIR avg R@1:  {aggregate['average_recall_at1']:.4f}")
        print(f"  CIR avg R@5:  {aggregate['average_recall_at5']:.4f}")
        print(f"  CIR avg R@10: {aggregate['average_recall_at10']:.4f}")
        print(f"  CIR avg recall: {aggregate['average_recall']:.4f}")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 跨模型统计摘要
    if len(all_results) > 1:
        print(f"\n  ===== {len(all_results)} 次运行的统计摘要 =====")
        summary_metrics = [
            "average_recall_at1",
            "average_recall_at5",
            "average_recall_at10",
            "average_recall",
        ]
        for metric in summary_metrics:
            values = [
                result["aggregate"][metric]
                for result in all_results
            ]
            print(f"  {metric:30s}: {np.mean(values):7.4f} ± {np.std(values):7.4f}")

        for category in categories:
            for metric in (
                "recall_at1",
                "recall_at5",
                "recall_at10",
            ):
                values = [
                    result["per_category"][category][metric]
                    for result in all_results
                ]
                name = f"{category}_{metric}"
                print(
                    f"  {name:30s}: "
                    f"{np.mean(values):7.4f} ± {np.std(values):7.4f}"
                )

    return all_results


def _metrics_document(identity, args, results):
    return {
        "schema_version": 1,
        "dataset": identity.dataset_slug,
        "dataset_format": identity.dataset_format,
        "evaluation_name": args.evaluation_name,
        "split": args.fashioniq_split,
        "results": results,
    }


def _csv_projection(categories, results):
    metric_names = ("recall_at1", "recall_at5", "recall_at10")
    category_fields = [
        f"{category}_{metric}"
        for category in categories
        for metric in metric_names
    ]
    aggregate_fields = [
        "average_recall_at1",
        "average_recall_at5",
        "average_recall_at10",
        "average_recall",
    ]
    common_fields = [
        "model_path",
        "epoch",
        "classification_val_accuracy",
    ]
    fieldnames = common_fields + category_fields + aggregate_fields
    rows = []
    for result in results:
        row = {field: result.get(field) for field in common_fields}
        for category in categories:
            for metric in metric_names:
                row[f"{category}_{metric}"] = (
                    result["per_category"][category][metric]
                )
        for field in aggregate_fields:
            row[field] = result["aggregate"][field]
        rows.append(row)
    return rows, fieldnames


def main(argv=None):
    args = build_parser().parse_args(argv)
    categories = args.dress_types or list_fashioniq_categories(
        args.fashioniq_split,
        args.fashioniq_root,
    )
    if not categories:
        root_description = (
            f" under {args.fashioniq_root}"
            if args.fashioniq_root
            else ""
        )
        raise FileNotFoundError(
            "No FashionIQ-format categories found for split "
            f"{args.fashioniq_split}{root_description}"
        )

    project_root = Path(__file__).resolve().parents[1]
    identity = resolve_fashioniq_evaluation_identity(
        project_root=project_root,
        dress_types=categories,
        split=args.fashioniq_split,
        dataset_root=args.fashioniq_root,
        output_dataset=args.output_dataset,
        root_resolver=resolve_fashioniq_root,
    )
    layout = create_evaluation_layout(
        project_root=project_root,
        output_root=args.output_root,
        identity=identity,
        evaluation_script="src/validate_retizero_lora.py",
        evaluation_name=args.evaluation_name,
        model_name="RetiZero LoRA",
        split=args.fashioniq_split,
        categories=categories,
        cli_args=vars(args),
        input_paths={
            "base_weight_path": args.base_weight_path,
            "model_paths": args.model_paths,
        },
        metrics_csv_filename=args.output_csv,
    )

    try:
        with tee_evaluation_output(layout.log):
            try:
                print(f"Evaluation output directory: {layout.root}")
                results = _run_evaluation(args, categories)
                document = _metrics_document(identity, args, results)
                rows, fieldnames = _csv_projection(categories, results)
                publish_evaluation_metrics(
                    layout,
                    document,
                    rows,
                    fieldnames,
                )
                print(f"结果已保存至: {layout.root}")
            except BaseException:
                traceback.print_exc()
                raise
    except BaseException as error:
        discard_evaluation_metrics(layout)
        finalize_evaluation(layout, "failed", error=error)
        raise
    else:
        finalize_evaluation(layout, "succeeded")


if __name__ == '__main__':
    main()
