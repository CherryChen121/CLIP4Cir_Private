import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
import traceback
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    CIRRDataset,
    FashionIQDataset,
    list_fashioniq_categories,
    resolve_fashioniq_root,
    squarepad_transform,
    targetpad_transform,
)
from combiner import Combiner
from dataset_identity import (
    resolve_dataset_identity,
    resolve_fashioniq_evaluation_identity,
)
from evaluation_outputs import (
    create_evaluation_layout,
    discard_evaluation_metrics,
    finalize_evaluation,
    publish_evaluation_metrics,
    tee_evaluation_output,
)
from fashioniq_evaluation import compute_recall_at_k
from utils import extract_index_features, collate_fn, element_wise_sum, device


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap common wrappers (DDP/DataParallel/custom wrappers) to reach the core model."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "clip_model"):
            unwrapped = unwrapped.clip_model
            continue
        break
    return unwrapped


def _is_blip_model_name(model_name: str) -> bool:
    return "BLIP" in str(model_name).upper()


def _model_handles_raw_text(model: nn.Module) -> bool:
    model_type = str(type(_unwrap_model(model)))
    return ("RetiZero" in model_type) or ("RETFound" in model_type) or ("BLIP" in model_type)


def _encode_text_features(model: nn.Module, input_captions: List[str]) -> torch.Tensor:
    if _model_handles_raw_text(model):
        return model(input_captions, mode='text')
    text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)
    return model(text_inputs, mode='text')


def _safe_load_state_dict(model: nn.Module, state_dict: dict, *, context: str):
    model_state = model.state_dict()
    shape_mismatch = []
    filtered_state = {}

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    if shape_mismatch:
        preview = "\n".join(
            f"  - {k}: ckpt={s1}, model={s2}" for k, s1, s2 in shape_mismatch[:10]
        )
        raise RuntimeError(
            f"{context} 检测到参数维度不匹配，已中止加载。\n{preview}"
        )

    load_result = model.load_state_dict(filtered_state, strict=False)
    print(
        f"{context} loaded. matched_keys={len(filtered_state)}, "
        f"missing_keys={len(load_result.missing_keys)}, unexpected_keys={len(load_result.unexpected_keys)}"
    )


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float, float]:
    """
    针对小库优化的 FashionIQ 验证指标计算函数 (R@1, R@5, R@10)
    :param relative_val_dataset: FashionIQ 验证数据集 (relative 模式)
    :param clip_model: CLIP 模型
    :param index_features: 验证集索引库特征 (Gallery features)
    :param index_names: 验证集索引库名称
    :param combining_function: 特征融合函数
    :return: (recall_at1, recall_at5, recall_at10)
    """

    # 1. 生成预测特征和对应的目标名称
    predicted_features, target_names = generate_fiq_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics (R@1, R@5, R@10)")

    return compute_recall_at_k(
        predicted_features,
        target_names,
        index_features,
        index_names,
        ks=(1, 5, 10),
    )


def generate_fiq_val_predictions(clip_model: nn.Module, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = torch.empty((0, index_features.shape[-1])).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):
        # 1. 展平描述
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        
        # 2. 自适应步长判定 (针对单条 caption 逻辑)
        if len(flattened_captions) == len(reference_names):
            input_captions = [cap.strip('.?, ').capitalize() for cap in flattened_captions]
        else:
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" 
                for i in range(0, len(flattened_captions), 2)
            ]
        
        with torch.no_grad():
            text_features = _encode_text_features(clip_model, input_captions)
            
            # 4. 获取图像特征 (从预计算的 index_features 中提取)
            if len(reference_names) == 1:
                reference_image_features = name_to_feat[reference_names[0]].unsqueeze(0)
            else:
                reference_image_features = torch.stack([name_to_feat[name] for name in reference_names])
            
            # --- 关键防护：确保 Batch 维度严格一致 ---
            if text_features.shape[0] != reference_image_features.shape[0]:
                min_bs = min(text_features.shape[0], reference_image_features.shape[0])
                text_features = text_features[:min_bs]
                reference_image_features = reference_image_features[:min_bs]
                batch_target_names = batch_target_names[:min_bs]

            # 5. 特征融合
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names

def fashioniq_val_retrieval(
        dress_type: str, combining_function: callable, clip_model: CLIP,
        preprocess: callable, split: str = "val", dataset_root=None):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the labeled evaluation datasets and extract the index features
    classic_val_dataset = FashionIQDataset(
        split, [dress_type], 'classic', preprocess, dataset_root=dataset_root)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset(
        split,
        [dress_type],
        'relative',
        preprocess,
        dataset_root=dataset_root,
        return_target=(split == "test"),
    )

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                   combining_function)


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, index_features.shape[-1])).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = _encode_text_features(clip_model, list(captions))
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function)


def _load_optional_clip_weights(clip_model: CLIP, clip_model_path: Path):
    """Support {'CLIP': ...}, {'state_dict': ...}, and raw OpenAI CLIP state_dict formats."""
    if not clip_model_path:
        return

    print('Trying to load the CLIP model')
    saved_state_dict = torch.load(clip_model_path, map_location=device)
    if isinstance(saved_state_dict, dict) and "CLIP" in saved_state_dict:
        clip_weights = saved_state_dict["CLIP"]
    elif isinstance(saved_state_dict, dict) and "state_dict" in saved_state_dict and isinstance(saved_state_dict["state_dict"], dict):
        clip_weights = saved_state_dict["state_dict"]
    else:
        clip_weights = saved_state_dict

    if any(k.startswith("module.") for k in clip_weights.keys()):
        clip_weights = {k.replace("module.", "", 1): v for k, v in clip_weights.items()}

    _safe_load_state_dict(clip_model, clip_weights, context="Validation CLIP checkpoint")


def _csv_projection(args, categories, results):
    common_fields = ["model_path", "epoch"]
    if args.dataset.lower() == "fashioniq":
        metric_names = ("recall_at1", "recall_at5", "recall_at10")
        category_fields = [
            f"{category}_{metric_name}"
            for category in categories
            for metric_name in metric_names
        ]
        aggregate_fields = [
            "average_recall_at1",
            "average_recall_at5",
            "average_recall_at10",
        ]
        fieldnames = common_fields + category_fields + aggregate_fields
        rows = []
        for result in results:
            row = {field: result.get(field) for field in common_fields}
            for category in categories:
                for metric_name in metric_names:
                    row[f"{category}_{metric_name}"] = (
                        result["per_category"][category][metric_name]
                    )
            for field in aggregate_fields:
                row[field] = result["aggregate"][field]
            rows.append(row)
        return rows, fieldnames

    aggregate_fields = [
        "group_recall_at1",
        "group_recall_at2",
        "group_recall_at3",
        "global_recall_at1",
        "global_recall_at5",
        "global_recall_at10",
        "global_recall_at50",
    ]
    fieldnames = common_fields + aggregate_fields
    rows = []
    for result in results:
        row = {field: result.get(field) for field in common_fields}
        row.update(
            {
                field: result["aggregate"][field]
                for field in aggregate_fields
            }
        )
        rows.append(row)
    return rows, fieldnames


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str,
                        help="CLIP model to use, e.g. 'RN50x4', 'ViT-B/32', 'ViT-L/14' (default remains RN50x4)")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument(
        "--fashioniq-root",
        type=str,
        default=None,
        help="Explicit root for a FashionIQ-style dataset; overrides root auto-discovery",
    )
    parser.add_argument(
        "--dress-types",
        nargs="+",
        default=None,
        help="FashionIQ-format categories to evaluate",
    )
    parser.add_argument(
        "--fashioniq-split",
        choices=("val", "test"),
        default="val",
        help="Labeled FashionIQ-format split to evaluate",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Evaluation output root; defaults to the project outputs directory",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Explicit actual dataset name used beneath the output root",
    )
    parser.add_argument(
        "--evaluation-name",
        default="validation",
        help="Descriptive label recorded in the evaluation manifest",
    )
    return parser


def _run_evaluation(args, categories):
    if _is_blip_model_name(args.clip_model_name):
        try:
            from src.blip_adapter import BLIPAdapter
        except ImportError:
            from blip_adapter import BLIPAdapter

        clip_model = BLIPAdapter(
            model_type=args.clip_model_name,
            model_path=str(args.clip_model_path) if args.clip_model_path else None,
            projection_dim=args.projection_dim,
            input_resolution=224,
            device=device,
            normalize_output=False,
        ).to(device)
        clip_preprocess = targetpad_transform(args.target_ratio, clip_model.visual.input_resolution)
    else:
        clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)

    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    _load_optional_clip_weights(clip_model, args.clip_model_path)

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess)

        aggregate = {
            "group_recall_at1": group_recall_at1,
            "group_recall_at2": group_recall_at2,
            "group_recall_at3": group_recall_at3,
            "global_recall_at1": recall_at1,
            "global_recall_at5": recall_at5,
            "global_recall_at10": recall_at10,
            "global_recall_at50": recall_at50,
        }
        for name, value in aggregate.items():
            print(f"{name} = {value}")
        return [
            {
                "model_path": (
                    str(args.clip_model_path)
                    if args.clip_model_path
                    else None
                ),
                "epoch": None,
                "per_category": {},
                "aggregate": aggregate,
            }
        ]

    elif args.dataset.lower() == 'fashioniq':
        average_recall1_list = []
        average_recall5_list = []
        average_recall10_list = []
        per_category = {}

        for dt in categories:
            r1, r5, r10 = fashioniq_val_retrieval(
                dt,
                combining_function,
                clip_model,
                preprocess,
                split=args.fashioniq_split,
                dataset_root=args.fashioniq_root,
            )
            average_recall1_list.append(r1)
            average_recall5_list.append(r5)
            average_recall10_list.append(r10)

            per_category[dt] = {
                "recall_at1": r1,
                "recall_at5": r5,
                "recall_at10": r10,
            }
            print(f"{dt}_recallat1 = {r1}")
            print(f"{dt}_recallat5 = {r5}")
            print(f"{dt}_recallat10 = {r10}")

        aggregate = {
            "average_recall_at1": mean(average_recall1_list),
            "average_recall_at5": mean(average_recall5_list),
            "average_recall_at10": mean(average_recall10_list),
        }
        print(f"average recall1 = {aggregate['average_recall_at1']}")
        print(f"average recall5 = {aggregate['average_recall_at5']}")
        print(f"average recall10 = {aggregate['average_recall_at10']}")
        return [
            {
                "model_path": (
                    str(args.clip_model_path)
                    if args.clip_model_path
                    else None
                ),
                "epoch": None,
                "per_category": per_category,
                "aggregate": aggregate,
            }
        ]
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


def _metrics_document(identity, args, results):
    split = (
        args.fashioniq_split
        if args.dataset.lower() == "fashioniq"
        else "val"
    )
    return {
        "schema_version": 1,
        "dataset": identity.dataset_slug,
        "dataset_format": identity.dataset_format,
        "evaluation_name": args.evaluation_name,
        "split": split,
        "results": results,
    }


def _resolve_evaluation_identity(args, project_root):
    if args.dataset.lower() == "fashioniq":
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
        identity = resolve_fashioniq_evaluation_identity(
            project_root=project_root,
            dress_types=categories,
            split=args.fashioniq_split,
            dataset_root=args.fashioniq_root,
            output_dataset=args.output_dataset,
            root_resolver=resolve_fashioniq_root,
        )
        return identity, categories, args.fashioniq_split

    if args.dataset.lower() == "cirr":
        identity = resolve_dataset_identity(
            dataset_format="cirr",
            dress_types=(),
            dataset_root_requested=None,
            dataset_root_resolved=None,
            output_dataset=args.output_dataset,
        )
        return identity, [], "val"

    raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


def main(argv=None):
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    identity, categories, split = _resolve_evaluation_identity(
        args,
        project_root,
    )
    layout = create_evaluation_layout(
        project_root=project_root,
        output_root=args.output_root,
        identity=identity,
        evaluation_script="src/validate.py",
        evaluation_name=args.evaluation_name,
        model_name=args.clip_model_name,
        split=split,
        categories=categories,
        cli_args=vars(args),
        input_paths={
            "clip_model_path": args.clip_model_path,
            "combiner_path": args.combiner_path,
        },
    )

    try:
        with tee_evaluation_output(layout.log):
            try:
                print(f"Evaluation output directory: {layout.root}")
                results = _run_evaluation(args, categories)
                document = _metrics_document(identity, args, results)
                rows, fieldnames = _csv_projection(
                    args,
                    categories,
                    results,
                )
                publish_evaluation_metrics(
                    layout,
                    document,
                    rows,
                    fieldnames,
                )
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
