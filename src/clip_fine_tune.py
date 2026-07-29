from comet_ml import Experiment
import json
import multiprocessing
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data_utils import (
    base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset,
    ToClipTensor, list_fashioniq_categories
)
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics

import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _dataloader_num_workers() -> int:
    env_value = os.environ.get("CLIP4CIR_NUM_WORKERS")
    if env_value is not None:
        return int(env_value)
    if not torch.cuda.is_available():
        return 0
    return multiprocessing.cpu_count()


class _NullExperiment:
    def train(self):
        return self

    def validate(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_name(self, *args, **kwargs):
        pass

    def log_code(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, mode='image'):
        if mode == 'image':
            return self.clip_model.encode_image(x)
        elif mode == 'text':
            return self.clip_model.encode_text(x)


def enable_grad_checkpointing(clip_model):
    """对 ViT 系列模型的视觉 transformer 启用 gradient checkpointing，
    以牺牲约 30% 速度换取约 40-60% 激活值显存节省。"""
    if hasattr(clip_model, "gradient_checkpointing_enable"):
        clip_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on transformers model.")
        return
    tf_model = getattr(clip_model, "_tf_model", None)
    if tf_model is not None and hasattr(tf_model, "gradient_checkpointing_enable"):
        tf_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on BLIP/BLIP2 transformers backbone.")
        return

    visual = getattr(clip_model, 'visual', None)
    if visual is None:
        return
    transformer = getattr(visual, 'transformer', None)
    if transformer is None:
        return
    resblocks = getattr(transformer, 'resblocks', None)
    if resblocks is None:
        return

    class CheckpointedSequential(nn.Sequential):
        def forward(self, x):
            for block in self:
                x = grad_checkpoint(block, x, use_reentrant=False)
            return x

    transformer.resblocks = CheckpointedSequential(*list(resblocks))
    print("Gradient checkpointing enabled on visual transformer resblocks.")


def _is_retfound_model_name(model_name: str) -> bool:
    return "RETFound" in str(model_name)


def _is_retizero_model_name(model_name: str) -> bool:
    return "RetiZero" in str(model_name)


def _is_blip_model_name(model_name: str) -> bool:
    name = str(model_name).upper()
    return "BLIP" in name


def _uses_raw_text_inputs(model_name: str) -> bool:
    return (
        _is_blip_model_name(model_name)
        or _is_retfound_model_name(model_name)
        or _is_retizero_model_name(model_name)
    )


def _safe_model_tag(model_name: str) -> str:
    """Keep raw model name for clip.load, but use a filesystem-safe tag for run folders."""
    name = str(model_name).strip()
    known = {
        "ViT-B/32": "vitb32",
        "ViT-L/14": "vitl14",
        "RN50x4": "rn50x4",
    }
    if name in known:
        return known[name]
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return normalized or "model"


def _build_clip_like_preprocess(input_dim: int = 224, force_rgb: bool = True):
    return transforms.Compose([
        transforms.Resize(input_dim, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_dim),
        ToClipTensor(force_rgb=force_rgb),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def _build_retfound_preprocess(input_dim: int = 224, force_rgb: bool = True):
    return _build_clip_like_preprocess(input_dim=input_dim, force_rgb=force_rgb)


def _build_preprocess(
    transform: str,
    input_dim: int,
    target_ratio: float,
    clip_preprocess,
    force_rgb: bool,
    medical_mode: bool,
    disable_targetpad_in_medical: bool,
):
    if transform == "clip":
        # Keep exact OpenAI preprocess only for legacy path.
        if force_rgb and not medical_mode:
            return clip_preprocess
        return _build_clip_like_preprocess(input_dim=input_dim, force_rgb=force_rgb)

    if transform == "squarepad":
        return squarepad_transform(input_dim, force_rgb=force_rgb)

    if transform == "targetpad":
        use_targetpad = not (medical_mode and disable_targetpad_in_medical)
        return targetpad_transform(
            target_ratio,
            input_dim,
            force_rgb=force_rgb,
            apply_targetpad=use_targetpad,
        )

    raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")


def _load_model_for_finetune(clip_model_name: str, kwargs: dict):
    force_rgb = bool(kwargs.get("force_rgb", True))
    if _is_blip_model_name(clip_model_name):
        try:
            from src.blip_adapter import BLIPAdapter
        except ImportError:
            from blip_adapter import BLIPAdapter

        model = BLIPAdapter(
            model_type=str(kwargs.get("blip_model_type") or clip_model_name),
            backend=str(kwargs.get("blip_backend", "auto")),
            model_name=kwargs.get("blip_model_name"),
            model_path=kwargs.get("clip_model_path"),
            projection_dim=int(kwargs.get("projection_dim", kwargs.get("blip_projection_dim", 768))),
            input_resolution=int(kwargs.get("blip_input_resolution", 224)),
            max_text_len=int(kwargs.get("blip_max_text_len", 77)),
            device=device,
        )
        return model, _build_clip_like_preprocess(
            input_dim=int(kwargs.get("blip_input_resolution", 224)),
            force_rgb=force_rgb,
        )

    if _is_retizero_model_name(clip_model_name):
        try:
            from src.retizero_adapter import RetiZeroAdapter
        except ImportError:
            from retizero_adapter import RetiZeroAdapter

        base_path = kwargs.get("retizero_base_path") or kwargs.get("clip_model_path")
        if not base_path:
            raise ValueError(
                "RetiZero requires --retizero-base-path pointing to RetiZero.pth"
            )
        model = RetiZeroAdapter(base_path)
        return model, _build_clip_like_preprocess(
            input_dim=224,
            force_rgb=force_rgb,
        )

    if _is_retfound_model_name(clip_model_name):
        try:
            from src.retfound_adapter import RETFoundAdapter
        except ImportError:
            from retfound_adapter import RETFoundAdapter

        backbone_path = kwargs.get("retfound_backbone_path") or kwargs.get("clip_model_path")
        if not backbone_path:
            raise ValueError("RETFound 需要提供 --retfound-backbone-path 或 --clip-model-path")
        text_model_name = kwargs.get("retfound_text_model", "ViT-L/14")
        projection_dim = int(kwargs.get("retfound_projection_dim", 768))

        model = RETFoundAdapter(
            backbone_path=backbone_path,
            text_model_name=text_model_name,
            projection_dim=projection_dim,
            input_resolution=224,
        )
        return model, _build_retfound_preprocess(224, force_rgb=force_rgb)

    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    return model, preprocess


def _maybe_load_custom_clip_weights(clip_model, clip_model_path: str, clip_model_name: str):
    if (
        not clip_model_path
        or _is_retfound_model_name(clip_model_name)
        or _is_retizero_model_name(clip_model_name)
        or _is_blip_model_name(clip_model_name)
    ):
        return

    print(f"Loading custom CLIP weights from: {clip_model_path}")
    checkpoint = torch.load(clip_model_path, map_location='cpu')
    if "state_dict" in checkpoint:
        clip_weights = checkpoint["state_dict"]
    elif "CLIP" in checkpoint:
        clip_weights = checkpoint["CLIP"]
    else:
        clip_weights = checkpoint
    if any(k.startswith("module.") for k in clip_weights.keys()):
        clip_weights = {k.replace("module.", "", 1): v for k, v in clip_weights.items()}
    missing, unexpected = clip_model.load_state_dict(clip_weights, strict=False)
    print(f"Custom weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def _maybe_initialize_adapter_for_optimizer(clip_model):
    if hasattr(clip_model, "initialize_projection_heads"):
        summary = clip_model.initialize_projection_heads(device=device)
        print("Adapter diagnostic before optimizer:")
        print(json.dumps(summary, indent=4, ensure_ascii=False))
    elif hasattr(clip_model, "diagnostic_summary"):
        print("Adapter diagnostic before optimizer:")
        print(json.dumps(clip_model.diagnostic_summary(), indent=4, ensure_ascii=False))


def _configure_finetune_parameters(
    clip_model: nn.Module,
    clip_model_name: str,
    encoder: str,
):
    if _is_retizero_model_name(clip_model_name):
        if encoder != "both":
            raise ValueError("RetiZero CIR fine-tuning only supports --encoder both")
        trainable_names = clip_model.configure_cir_finetuning()
        print(
            "RetiZero CIR fine-tuning: vision LoRA + vision/text projection heads "
            f"({len(trainable_names)} tensors)"
        )
        return

    if encoder == "text":
        print("Only the CLIP text encoder will be fine-tuned")
        for parameter in clip_model.visual.parameters():
            parameter.requires_grad = False
    elif encoder == "image":
        print("Only the CLIP image encoder will be fine-tuned")
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
        for parameter in clip_model.visual.parameters():
            parameter.requires_grad = True
    elif encoder == "both":
        print("Both CLIP encoders will be fine-tuned")
    else:
        raise ValueError("encoder parameter should be in ['text', 'image', 'both']")


def _trainable_parameter_list(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def _print_trainable_parameter_summary(model: nn.Module, optimizer=None):
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel(), id(param)))

    total = sum(numel for _, numel, _ in trainable)
    print(f"Trainable parameter tensors: {len(trainable)}, scalars: {total}")

    buckets = {}
    for name, numel, _ in trainable:
        clean_name = name
        if clean_name.startswith("module."):
            clean_name = clean_name[len("module."):]
        if clean_name.startswith("clip_model."):
            clean_name = clean_name[len("clip_model."):]
        bucket = clean_name.split(".", 1)[0]
        buckets[bucket] = buckets.get(bucket, 0) + numel
    print("Trainable parameter groups:")
    print(json.dumps(buckets, indent=4, ensure_ascii=False))

    if optimizer is not None:
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        missing = [name for name, _, param_id in trainable if param_id not in optimizer_param_ids]
        print(
            f"Optimizer coverage: {len(trainable) - len(missing)}/{len(trainable)} trainable tensors included"
        )
        if missing:
            raise RuntimeError(
                "Optimizer is missing trainable parameters. First missing tensors: "
                + ", ".join(missing[:10])
            )


def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                      **kwargs):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned CLIP model
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best CLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
    model_tag = _safe_model_tag(clip_model_name)
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{model_tag}_{training_start}")
    training_path.mkdir(exist_ok=True, parents=True)

    clip_model, clip_preprocess = _load_model_for_finetune(clip_model_name, kwargs)

    # 加载自定义预训练权重（如 BMC_CLIP_CF）
    clip_model_path = kwargs.get('clip_model_path', None)
    _maybe_load_custom_clip_weights(clip_model, clip_model_path, clip_model_name)

    # 对 ViT 架构启用 gradient checkpointing，节省激活值显存
    enable_grad_checkpointing(clip_model)

    _configure_finetune_parameters(clip_model, clip_model_name, encoder)

    clip_model = clip_model.to(device)
    _maybe_initialize_adapter_for_optimizer(clip_model)

    # 2. 【核心修改】先用 Wrapper 包装，再开启 DataParallel
    # 这一步保证了调用模型时会进入 forward，从而触发多卡数据分发
    clip_model = CLIPWrapper(clip_model)

    if torch.cuda.device_count() > 1:
        print(f"🚀 检测到 {torch.cuda.device_count()} 张显卡，开启 DataParallel 模式")
        clip_model = torch.nn.DataParallel(clip_model)

    # 3. 移至设备并保持模式
    clip_model = clip_model.to(device)
    clip_model.eval().float()
    
    # 4. 获取分辨率 (注意：现在层级变成了 DataParallel -> CLIPWrapper -> CLIP)
    if hasattr(clip_model, 'module'):
        actual_clip = clip_model.module.clip_model
    else:
        actual_clip = clip_model.clip_model
        
    input_dim = actual_clip.visual.input_resolution#########################################

    target_ratio = kwargs.get('target_ratio', 1.25)
    medical_mode = bool(kwargs.get("medical_mode", False))
    disable_targetpad_in_medical = bool(kwargs.get("disable_targetpad_in_medical", False))
    force_rgb = bool(kwargs.get("force_rgb", True))
    preprocess = _build_preprocess(
        transform=transform,
        input_dim=input_dim,
        target_ratio=target_ratio,
        clip_preprocess=clip_preprocess,
        force_rgb=force_rgb,
        medical_mode=medical_mode,
        disable_targetpad_in_medical=disable_targetpad_in_medical,
    )
    fashioniq_root = kwargs.get("fashioniq_root")
    print(
        "🧪 Fine-tune preprocess: "
        f"transform={transform}, force_rgb={'ON' if force_rgb else 'OFF'}, "
        f"medical_mode={'ON' if medical_mode else 'OFF'}, "
        f"targetpad_in_medical={'OFF' if (medical_mode and disable_targetpad_in_medical) else 'ON'}"
    )

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        index_features_list = []
        index_names_list = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset(
            'val', [dress_type], 'relative', preprocess, dataset_root=fashioniq_root)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset(
            'val', [dress_type], 'classic', preprocess, dataset_root=fashioniq_root)
        classic_val_datasets.append(classic_val_dataset)
        if encoder == 'text':
            index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
            index_features_list.append(index_features_and_names[0])
            index_names_list.append(index_features_and_names[1])

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset(
        'train', train_dress_types, 'relative', preprocess, dataset_root=fashioniq_root)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=_dataloader_num_workers(), pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    trainable_params = _trainable_parameter_list(clip_model)
    optimizer = optim.AdamW(
        [{'params': trainable_params, 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    _print_trainable_parameter_summary(clip_model, optimizer)
    scheduler_patience = int(kwargs.get("scheduler_patience", 5))
    scheduler_factor = float(kwargs.get("scheduler_factor", 0.5))
    min_lr = float(kwargs.get("min_lr", 1e-8))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr, verbose=True
    )
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                # 自动适配单条或多条 caption
                # 1. 确保 captions 是一个简单的字符串列表 [cap1, cap2, ..., cap_n]
                # 如果你在 data_utils.py 里已经改成了返回单个字符串，这里的 captions 已经是列表了
                if isinstance(captions[0], tuple) or isinstance(captions[0], list):
                    # 以防万一，如果读进来还是嵌套格式，取第一个
                    flattened_captions = [c[0] for c in captions]
                else:
                    flattened_captions = list(captions)

                # 2. 直接进行 Tokenize，不再经过 generate_randomized_fiq_caption
                if _uses_raw_text_inputs(clip_model_name):
                    text_inputs = flattened_captions
                else:
                    text_inputs = clip.tokenize(flattened_captions, context_length=77, truncate=True).to(device, non_blocking=True)

                # Extract the features, compute the logits and the loss
                with torch.cuda.amp.autocast():
                    # 【修改点 1】：直接调用模型，会自动进入 CLIPWrapper.forward 并分发到多张显卡
                    reference_features = clip_model(reference_images, mode='image')
                    
                    # 【修改点 2】：同上，处理文本
                    caption_features = clip_model(text_inputs, mode='text')

                    predicted_features = combining_function(reference_features, caption_features)

                    # 【修改点 3】：同上，处理目标图像
                    target_features = F.normalize(clip_model(target_images, mode='image'))

                    logits = 100 * predicted_features @ target_features.T
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0 or epoch == num_epochs - 1:
            with experiment.validate():
                recalls_at1 = []
                recalls_at5 = []
                recalls_at10 = []

                # 遍历验证数据集（对应 FashionIQ 的不同类别）
                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                         idx_to_dress_mapping):
                    if encoder == 'text':
                        index_features, index_names = index_features_list[idx], index_names_list[idx]
                    else:
                        # 此函数已在 utils.py 中修改为支持多卡
                        index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
                    
                    # 【核心修改点 1】：解包 3 个返回值 (R@1, R@5, R@10)
                    recall_at1, recall_at5, recall_at10 = compute_fiq_val_metrics(
                        relative_val_dataset, clip_model, index_features, index_names, combining_function
                    )
                    
                    recalls_at1.append(recall_at1)
                    recalls_at5.append(recall_at5)
                    recalls_at10.append(recall_at10)

                # 汇总结果字典
                results_dict = {}
                for i in range(len(recalls_at10)):
                    category = idx_to_dress_mapping[i]
                    results_dict[f'{category}_recall_at1'] = recalls_at1[i]
                    results_dict[f'{category}_recall_at5'] = recalls_at5[i]
                    results_dict[f'{category}_recall_at10'] = recalls_at10[i]
                
                # 计算平均指标
                avg_r1 = mean(recalls_at1)
                avg_r5 = mean(recalls_at5)
                avg_r10 = mean(recalls_at10)
                
                results_dict.update({
                    f'average_recall_at1': avg_r1,
                    f'average_recall_at5': avg_r5,
                    f'average_recall_at10': avg_r10,
                    # 【核心修改点 2】：用 R@10 作为综合评价指标的基础
                    f'average_recall': (avg_r1 + avg_r5 + avg_r10) / 3
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(results_dict, epoch=epoch)

                # 验证结果保存至 CSV
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # --- 保存模型逻辑 ---
            if save_training:
                # 【核心修改点 3】：使用 R@1,5,10 的综合平均值来判定“最佳模型”
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    print(f"🥇 New best model found at epoch {epoch}! Saving...")
                    save_model('tuned_clip_best', epoch, clip_model, training_path)
                elif not save_best:
                    save_model(f'tuned_clip_{epoch}', epoch, clip_model, training_path)
            # ReduceLROnPlateau: 根据 average_recall 调整学习率
            scheduler.step(results_dict['average_recall'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch}] LR after scheduler step: {current_lr:.2e}")

def clip_finetune_cirr(num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                       **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=True, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = _load_model_for_finetune(clip_model_name, kwargs)

    # 加载自定义预训练权重（如 BMC_CLIP_CF）
    clip_model_path = kwargs.get('clip_model_path', None)
    _maybe_load_custom_clip_weights(clip_model, clip_model_path, clip_model_name)

    # 对 ViT 架构启用 gradient checkpointing，节省激活值显存
    enable_grad_checkpointing(clip_model)

    _configure_finetune_parameters(clip_model, clip_model_name, encoder)

    clip_model = clip_model.to(device)
    _maybe_initialize_adapter_for_optimizer(clip_model)
    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution

    target_ratio = kwargs.get('target_ratio', 1.25)
    medical_mode = bool(kwargs.get("medical_mode", False))
    disable_targetpad_in_medical = bool(kwargs.get("disable_targetpad_in_medical", False))
    force_rgb = bool(kwargs.get("force_rgb", True))
    preprocess = _build_preprocess(
        transform=transform,
        input_dim=input_dim,
        target_ratio=target_ratio,
        clip_preprocess=clip_preprocess,
        force_rgb=force_rgb,
        medical_mode=medical_mode,
        disable_targetpad_in_medical=disable_targetpad_in_medical,
    )
    print(
        "🧪 Fine-tune preprocess: "
        f"transform={transform}, force_rgb={'ON' if force_rgb else 'OFF'}, "
        f"medical_mode={'ON' if medical_mode else 'OFF'}, "
        f"targetpad_in_medical={'OFF' if (medical_mode and disable_targetpad_in_medical) else 'ON'}"
    )

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=_dataloader_num_workers(), pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    trainable_params = _trainable_parameter_list(clip_model)
    optimizer = optim.AdamW(
        [{'params': trainable_params, 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    _print_trainable_parameter_summary(clip_model, optimizer)
    scheduler_patience = int(kwargs.get("scheduler_patience", 5))
    scheduler_factor = float(kwargs.get("scheduler_factor", 0.5))
    min_lr = float(kwargs.get("min_lr", 1e-8))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr, verbose=True
    )
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Extract the features, compute the logits and the loss
                with torch.cuda.amp.autocast():
                    reference_features = clip_model.encode_image(reference_images)
                    if _uses_raw_text_inputs(clip_model_name):
                        text_inputs = list(captions)
                    else:
                        text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(
                            device, non_blocking=True
                        )
                    text_features = clip_model.encode_text(text_inputs)

                    target_features = F.normalize(clip_model.encode_image(target_images), dim=-1)
                    predicted_features = combining_function(reference_features, text_features)

                    logits = 100 * predicted_features @ target_features.T

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                if encoder != 'text':
                    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)
                results = compute_cirr_val_metrics(relative_val_dataset, clip_model, val_index_features,
                                                   val_index_names, combining_function)
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }
                print(json.dumps(results_dict, indent=4))

                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                if save_training:
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('tuned_clip_arithmetic', epoch, clip_model, training_path)
                    if save_best and results_dict['harmonic_mean'] > best_harmonic:
                        best_harmonic = results_dict['harmonic_mean']
                        save_model('tuned_clip_harmonic', epoch, clip_model, training_path)
                    if save_best and results_dict['geometric_mean'] > best_geometric:
                        best_geometric = results_dict['geometric_mean']
                        save_model('tuned_clip_geometric', epoch, clip_model, training_path)
                    if not save_best:
                        save_model(f'tuned_clip_{epoch}', epoch, clip_model, training_path)

                scheduler.step(results_dict['arithmetic_mean'])
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}] LR after scheduler step: {current_lr:.2e}")


if __name__ == '__main__':
    # 检查 PyTorch 实际能看到的显卡数量
    n_gpu = torch.cuda.device_count()
    print(f"--- 显卡检查报告 ---")
    print(f"PyTorch 识别到的可用显卡数: {n_gpu}")

    if n_gpu > 0:
        for i in range(n_gpu):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前正在使用的显卡索引: {torch.cuda.current_device()}")
    else:
        print("警告：未检测到 CUDA 显卡，程序将运行在 CPU 上。")
    print(f"--------------------")

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--dress-types", nargs='+', default=None,
                        help="FashionIQ-format categories to use, e.g. IDRiD or CH CO NM RB RCH UM")
    parser.add_argument(
        "--fashioniq-root",
        type=str,
        default=None,
        help="Explicit root for a FashionIQ-style dataset; overrides root auto-discovery",
    )
    parser.add_argument("--clip-model-name", default="RN50x4", type=str,
                        help="CLIP model to use, e.g. 'RN50x4', 'ViT-B/32', 'ViT-L/14' (default remains RN50x4)")
    parser.add_argument("--encoder", default='both', type=str,
                        help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--scheduler-patience", default=5, type=int,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler-factor", default=0.5, type=float,
                        help="Multiplicative LR decay factor for ReduceLROnPlateau")
    parser.add_argument("--min-lr", default=1e-8, type=float,
                        help="Lower bound for ReduceLROnPlateau learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--clip-model-path", type=str, default=None,
                        help="Path to a custom CLIP checkpoint (e.g. BMC_CLIP_CF.pt) to load on top of the base architecture")
    parser.add_argument("--retizero-base-path", type=str, default=None,
                        help="Path to the base RetiZero.pth checkpoint")
    parser.add_argument("--retfound-backbone-path", type=str, default=None,
                        help="Path to RETFound backbone checkpoint (e.g. RETFound_mae_natureCFP.pth)")
    parser.add_argument("--retfound-text-model", type=str, default="ViT-L/14",
                        help="OpenAI CLIP model name used for RETFound text tower")
    parser.add_argument("--retfound-projection-dim", type=int, default=768,
                        help="Unified embedding dim for RETFound adapter outputs")
    parser.add_argument("--blip-model-type", type=str, default="BLIP",
                        help="BLIP variant hint, e.g. BLIP or BLIP2")
    parser.add_argument("--blip-backend", type=str, default="auto",
                        help="BLIP backend: auto/transformers/lavis")
    parser.add_argument("--blip-model-name", type=str, default=None,
                        help="Optional backend model name, e.g. Salesforce/blip2-opt-2.7b")
    parser.add_argument("--blip-projection-dim", type=int, default=768,
                        help="Unified embedding dim for BLIP adapter outputs")
    parser.add_argument("--blip-input-resolution", type=int, default=224,
                        help="BLIP image resolution for adapter preprocessing")
    parser.add_argument("--blip-max-text-len", type=int, default=77,
                        help="Maximum text length used by BLIP tokenizer")
    parser.add_argument("--medical-mode", action='store_true',
                        help="Enable medical preprocessing mode for ablation")
    parser.add_argument("--disable-targetpad-in-medical", action='store_true',
                        help="When --medical-mode is enabled, bypass TargetPad in train/eval pipelines")
    parser.add_argument("--no-force-rgb", dest="force_rgb", action='store_false',
                        help="Do not force PIL RGB conversion; keep source channels then coerce tensor to 3 channels")
    parser.set_defaults(force_rgb=True)

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "dress_types": args.dress_types,
        "fashioniq_root": args.fashioniq_root,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "encoder": args.encoder,
        "save_best": args.save_best,
        "clip_model_path": args.clip_model_path,
        "retizero_base_path": args.retizero_base_path,
        "retfound_backbone_path": args.retfound_backbone_path,
        "retfound_text_model": args.retfound_text_model,
        "retfound_projection_dim": args.retfound_projection_dim,
        "blip_model_type": args.blip_model_type,
        "blip_backend": args.blip_backend,
        "blip_model_name": args.blip_model_name,
        "blip_projection_dim": args.blip_projection_dim,
        "blip_input_resolution": args.blip_input_resolution,
        "blip_max_text_len": args.blip_max_text_len,
        "medical_mode": args.medical_mode,
        "disable_targetpad_in_medical": args.disable_targetpad_in_medical,
        "force_rgb": args.force_rgb,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} clip fine-tuning",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet logging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = _NullExperiment()

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'cirr':
        training_hyper_params.pop("dress_types", None)
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        valid_dress_types = args.dress_types or list_fashioniq_categories(
            "train", args.fashioniq_root)
        if not valid_dress_types:
            valid_dress_types = ['CH', 'CO', 'NM', 'RB', 'RCH', 'UM', 'IDRiD']

        training_hyper_params.update(
            {
                'train_dress_types': valid_dress_types, 
                'val_dress_types': valid_dress_types
            }
        )
        training_hyper_params.pop("dress_types", None)
        clip_finetune_fiq(**training_hyper_params)
