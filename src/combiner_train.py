import os
os.environ["COMET_LOGGING_CONSOLE"] = "info"

from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics

print(f">>> 隔离校验：Torch 当前真实看到的显卡数量为: {torch.cuda.device_count()}")

import torch.nn as nn

class CLIPWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, mode=None):
        if mode == 'image':
            return self.clip_model.encode_image(x)
        elif mode == 'text':
            return self.clip_model.encode_text(x)
        return self.clip_model(x)

    def __getattr__(self, name):
        try:
            # 1. 优先调用父类(nn.Module)的方法
            # 这样 self.clip_model 这种子模块才能被正确找到，不会触发递归
            return super().__getattr__(name)
        except AttributeError:
            # 2. 如果父类找不到（比如 visual），再去 clip_model 内部找
            # 注意：这里必须使用 self._modules 字典直接访问，这是最安全的防递归写法
            return getattr(self._modules['clip_model'], name)

def combiner_training_fiq(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    针对医疗 UWF 数据集优化后的训练函数
    """
    best_avg_recall = 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + f"_pid{os.getpid()}"
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_fiq_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # 保存超参数
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

# === 模型加载逻辑对齐 ===
    if "RetiZero" in clip_model_name:
        print("🔧 正在初始化 RetiZero 适配器...")
        try:
            from src.retizero_adapter import RetiZeroAdapter
        except ImportError:
            from retizero_adapter import RetiZeroAdapter
        # 优先使用 retizero_base_path 作为 base 权重路径
        # 如果未指定，则回退到 clip_model_path（兼容旧用法）
        model_path = kwargs.get("retizero_base_path") or \
                     kwargs.get("clip_model_path", "pretrained_models/RetiZero.pth")
        
        # 1. 实例化适配器
        clip_model = RetiZeroAdapter(model_path).to(device)
        
        # 2. 为了兼容原代码的 clip_model.visual.xxx 调用，我们在适配器里手动挂载属性
        # 假设 RetiZero 输出是 1024，输入是 224
        clip_model.visual = type('', (), {})() # 创建一个空对象
        clip_model.visual.input_resolution = 224
        clip_model.visual.output_dim = 1024 
        
        # --- src/combiner_train.py ---

        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        # 修正：加上 is_train 参数，即使函数内部没用到它，也要接收它
        def build_transforms(input_dim=224, is_train=False):
            """
            针对 FashionIQ 和 RetiZero 的通用预处理
            增加了 is_train 参数以对齐 combiner_train.py 的调用接口
            """
            # 统一返回两个值，因为原本的代码里用了 _, clip_preprocess = ...
            # 第一个值通常是带 Augmentation 的，第二个是 Eval 用的。这里我们统一用标准的。
            preprocess = transforms.Compose([
                transforms.Resize(input_dim, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_dim),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073), 
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            ])
            return preprocess, preprocess # 返回两次，对齐解包逻辑

        _, clip_preprocess = build_transforms(is_train=False)
    else:
        # 官方 CLIP 逻辑
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        clip_model = CLIPWrapper(clip_model)

    # === 关键：必须保留 eval() ===
    clip_model.eval() 

    # 下面这两行现在对 RetiZero 和官方 CLIP 都能跑通了
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    # 预处理选择
    if transform == "clip":
        preprocess = clip_preprocess
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    
    # --- src/combiner_train.py 修改如下 ---

    # --- 修正后的加载逻辑（带自动身份识别） ---
    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        
        # 1. 自动检测模型身份
        is_retizero = (kwargs.get("clip_model_name") == 'RetiZero') or \
                      (type(clip_model).__name__ == 'RetiZeroAdapter')

        if is_retizero:
            # 检查 clip_model_path 是否指向 LoRA 微调 checkpoint
            clip_model_path = kwargs["clip_model_path"]
            ckpt_probe = torch.load(clip_model_path, map_location='cpu')
            is_lora_ckpt = 'state_dict' in ckpt_probe and any(
                k.startswith('img_encoder.') for k in ckpt_probe.get('state_dict', {}).keys()
            )
            del ckpt_probe  # 释放内存

            if is_lora_ckpt:
                print('🔧 检测到 LoRA 微调 checkpoint，正在加载 vision encoder 权重...')
                clip_model.load_lora_checkpoint(clip_model_path)
            else:
                print('✨ [Debug] 身份确认：RetiZero base 模型。')
                print('✨ 权重已在初始化阶段完成加载，跳过冗余逻辑。')
        else:
            # 原有的加载逻辑，仅针对非 RetiZero 模型
            clip_model_path = kwargs["clip_model_path"]
            saved_state_dict = torch.load(clip_model_path, map_location=device)
            
            # 兼容性处理：优先取 "CLIP" 键，再取 "state_dict" 键，否则取全部
            if "CLIP" in saved_state_dict:
                clip_weights = saved_state_dict["CLIP"]
            elif "state_dict" in saved_state_dict and isinstance(saved_state_dict["state_dict"], dict):
                clip_weights = saved_state_dict["state_dict"]
            else:
                clip_weights = saved_state_dict
            
            # 去除 DataParallel/DDP 的 "module." 前缀
            if any(k.startswith("module.") for k in clip_weights.keys()):
                clip_weights = {k.replace("module.", "", 1): v for k, v in clip_weights.items()}
            
            if any(k.startswith("clip_model.") for k in clip_weights.keys()):
                clip_model.load_state_dict(clip_weights)
            else:
                new_state_dict = {f"clip_model.{k}": v for k, v in clip_weights.items()}
                clip_model.load_state_dict(new_state_dict)
            print('CLIP model loaded successfully')

    # 统一转为 float32
    clip_model = clip_model.float()


    # 准备验证集特征
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

    # 1. 动态获取特征维度 (针对 RetiZero 进行强制矫正)
    if "RetiZero" in clip_model_name:
        feature_dim = 512
        print(f"🛠️ 身份确认：RetiZero。强制同步 Combiner 输入维度为: {feature_dim}")
    else:
        # 官方 CLIP (RN50x4 等) 保持原有逻辑
        # 增加一个 DataParallel 的穿透判断
        temp_model = clip_model.module if hasattr(clip_model, 'module') else clip_model
        feature_dim = temp_model.visual.output_dim 
    
    # 2. 初始化 Combiner
    # 注意：这里我们建议把 projection_dim 和 hidden_dim 也对齐到 512，
    # 这样可以让残差连接（Residual Connection）的计算更加顺滑
    combiner = Combiner(clip_feature_dim=feature_dim, 
                        projection_dim=feature_dim, # 建议设为 512
                        hidden_dim=feature_dim).to(device, non_blocking=True) # 建议设为 512

    print(f"📊 Combiner 实例已创建，输入维度: {feature_dim}, 隐藏层维度: {feature_dim}")

    # 3. 多卡支持
    if torch.cuda.device_count() > 1:
        print(f"🚀 检测到 {torch.cuda.device_count()} 张显卡，开启 Combiner 的 DataParallel 模式")
        combiner = nn.DataParallel(combiner)
    
    combiner = combiner.to(device)

    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if save_best:
        best_avg_recall = 0

    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    print('Training loop started')
    for epoch in range(num_epochs):
        # 只对官方 CLIP 模型进行权重转换
        if torch.cuda.is_available() and "RetiZero" not in clip_model_name:
            clip.model.convert_weights(clip_model)
        
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images.size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # 1. 展平并处理描述文本
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                
                if len(flattened_captions) == images_in_batch:
                    input_captions = [cap.strip('.?, ').capitalize() for cap in flattened_captions]
                else:
                    # FashionIQ 双描述合并逻辑
                    input_captions = generate_randomized_fiq_caption(flattened_captions)

                # 2. 准备文本输入（核心修改：RetiZero 不在循环外分词）
                if "RetiZero" in clip_model_name:
                    # 保持为原始字符串列表 List[str]
                    text_inputs = input_captions 
                else:
                    # 官方 CLIP 依然需要提前分词为 Tensor
                    text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

                # 3. 提取特征
                with torch.no_grad():
                    # --- 提取图像特征 ---
                    reference_image_features = torch.vstack([
                        clip_model.encode_image(mini_batch).float() 
                        for mini_batch in torch.split(reference_images, clip_bs)
                    ])
                    target_image_features = torch.vstack([
                        clip_model.encode_image(mini_batch).float() 
                        for mini_batch in torch.split(target_images, clip_bs)
                    ])

                    # --- 提取文本特征（核心修复：手动切片并支持 List/Tensor） ---
                    text_features_list = []
                    num_samples = len(input_captions)
                    
                    for i in range(0, num_samples, clip_bs):
                        # 无论是 List[str] 还是 Tensor，Python 切片语法 [i:i+bs] 都是通用的
                        mini_batch_text = text_inputs[i : i + clip_bs]
                        
                        # 调用 encode_text。如果是 RetiZero，它内部会完成分词
                        batch_text_feat = clip_model.encode_text(mini_batch_text).float()
                        text_features_list.append(batch_text_feat)
                    
                    text_features = torch.vstack(text_features_list)

                # 4. 后续计算 Logits 和 Loss (保持你之前的逻辑即可)
                with torch.cuda.amp.autocast():
                    # 1. 多卡并行提取融合特征 (此时返回的是拼接后的 N x Dim 特征)
                    fused_features = combiner(reference_image_features, text_features)

                    # 2. 在主卡上进行归一化 (确保相似度计算在单位球面上)
                    fused_features = F.normalize(fused_features, dim=-1)
                    target_image_features = F.normalize(target_image_features, dim=-1)

                    # --- 修正后的 Logits 计算逻辑 ---

                    # 1. 获取原始模型实例
                    raw_model = combiner.module if isinstance(combiner, torch.nn.DataParallel) else combiner

                    # 2. 直接获取缩放系数（不再取指数）
                    # 既然 Combiner 里定义了 self.logit_scale = 100，这个值本身就是最强的缩放了
                    l_scale = raw_model.logit_scale 
                    if torch.is_tensor(l_scale):
                        # 如果它是 Tensor (nn.Parameter)，取数值
                        l_scale = l_scale.detach().item() 
                    else:
                        # 如果它是普通的 int 或 float
                        l_scale = float(l_scale)

                    # 3. 计算相似度矩阵
                    # 这里的公式变为：L = 100 * F * T^T
                    logits = l_scale * (fused_features @ target_image_features.t())

                    if torch.isnan(fused_features).any() or torch.isnan(target_image_features).any():
                         print("警告：输入特征检测到 NaN！")
                    # 5. 计算 CrossEntropy Loss
                    # ground_truth 是 [0, 1, 2, ..., batch_size-1]
                    ground_truth = torch.arange(logits.size(0), dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 在 src/combiner_train.py 的训练循环里
                if idx == 0:  # 每个 Epoch 只打印一次
                    for name, param in combiner.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # 观察这个均值是否在 Epoch 之间发生变化
                            print(f"DEBUG: Epoch {epoch} 层 {name} 权重均值: {param.data.mean().item():.12f}")
                            print(f"DEBUG: 梯度范数: {param.grad.norm().item():.10f}")
                            break

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            # 记录 Epoch 级别指标
            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        scheduler.step()

        # --- 验证阶段 (R@1, R@5, R@10 版) ---
        if epoch % validation_frequency == 0:
            clip_model = clip_model.float()
            with experiment.validate():
                combiner.eval()
                recalls_at1 = []
                recalls_at5 = []
                recalls_at10 = []

                for relative_val_dataset, index_features, index_names, dress_type in zip(
                        relative_val_datasets, index_features_list, index_names_list, val_dress_types):
                    
                    # 使用正确的合并函数引用（处理 DataParallel 包装）
                    combining_func = combiner.module.combine_features if isinstance(combiner, nn.DataParallel) else combiner.combine_features
                    
                    # 接收重构后的三个指标
                    r1, r5, r10 = compute_fiq_val_metrics(
                        relative_val_dataset, clip_model, index_features, index_names, combining_func)
                    
                    recalls_at1.append(r1)
                    recalls_at5.append(r5)
                    recalls_at10.append(r10)

                # 构造结果字典
                results_dict = {}
                for i, dress_type in enumerate(val_dress_types):
                    results_dict[f'{dress_type}_recall_at1'] = recalls_at1[i]
                    results_dict[f'{dress_type}_recall_at5'] = recalls_at5[i]
                    results_dict[f'{dress_type}_recall_at10'] = recalls_at10[i]
                
                # 计算全类别平均指标
                avg_r1 = sum(recalls_at1) / len(recalls_at1)
                avg_r5 = sum(recalls_at5) / len(recalls_at5)
                avg_r10 = sum(recalls_at10) / len(recalls_at10)
                
                # 综合评价指标：(R1 + R5 + R10) / 3
                current_average_recall = (avg_r1 + avg_r5 + avg_r10) / 3

                results_dict.update({
                    'average_recall_at1': avg_r1,
                    'average_recall_at5': avg_r5,
                    'average_recall_at10': avg_r10,
                    'average_recall': current_average_recall
                })

                print(f"\n--- Epoch {epoch} Validation Results ---")
                print(json.dumps(results_dict, indent=4))
                
                # 记录到实验平台
                experiment.log_metrics(results_dict, epoch=epoch)

                # 记录到 CSV 文件
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # --- 保存最优模型逻辑 ---
            if save_training:
                # 使用新的综合指标进行比较
                if save_best and current_average_recall > best_avg_recall:
                    best_avg_recall = current_average_recall
                    print(f"🥇 Found better model at epoch {epoch}, saving...")
                    save_model('combiner', epoch, combiner, training_path)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)
    
    # --- 方案 A 补丁开始：期末全量评估 (适配 R1, R5, R10) ---
    print("\n" + "="*40)
    print("🚀 所有训练轮次已完成，开始最终评估 (Final Validation)...")
    print("="*40)
    
    clip_model = clip_model.float()
    with experiment.validate():
        combiner.eval()
        recalls_at1 = []   # 新增
        recalls_at5 = []   # 新增
        recalls_at10 = []

        # 遍历所有类别进行验证
        for relative_val_dataset, index_features, index_names, dress_type in zip(
                relative_val_datasets, index_features_list, index_names_list, val_dress_types):
            
            combining_func = combiner.module.combine_features if isinstance(combiner, nn.DataParallel) else combiner.combine_features
            
            # 核心修复：这里必须解包 3 个值
            r1, r5, r10 = compute_fiq_val_metrics(
                relative_val_dataset, clip_model, index_features, index_names, combining_func)
            
            recalls_at1.append(r1)
            recalls_at5.append(r5)
            recalls_at10.append(r10)

        # 汇总所有类别的结果
        results_dict = {}
        for i, dress_type in enumerate(val_dress_types):
            results_dict[f'{dress_type}_recall_at1'] = recalls_at1[i]
            results_dict[f'{dress_type}_recall_at5'] = recalls_at5[i]
            results_dict[f'{dress_type}_recall_at10'] = recalls_at10[i]
        
        # 计算全类别平均分
        avg_r1 = sum(recalls_at1) / len(recalls_at1)
        avg_r5 = sum(recalls_at5) / len(recalls_at5)
        avg_r10 = sum(recalls_at10) / len(recalls_at10)
        final_avg_recall = (avg_r1 + avg_r5 + avg_r10) / 3 # 三指标平均
        
        results_dict.update({
            'average_recall_at1': avg_r1,
            'average_recall_at5': avg_r5,
            'average_recall_at10': avg_r10,
            'average_recall': final_avg_recall
        })

        print("\n✨ 最终期末评估结果:")
        print(json.dumps(results_dict, indent=4))
        
        # 记录到 Comet 和 CSV
        experiment.log_metrics(results_dict, epoch=num_epochs) 
        log_dict = {'epoch': num_epochs}
        log_dict.update(results_dict)
        validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
        validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

        # 如果最后一轮的表现刷写了纪录，保存它
        if save_training and final_avg_recall > best_avg_recall:
            print("🏆 最后一轮达到了历史最高精度，正在更新最优模型...")
            save_model('combiner', num_epochs, combiner, training_path)
            
    print(f"🏁 训练与评估全部结束。模型保存在: {training_path}")
    # --- 方案 A 补丁结束 ---


def combiner_training_cirr(projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                           combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str,
                           save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on CIRR dataset keeping frozen the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_cirr_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    clip_model = clip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the combiner and the train dataset
    # --- 修改后 ---
    # 1. 先实例化原始模型并移至 GPU
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    
    # 2. 增加 DataParallel 包装（核心步骤）
    if torch.cuda.device_count() > 1:
        print(f"🚀 检测到 {torch.cuda.device_count()} 张显卡！正在开启 DataParallel 多卡加速...")
        combiner = nn.DataParallel(combiner)
    
    # 3. 接着才是定义 Dataset 和 Loader
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    # ... 后面的 DataLoader 保持不变
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
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

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)
                text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)

                # Extract the features with CLIP
                with torch.no_grad():
                    reference_images_list = torch.split(reference_images, clip_bs)
                    reference_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in reference_images_list])
                    target_images_list = torch.split(target_images, clip_bs)
                    target_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in target_images_list])

                    text_inputs_list = torch.split(text_inputs, clip_bs)
                    text_features = torch.vstack(
                        [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list])

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    logits = combiner(reference_features, text_features, target_features)
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
            clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()

                # Compute and log validation metrics
                results = compute_cirr_val_metrics(relative_val_dataset, clip_model, val_index_features,
                                                   val_index_names, combiner.combine_features)
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

                # Save model
                if save_training:
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('combiner_arithmetic', epoch, combiner, training_path)
                    if save_best and results_dict['harmonic_mean'] > best_harmonic:
                        best_harmonic = results_dict['harmonic_mean']
                        save_model('combiner_harmonic', epoch, combiner, training_path)
                    if save_best and results_dict['geometric_mean'] > best_geometric:
                        best_geometric = results_dict['geometric_mean']
                        save_model('combiner_geometric', epoch, combiner, training_path)
                    if not save_best:
                        save_model(f'combiner_{epoch}', epoch, combiner, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    parser.add_argument("--dress-types", nargs='+', default=['shirt', 'dress', 'toptee'], help="fashionIQ categories")
    parser.add_argument("--retizero-base-path", type=str, default=None,
                        help="Path to base RetiZero.pth (used when --clip-model-path is a LoRA checkpoint)")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "retizero_base_path": args.retizero_base_path,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} combiner training",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'cirr':
        combiner_training_cirr(**training_hyper_params)

    elif args.dataset.lower() == 'fashioniq':
        # 将硬编码的服装类别替换为从命令行传入的 args.dress_types
        training_hyper_params.update(
            {'train_dress_types': args.dress_types, 'val_dress_types': args.dress_types})
        combiner_training_fiq(**training_hyper_params)