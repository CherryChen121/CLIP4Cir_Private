import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



def extract_index_features(dataset, clip_model):
    # --- 修正后的特征维度获取逻辑 ---
    
    # 1. 穿透包装层
    actual_clip = clip_model
    if hasattr(actual_clip, 'module'):
        actual_clip = actual_clip.module  # 处理 DataParallel
    
    # 2. 尝试获取维度 (增加对 RetiZero 的兼容)
    if hasattr(actual_clip, 'visual'):
        # 官方 CLIP 路径
        feature_dim = actual_clip.visual.output_dim
    elif hasattr(actual_clip, 'retizero'):
        # 如果还没穿透 Adapter，直接从这里拿
        feature_dim = actual_clip.retizero.vision_model.projection_head_vision.projection.out_features
    elif hasattr(actual_clip, 'vision_model'):
        # 已经剥到了 CLIPRModel 这一层
        feature_dim = actual_clip.vision_model.projection_head_vision.projection.out_features
    else:
        # 万能保险：如果都找不到，根据你之前的测试结果，强行赋值 512
        print("⚠️ 无法检测到特征维度，正在使用预设值 512")
        feature_dim = 512

    print(f"📊 提取器最终确认维度: {feature_dim}")

    # 2. 初始化容器为 None
    index_features = None  
    index_names = []
    
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)

    # 3. 提取特征
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            # 这里的 batch_features 是真实跑出来的 [batch_size, 512]
            batch_features = clip_model(images, mode='image')
            
            # --- 核心黑科技：第一次拿到特征时才建立容器 ---
            if index_features is None:
                real_dim = batch_features.shape[1] # 获取真实的维度，这里绝对会是 512
                print(f"✨ 动态对齐成功！识别到真实特征维度为: {real_dim}")
                index_features = torch.empty((0, real_dim)).to(device, non_blocking=True)
            # -------------------------------------------

            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)

    return index_features, index_names

def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    
    # 去掉 DataParallel 层（如果有的话）
    temp_model = model_to_save
    if hasattr(temp_model, 'module'):
        temp_model = temp_model.module
    
    # 检查是否是 Combiner 模型
    if hasattr(temp_model, 'combiner_layer'):
        # 这是 Combiner 模型，直接保存
        state_dict = temp_model.state_dict()
        model_name = temp_model.__class__.__name__
    elif hasattr(temp_model, 'clip_model'):
        # 这是包含 CLIP 的模型（CLIPWrapper），提取 CLIP 权重
        actual_clip_model = temp_model.clip_model
        state_dict = actual_clip_model.state_dict()
        model_name = actual_clip_model.__class__.__name__
    else:
        # 其他情况，直接保存模型权重
        state_dict = temp_model.state_dict()
        model_name = temp_model.__class__.__name__

    torch.save({
        'epoch': cur_epoch,
        model_name: state_dict,
    }, str(models_path / f'{name}.pt'))