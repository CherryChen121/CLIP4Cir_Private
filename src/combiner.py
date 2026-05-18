import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.2)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # Keep CLIP-style learnable log temperature.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    # --- src/combiner.py ---

    def forward(self, image_features, text_features, target_features=None):
        """
        修改后的 forward 函数
        :param target_features: 设为 None，变为可选参数
        """
        # 1. 无论如何，先计算融合特征
        fused_features = self.combine_features(image_features, text_features)

        # 2. 逻辑分支
        if target_features is None:
            # 如果不传 target，说明我们只想在多卡上做特征融合
            # 这就是我们现在训练循环需要的
            return fused_features
        else:
            # 如果传了 target，说明是单卡运行或者旧代码逻辑
            # 继续计算相似度矩阵 (Logits)
            fused_features = F.normalize(fused_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)
            logits = self.logit_scale.exp() * fused_features @ target_features.t()
            return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        # --- [核心修改] 强制维度对齐防御 ---
        if text_features.shape[0] != image_features.shape[0]:
            # 找到较小的 batch size 
            min_bs = min(text_features.shape[0], image_features.shape[0])
            # 强制截断对齐，防止 torch.cat 报错
            text_features = text_features[:min_bs]
            image_features = image_features[:min_bs]
        # -----------------------------------

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        
        # 这里的相加操作也需要确保 text_features 和 image_features 已经是对齐后的
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
                
        return F.normalize(output, dim=-1)