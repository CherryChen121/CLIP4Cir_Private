import torch
import torch.nn as nn
import os
from iden_modules import CLIPRModel
from transformers import AutoTokenizer

class RetiZeroAdapter(nn.Module):
    def __init__(self, model_path):
        super().__init__() # super 应该在最前面
        
        # 1. 确定本地路径并校验
        local_bert_path = "/data0/qrchen/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b"

        if not os.path.exists(local_bert_path):
            raise FileNotFoundError(f"❌ 错误：路径不存在，请检查：{local_bert_path}")

        # 2. 加载本地 Tokenizer（只保留这一行，且必须带 local_files_only）
        print(f"🔄 正在从本地加载分词器: {local_bert_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_bert_path, local_files_only=True)
        
        # 3. 加载 RetiZero 核心模型
        # 注意：如果 CLIPRModel 内部也报错，可能需要进到 iden_modules 里修改它的 BERT 加载路径
        print(f"🔄 正在加载 RetiZero 权重: {model_path}")
        self.retizero = CLIPRModel(vision_type="lora", weights_path=model_path)
        
        # 4. 在真实 vision module 上补齐 CLIP4Cir 所需的元数据。
        self.retizero.vision_model.input_resolution = 224
        self.retizero.vision_model.output_dim = 512
        
        # 5. 设置温度系数 (CLIP 相似度计算必用)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    @property
    def visual(self):
        """Expose the real vision module through CLIP's conventional interface."""
        return self.retizero.vision_model

    def configure_cir_finetuning(self):
        """Train only vision LoRA weights and the two multimodal projection heads."""
        self.requires_grad_(False)

        vision_backbone = self.retizero.vision_model.model
        for layer in [*vision_backbone.w_As, *vision_backbone.w_Bs]:
            layer.requires_grad_(True)

        self.retizero.vision_model.projection_head_vision.requires_grad_(True)
        self.retizero.text_model.projection_head_text.requires_grad_(True)

        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def encode_image(self, image):
        # 修正：加上 .retizero 前缀
        features = self.retizero.vision_model(image) 
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, text):
        device = next(self.parameters()).device
        
        # 1. 核心修复：指定 max_length 为 77 或 512
        # FashionIQ 任务建议设为 77（对齐 CLIP 标准），如果你想保留更多信息可以设为 512
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77,  # 这里的数字必须 <= 512，设为 77 能显著加快训练速度
            return_tensors="pt"
        ).to(device)
        
        # 2. 移除 TextModel 不支持的参数
        inputs.pop('token_type_ids', None)
        
        # 3. 提取特征
        features = self.retizero.text_model(**inputs) 
        
        # 4. 归一化
        return features / features.norm(dim=-1, keepdim=True)
    @staticmethod
    def _strip_wrapper_prefixes(state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            while key.startswith("module.") or key.startswith("clip_model."):
                if key.startswith("module."):
                    key = key[len("module."):]
                elif key.startswith("clip_model."):
                    key = key[len("clip_model."):]
            cleaned[key] = value
        return cleaned

    def load_checkpoint(self, checkpoint_path):
        """Load either a native CLIP4Cir adapter or a legacy classifier LoRA checkpoint."""
        print(f"🔄 正在加载 RetiZero checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Unsupported RetiZero checkpoint object: {type(checkpoint).__name__}"
            )

        epoch = checkpoint.get("epoch", -1)
        if isinstance(checkpoint.get("RetiZeroAdapter"), dict):
            adapter_state = self._strip_wrapper_prefixes(
                checkpoint["RetiZeroAdapter"]
            )
            self.load_state_dict(adapter_state, strict=True)
            metric = checkpoint.get("average_recall", -1)
            print(
                "  ✅ 已恢复完整 RetiZeroAdapter "
                f"(epoch={epoch}, average_recall={metric})"
            )
            return epoch, metric

        legacy_state = checkpoint.get("state_dict")
        if isinstance(legacy_state, dict):
            vision_state = {
                key[len("img_encoder."):]: value
                for key, value in legacy_state.items()
                if key.startswith("img_encoder.")
            }
            if vision_state:
                self.retizero.vision_model.model.load_state_dict(
                    vision_state,
                    strict=True,
                )
                metric = checkpoint.get("mean_ACC", -1)
                print(
                    "  ✅ 已加载旧版分类 LoRA vision 权重 "
                    f"(epoch={epoch}, mean_ACC={metric})"
                )
                return epoch, metric

        keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise ValueError(
            "Unsupported RetiZero checkpoint format. "
            f"Expected 'RetiZeroAdapter' or legacy 'state_dict/img_encoder.*'; got keys: {keys}"
        )

    def load_lora_checkpoint(self, lora_checkpoint_path):
        """Backward-compatible alias for legacy callers."""
        return self.load_checkpoint(lora_checkpoint_path)
    # 在 RetiZeroAdapter 类内部添加这个方法
    def forward(self, x, mode):
        """
        这个 forward 函数是给 DataParallel 和 utils.py 调用的“开关”
        :param x: 输入的数据（图片 Tensor 或文本列表）
        :param mode: 'image' 或 'text'
        """
        if mode == 'image':
            return self.encode_image(x)
        elif mode == 'text':
            return self.encode_text(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. 必须是 'image' 或 'text'")
