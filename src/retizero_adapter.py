import torch
import torch.nn as nn
import os
from iden_modules import CLIPRModel
from transformers import AutoTokenizer, AutoModel

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
        
        # 4. 【接口伪装】手动挂载 .visual 属性，对齐 CLIP4Cir 接口
        self.visual = type('', (), {})() 
        self.visual.input_resolution = 224
        self.visual.output_dim = 512 
        
        # 5. 设置温度系数 (CLIP 相似度计算必用)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

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