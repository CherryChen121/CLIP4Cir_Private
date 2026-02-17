import torch
import sys

# 1. 模拟 PYTHONPATH，确保能搜到你的 adapter 和模块
sys.path.append(".") 

try:
    from src.retizero_adapter import RetiZeroAdapter
    print("✅ 成功找到 Adapter 类")

    # 2. 模拟加载逻辑
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data0/qrchen/projects/CLIP4Cir/pretrained_models/fashionIQ/RetiZero.pth" # 请修改为你的真实路径
    
    print(f"🔄 正在尝试加载权重并初始化模型...")
    clip_model = RetiZeroAdapter(model_path).to(device)
    clip_model.eval()
    
    # 3. 验证你手动挂载的属性
    # 模拟 combiner_train.py 里的初始化逻辑
    input_res = clip_model.visual.input_resolution
    out_dim = clip_model.visual.output_dim
    print(f"📊 接口对齐检查: 输入尺寸={input_res}, 输出维度={out_dim}")

    # 4. 模拟特征提取
    dummy_img = torch.randn(1, 3, input_res, input_res).to(device)
    dummy_text = ["这是一张眼底图"]
    
    with torch.no_grad():
        img_feats = clip_model.encode_image(dummy_img)
        # 注意：这里要测试你的 Adapter 是否内置了 tokenizer
        text_feats = clip_model.encode_text(dummy_text)

    print(f"✨ 图像特征形状: {img_feats.shape} (预期 [1, {out_dim}])")
    print(f"✨ 文本特征形状: {text_feats.shape} (预期 [1, {out_dim}])")
    
    if img_feats.shape[-1] == out_dim:
        print("🚀 结论：模型加载与维度对齐通过！")

except Exception as e:
    print(f"❌ 测试失败！错误信息: {e}")
    import traceback
    traceback.print_exc()