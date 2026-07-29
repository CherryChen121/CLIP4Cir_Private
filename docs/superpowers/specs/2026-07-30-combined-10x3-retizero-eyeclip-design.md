# Combined 10×3 RetiZero 与 EyeCLIP 补齐设计

## 目标

将 `命令.sh` 顶部的 Combined 实验矩阵整理为 10 个模型、3 个阶段、每种配置 1 次，共 30 条训练命令：

- Phase A：原始/预训练骨干 + Combiner，10 条；
- Phase B：骨干模型微调，10 条；
- Phase C：微调后骨干 + Combiner，10 条。

Combined 暂时移除 RETFound，补齐 EyeCLIP 的 Phase A，以及 RetiZero 的 Phase A、Phase B。现有 UWF、IDRiD 命令和旧版 `run_validate_lora.sh` 不变。

## 模型集合

三个阶段使用同一组 10 个模型：

1. OpenAI CLIP ViT-B/32；
2. OpenAI CLIP ViT-L/14；
3. BMC-CLIP；
4. OpenAI CLIP RN50x4 Full FT；
5. OpenAI CLIP RN50x4 No FT；
6. EyeCLIP；
7. RetiZero；
8. BLIP；
9. BLIP2 ITM；
10. BLIP2 FLAN-T5-XXL。

RETFound 只从 Combined 区域移除；不删除适配器代码，也不改 UWF/IDRiD 的历史命令。

## EyeCLIP

EyeCLIP 继续复用 OpenAI CLIP ViT-B/32 架构和现有通用 checkpoint 加载器。

- Phase A 新增 `eyeclip_clip4cir_vitb32.pt + Combiner`；
- Phase B 保留现有 EyeCLIP 微调命令；
- Phase C 保留现有 `tuned_eyeclip_best.pt + Combiner` 命令。

新增轻量测试确认 EyeCLIP 命令在 A/B/C 各出现一次，且 Phase A 使用原始 EyeCLIP 权重。

## RetiZero 原生 CIR 微调

`run_validate_lora.sh` 是旧 RetiZero 分类 LoRA checkpoint 的 UWF 验证脚本，不是 CIR 训练入口。其价值是明确旧 checkpoint 的格式：`state_dict` 中的 `img_encoder.*` 映射到 RetiZero vision encoder。

新的 Phase B 直接接入 `clip_fine_tune.py`，使用 Internal CIR triplet 目标：

- reference image 与 modification text 做逐元素相加；
- 与 target image 计算 batch 内对比损失；
- 冻结基础 ViT；
- 冻结 BioClinicalBERT 主体；
- 训练 vision LoRA 参数、vision projection、text projection；
- 保存完整 `RetiZeroAdapter` checkpoint，供 Phase C 无损恢复。

RetiZero 适配器提供真实的 `visual` 访问接口和 224/512 元数据，不再使用没有参数的临时空对象。RetiZero 接受原始字符串列表，由适配器内部 tokenizer 处理。

## RetiZero checkpoint 兼容

适配器提供一个统一加载入口，支持两种格式：

1. 旧分类 LoRA checkpoint：顶层 `state_dict`，只读取 `img_encoder.*`，忽略 classifier；
2. 新 CLIP4Cir CIR checkpoint：顶层 `RetiZeroAdapter`，恢复完整适配器状态。

Phase A 只传 `--retizero-base-path pretrained_models/RetiZero.pth`，冻结 RetiZero 并训练 Combiner。Phase B 从同一 base 权重开始做 CIR 微调。Phase C 同时传 base 权重与 `pretrained_models/Combined/tuned_retizero_best.pt`，先构造架构，再恢复完整 CIR checkpoint，最后冻结骨干训练 Combiner。

## 命令约束

Combined 30 条命令全部满足：

- `CUDA_VISIBLE_DEVICES=0`；
- `nohup python`；
- `--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset`；
- `--dress-types Internal`；
- 每个日志文件唯一，且以 `run1_combined_` 开头；
- Phase A/B/C 各 10 条；
- Combined 区域不出现 RETFound。

Phase B 最佳模型仍需人工整理到 `pretrained_models/Combined/` 后再启动 Phase C。RetiZero 的目标文件名统一为 `tuned_retizero_best.pt`。

## 验证标准

1. RetiZero 模型名由微调入口识别，不再落入 `clip.load("RetiZero")`；
2. RetiZero 使用原始文本输入；
3. RetiZero CIR 微调仅开放 LoRA 和双模态 projection 参数；
4. 新完整 checkpoint 和旧 `img_encoder.*` checkpoint 均可加载；
5. Combined 命令矩阵恰好 30 条，A/B/C 各 10 条；
6. Combined 无 RETFound，EyeCLIP 和 RetiZero 在每个阶段各一条；
7. UWF/IDRiD 命令后缀逐字节不变；
8. Python 语法检查、Shell 语法检查及完整测试通过。
