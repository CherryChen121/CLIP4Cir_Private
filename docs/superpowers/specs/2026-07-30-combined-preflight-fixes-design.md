# Combined 命令预检修复设计

## 目标

修复 `命令.sh` 中 Combined 数据集 10×3 命令矩阵的三个预检问题，使 Phase A、Phase B 可以按顺序直接启动，同时保持 Phase C 与前两阶段的模型集合一致。

## 设计

1. RN50x4 No FT 使用 OpenAI 原始 `RN50x4` 初始化，不再传入实际只包含 Combiner 参数的 `clip_RN50x4_noft.pt`。
2. BLIP ITM Large 的 Phase A、Phase B 显式传入 `pytorch_model.bin`；模型配置与分词器仍从同一模型目录加载。
3. 移除不具备检索文本路径的 BLIP2 FLAN-T5-XXL，替换为本机完整缓存且已通过离线图文前向验证的 BLIP ITM Base COCO。
4. BLIP ITM Base 使用稳定项目路径 `pretrained_models/blip_itm_base_coco`。该路径链接到本机 Hugging Face 缓存快照，避免训练命令绑定缓存哈希目录。
5. Phase C 同步替换第十个模型，预期读取 Phase B 后整理出的 `pretrained_models/Combined/tuned_blip_itm_base_coco_best.pt`，继续保持 10×3=30。

## 不变项

- Combined 每种配置只运行 1 次。
- 全部命令使用 GPU 0、`nohup`、独立日志文件。
- 数据集固定为 `Combined_Fundus_CIR_Dataset` 的 `Internal` 类别。
- 原有 UWF 与 IDRiD 命令逐字节不变。
- 不自动启动任何训练任务。

## 验收

- 新增命令测试先在旧命令上失败，再在修复后通过。
- Combined 三个阶段各 10 条，FLAN 不再出现，BLIP ITM Base 每阶段恰好一条。
- RN50x4 No FT 的 Phase A、B 不含自定义 checkpoint。
- BLIP ITM Large 的 Phase A、B 指向实际权重文件。
- BLIP ITM Base 离线加载预训练投影头并完成一次图像/文本编码。
- 全量测试通过，shell 语法检查通过，旧命令区哈希不变。
