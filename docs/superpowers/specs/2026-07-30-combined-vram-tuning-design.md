# Combined Internal 训练命令显存调优设计

## 目标

只调整 `命令.sh` 中 `COMBINED_COMMANDS_BEGIN` 与
`COMBINED_COMMANDS_END` 之间的 30 条训练命令，使其适配本机 8 张
49,140 MiB RTX 4090。训练峰值显存以约 38–42 GiB 为安全上限，保留
15–20% 余量，避免图像尺寸波动、验证阶段和 CUDA 内存碎片导致 OOM。

目标值是安全上限，不要求冻结骨干的轻量任务为了占满显存而显著改变
对比学习 batch，从而影响不同模型实验的可比性。

## 不变项

- 保留现有 GPU 分配，包括工作区中尚未提交的 GPU 1、2、3 分配。
- 保留模型、checkpoint、投影维度、隐藏维度、epoch、学习率、变换、
  验证频率、保存选项和日志文件名。
- 不修改 Python 训练代码，不增加梯度累积或自动显存探测逻辑。
- 不修改 Combined 区域以外的 UWF、IDRiD 等命令。

## 公共运行参数

每条训练命令增加 `CLIP4CIR_NUM_WORKERS=8`。训练代码在未设置该变量时
会为每个任务启动 128 个 DataLoader workers；多个任务并行时会造成
CPU 过度调度、epoch 启动延迟和 GPU 空转。每任务 8 个 workers 在并行
运行多条命令时为更稳健的默认值。

目标区域顶部 GPU 说明同步改为：本机 GPU 0–7 均为 49,140 MiB，实际卡号
以每条命令的 `CUDA_VISIBLE_DEVICES` 为准。

## Phase A：冻结骨干 + Combiner

CLIP 特征提取位于 `torch.no_grad()` 中，`batch-size` 决定 Combiner 的
对比学习 batch，`clip-bs` 决定冻结骨干每次前向的显存峰值。因此保持
现有 `batch-size`，主要提高 `clip-bs`。

| 模型 | batch-size | clip-bs |
| --- | ---: | ---: |
| OpenAI CLIP ViT-B/32 | 128 | 128 |
| OpenAI CLIP ViT-L/14 | 128 | 64 |
| BMC_CLIP_CF ViT-L/14 | 128 | 64 |
| RN50x4 Full FT checkpoint | 128 | 64 |
| RN50x4 No FT | 128 | 64 |
| EyeCLIP ViT-B/32 | 128 | 128 |
| RetiZero | 256 | 128 |
| BLIP ITM Large COCO，384 px | 256 | 32 |
| BLIP2 ITM ViT-G，364 px | 256 | 16 |
| BLIP ITM Base COCO，384 px | 256 | 64 |

BLIP2 ViT-G 和 RetiZero 保留原有 `clip-bs`，因为模型体量较大且当前值
已经属于该模型族的高吞吐档位；公共 workers 限制仍会应用到命令。

## Phase B：骨干微调

该阶段需要保存反向传播激活，只有 `batch-size` 可控制显存。参数按模型
体量和输入分辨率分档；学习率保持不变，避免同时改变优化策略。

| 模型 | batch-size |
| --- | ---: |
| OpenAI CLIP ViT-B/32 | 128 |
| OpenAI CLIP ViT-L/14 | 64 |
| BMC_CLIP_CF ViT-L/14 | 64 |
| RN50x4 Full FT checkpoint | 64 |
| RN50x4 No FT | 64 |
| EyeCLIP ViT-B/32 | 128 |
| RetiZero LoRA + projection | 32 |
| BLIP ITM Large COCO，384 px | 16 |
| BLIP2 ITM ViT-G，364 px | 4 |
| BLIP ITM Base COCO，384 px | 32 |

RN50x4 从 128 下调到 64，是因为其端到端反向传播和较高输入分辨率使
现有配置存在 OOM 风险；这项调整以稳定运行优先。

## Phase C：微调后骨干 + Combiner

Phase C 再次冻结骨干，因此与 Phase A 使用相同的模型族参数：

| 模型 | batch-size | clip-bs |
| --- | ---: | ---: |
| ViT-B/32 Fine-tuned | 128 | 128 |
| ViT-L/14 Fine-tuned | 128 | 64 |
| BMC_CLIP_CF Fine-tuned | 128 | 64 |
| RN50x4 Full FT Fine-tuned | 128 | 64 |
| RN50x4 No FT Fine-tuned | 128 | 64 |
| RetiZero LoRA Fine-tuned | 256 | 128 |
| EyeCLIP Fine-tuned | 128 | 128 |
| BLIP Fine-tuned，384 px | 256 | 32 |
| BLIP2 ITM ViT-G Fine-tuned，364 px | 256 | 16 |
| BLIP ITM Base COCO Fine-tuned，384 px | 256 | 64 |

## 验证

修改完成后执行以下静态验证：

1. `bash -n 命令.sh` 通过。
2. Combined 目标区域仍包含恰好 30 条活动训练命令。
3. 30 条命令均包含 `CLIP4CIR_NUM_WORKERS=8`。
4. Phase A/B/C 各 10 条，参数与上述矩阵逐项一致。
5. Combined 区域外没有产生改动。
6. 原有未提交 GPU 分配改动仍然保留。

实际显存峰值仍需在各模型首次运行的前 5–10 个 step 通过 `gpustat` 或
`nvidia-smi` 观察。若某个第三方 checkpoint 的实现显存行为与模型族
预期不同，应只回退该模型一个档位，不联动修改其他实验。
