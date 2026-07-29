# Combined Fundus CIR Dataset 适配设计

## 目标

将 `/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset` 作为一个显式、隔离的数据集根目录接入 CLIP4Cir，使项目能够：

1. 使用 `Internal/train` 训练；
2. 使用 `Internal/val` 选择最佳模型；
3. 分别在 `Internal/test`、`ODIR5K/test`、`GRAPE/test` 上计算带真值的 Recall@1、Recall@5、Recall@10；
4. 保持现有 FashionIQ、UWF、IDRiD 默认行为可用；
5. 在 `命令.sh` 最顶部提供 Combined 的完整实验命令，每种模型与微调方式只运行一次，全部使用 GPU 0 和 `nohup` 后台机制。

## 数据划分

- 训练类别：`Internal`
- 验证类别：`Internal`
- 测试类别：`Internal`、`ODIR5K`、`GRAPE`
- ODIR5K 和 GRAPE 不参与训练或最佳模型选择。

## 参数与数据隔离

训练和验证入口增加显式数据集根目录参数 `--fashioniq-root`。程序在构造 FashionIQ 风格数据集前使用该参数限定根目录，不再依赖会扫描多个数据集根目录的全局类别发现。

类别仍通过 `--dress-types` 显式传入：

- 训练命令固定为 `--dress-types Internal`；
- 测试评估一次只指定一个类别，避免跨测试集混合统计。

没有传入 `--fashioniq-root` 时，现有环境变量及默认数据根目录解析方式保持不变。

## 带真值测试评估

现有 FashionIQ `test` 模式返回参考图像但忽略 `target`，适用于没有公开真值的挑战测试集。为了不破坏该行为，新增显式的带真值测试评估路径，而不全局改变原 test 返回结构。

带真值测试评估复用 val 的检索计算：

- gallery 来自对应类别的 `split.<category>.test.json`；
- query 来自 `cap.<category>.test.json`；
- 每条 query 返回 reference name、target name 和 caption；
- 计算目标图像在完整 gallery 中的排名；
- 输出 Recall@1、Recall@5、Recall@10 及三者平均值。

如果测试标注缺少 target、目标图像不在 gallery 或文件不存在，程序应给出包含类别和 split 的明确错误，而不是静默跳过。

## 命令矩阵

在 `命令.sh` 顶部新增独立的 Combined 区域，保留现有 UWF 和 IDRiD 区域不变。

Combined 包含 29 条训练命令：

- 8 条原始/预训练骨干 + Combiner；
- 10 条骨干微调；
- 11 条微调后骨干 + Combiner。

每种配置只保留一条命令：

- `CUDA_VISIBLE_DEVICES=0`
- `nohup`
- 独立日志文件
- `--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset`
- `--dress-types Internal`

依赖微调 checkpoint 的 Combiner 命令沿用分阶段执行设计：先完成骨干微调并整理最佳 checkpoint，再启动对应 Combiner。命令区注释必须明确不能把整个脚本一次性执行。

测试评估命令分别针对 Internal、ODIR5K、GRAPE，避免将三个域聚合成一个不透明指标。

## 兼容性

- 不修改现有 UWF/IDRiD 命令的数量、GPU 分配和参数；
- 不改变无真值 FashionIQ test 推理的返回结构；
- 新参数有默认值，旧命令无需新增参数即可继续工作；
- Combined 命令不依赖全局类别自动扫描。

## 验证标准

1. 显式根目录只发现 Combined 中指定的类别，不混入 UWF、IDRiD 或原 FashionIQ；
2. Internal train/val 可由两个训练入口实例化；
3. 三个 test 类别均能返回 reference、target、caption，并通过真实文件完整性检查；
4. 测试评估能对最小样本计算 R@1/R@5/R@10；
5. `命令.sh` 顶部 Combined 训练命令恰好 29 条；
6. 每个 Combined 配置恰好出现一次；
7. Combined 训练命令全部为 GPU 0、使用 `nohup`，并包含显式根目录与 Internal 类别；
8. 现有 Python 文件通过语法检查，新增及相关测试全部通过。
