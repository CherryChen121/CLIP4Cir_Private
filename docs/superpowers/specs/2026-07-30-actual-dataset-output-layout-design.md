# CLIP4Cir 实际数据集输出分层设计

## 背景

CLIP4Cir 的数据加载接口使用 FashionIQ 格式，因此训练参数中的
`--dataset FashionIQ` 表示数据格式，而不一定表示实际数据来源。项目曾通过
`fashionIQ_dataset` 软链接在 UWF 与 IDRiD 数据集之间切换，后续又通过
`--fashioniq-root` 使用 `Combined_Fundus_CIR_Dataset`。现有输出因此全部位于
`outputs/fashioniq/`，混合了三个实际数据集。

当前 215 个运行均可从保留的运行证据可靠分类：

- IDRiD：172 个运行；
- UWF：41 个运行；
- Combined Fundus CIR：2 个运行；
- 原生 FashionIQ：0 个运行。

分类证据包括 Combiner 超参数中的 `train_dress_types`、`val_dress_types` 和
`fashioniq_root`，以及 CLIP 微调验证 CSV 中的数据集指标列。分类不依赖运行
日期、模型名称或人工猜测。

## 目标

1. 现有运行按实际数据集重组，不再混放于 `outputs/fashioniq/`。
2. 后续训练自动识别实际数据集，并允许使用显式参数覆盖。
3. 保留 FashionIQ 作为数据格式概念，同时在运行清单记录实际数据集与数据根
   路径。
4. 使用同文件系统原子重命名，不复制约 99 GB 权重。
5. 重分类前后验证普通文件大小与 SHA-256，不覆盖任何目标。
6. 删除不再需要的 5 份 legacy Excel，保留顶层迁移审计 CSV 与 JSON。

## 目标结构

```text
outputs/
├── migration_manifest.csv
├── migration_report.json
├── idrid/
│   ├── clip-finetune/
│   └── combiner/
├── uwf/
│   ├── clip-finetune/
│   └── combiner/
└── combined-fundus-cir/
    ├── clip-finetune/
    └── combiner/
```

未来若真正使用原生 FashionIQ 数据集，则允许创建
`outputs/fashioniq/<stage>/<model>/<run-id>/`。重分类完成后，由于当前没有
原生 FashionIQ 运行，旧的空 `outputs/fashioniq/` 必须删除。

## 现有运行分类

### Combiner

Combiner 运行存在 `training_hyperparameters.json`，按以下证据分类：

- `train_dress_types == ["IDRiD"]` 且验证指标为 `IDRiD_*`：`idrid`；
- `train_dress_types == ["CH", "CO", "NM", "RB", "RCH", "UM"]`：
  `uwf`；
- `train_dress_types == ["Internal"]` 且 `fashioniq_root` 解析为
  `/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset`：
  `combined-fundus-cir`。

当前结果为 118 个 IDRiD、31 个 UWF 和 2 个 Combined Fundus CIR Combiner
运行。

### CLIP 微调

历史 CLIP 微调运行缺少完整超参数文件，因此读取
`validation_metrics.csv` 表头：

- 存在 `IDRiD_recall_*` 列：`idrid`；
- 存在 `CH_recall_*`、`CO_recall_*`、`NM_recall_*`、`RB_recall_*`、
  `RCH_recall_*` 和 `UM_recall_*` 列：`uwf`；
- 存在 `Internal_recall_*` 列：`combined-fundus-cir`。

当前结果为 54 个 IDRiD 和 10 个 UWF CLIP 微调运行。

任何运行若证据缺失、证据互相冲突或目标路径已存在，均标记为未解决，并阻止
整次重分类执行。

## Legacy Excel 与审计文件

`outputs/reports/legacy/` 中的 5 份 Excel 是历史指标汇总，原始运行目录中的
`train_metrics.csv` 与 `validation_metrics.csv` 已完整保留，因此 Excel 不再
需要。重分类验证完成后删除这 5 个文件，并逐层移除空的
`outputs/reports/legacy/` 与 `outputs/reports/`。

保留：

- `outputs/migration_manifest.csv`；
- `outputs/migration_report.json`。

重分类后必须原子更新审计清单中的 `new_path` 和实际数据集字段。原始
`old_path`、文件大小、SHA-256 和失败实验记录保持不变；普通迁移文件的动作
状态保持不变。
`migration_report.json` 增加按实际数据集统计的运行数。

5 个被批准删除的 Excel 对应清单行不伪装为仍然存在：状态改为
`deleted-approved-report`，保留历史旧路径、新路径、大小与 SHA-256，并把
`reason` 设为 `user-approved-obsolete-legacy-summary`。验证器对该状态要求
目标文件不存在；其他 `moved` 或 `deduplicated` 行仍要求目标文件存在且哈希
匹配。

## 后续训练的数据集解析

两个训练入口都新增 `--output-dataset` 参数。实际输出数据集按以下优先级
解析：

1. 显式 `--output-dataset`；
2. `--fashioniq-root` 或默认数据根路径解析符号链接后的真实路径；
3. `--dress-types`。

自动映射规则：

- 路径或类别指向 IDRiD：`idrid`；
- 路径或类别指向 UWF：`uwf`；
- `Internal` 类别配合 `Combined_Fundus_CIR_Dataset`：
  `combined-fundus-cir`；
- 原生 FashionIQ 类别 `dress`、`shirt`、`toptee`：`fashioniq`。

显式覆盖值经过现有 slug 规范化后使用。没有显式覆盖时，如果路径证据与类别
证据冲突，或所有证据都无法识别，训练在创建运行目录前报错；不得静默回退到
`fashioniq`。

## 运行清单

新运行的 `run_manifest.json` 至少记录：

```json
{
  "dataset": "combined-fundus-cir",
  "dataset_slug": "combined-fundus-cir",
  "dataset_format": "fashioniq",
  "dataset_root_requested": "/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset",
  "dataset_root_resolved": "/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset"
}
```

对于默认的 `fashionIQ_dataset` 软链接，`dataset_root_requested` 记录项目内
软链接路径，`dataset_root_resolved` 记录训练开始时解析到的实际路径。这样即使
软链接以后改变，历史运行仍可追溯。

现有 215 个 `run_manifest.json` 在重分类时更新 `dataset`、`dataset_slug`、
`dataset_format`，并在证据可恢复时补充根路径字段。无法证明历史根路径时不
伪造路径，而是在 `dataset_classification_evidence` 字段中记录实际依据，例如
`validation_metrics:IDRiD_recall_at1` 或
`training_hyperparameters:train_dress_types=IDRiD`。

## 事务重分类

新增独立 dry-run/apply/verify/finalize 工作流：

1. 扫描 `outputs/fashioniq/` 的运行目录并生成重分类计划；
2. 记录每个普通文件的源路径、目标路径、大小、修改时间和 SHA-256；
3. 检查相关训练进程、符号链接、源文件变化、目标冲突和同文件系统约束；
4. 将运行原子移动到 `outputs/.dataset-reclassify-staging/`；
5. 在 staging 位置重新验证大小和 SHA-256；
6. 原子移动到最终数据集目录；
7. 原子更新运行清单与顶层迁移审计文件；
8. 从最终路径再次验证全部文件；
9. 删除批准移除的 5 份 Excel；
10. 逐层删除空的 staging、`reports/` 和 `fashioniq/` 目录。

在最终审计文件写入和文件验证完成前，不删除 Excel 或旧空目录。任一移动或
验证失败时，已移动运行回滚到原位置；回滚失败必须停止并保留 staging，禁止
继续清理。

## 并发与安全门槛

apply 前重新检查 `clip_fine_tune.py` 与 `combiner_train.py` 进程。只要存在
可能写入 `outputs/fashioniq/` 的训练，apply 拒绝执行。新训练代码完成并验证
之前，不启动历史重分类。

重分类不跟随运行目录内符号链接，不跨文件系统，不使用递归强制删除，不覆盖
任何已有目标。目录删除仅使用逐层 `rmdir`；Excel 删除使用计划中列出的 5 个
精确路径。

## 测试与验收

单元测试覆盖：

- IDRiD、UWF、Combined Fundus CIR 和原生 FashionIQ 自动识别；
- `--output-dataset` 显式覆盖与 slug 规范化；
- requested/resolved 数据根路径记录；
- 软链接目标解析；
- 路径与类别证据冲突时拒绝创建输出；
- 现有 Combiner 超参数和 CLIP 验证表头分类；
- 分类缺失、目标冲突、活跃写入与源文件变化拦截；
- staging 移动、失败回滚和审计文件原子更新；
- dry-run 不修改任何文件。

真实数据验收：

1. dry-run 恰好分类 172 个 IDRiD、41 个 UWF 和 2 个 Combined 运行；
2. 迁移前后 215 个运行和 201 个 checkpoint 均可对应；
3. 782 个保留的原迁移文件大小与 SHA-256 不变，另 5 个 Excel 在审计中明确
   标记为 `deleted-approved-report`，两者合计覆盖原 787 个动作；
4. 215 个运行清单的数据集字段与最终路径一致；
5. 两个顶层审计文件中除 `deleted-approved-report` 外的目标路径全部存在；
6. 5 份 Excel 已删除，`outputs/reports/` 不存在；
7. `outputs/fashioniq/` 与 `.dataset-reclassify-staging/` 不存在；
8. 两个训练入口的 smoke test 分别只在正确的实际数据集目录创建运行；
9. 最终重分类 dry-run 报告无待处理、未解决、冲突或错误项。
