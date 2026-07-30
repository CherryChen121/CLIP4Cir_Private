# Combined GPU 队列占用后持续运行与每分钟审计设计

## 背景

现有 Combined GPU 队列在新任务启动前采用保守审查：GPU 必须连续 5 次、每次间隔约 60 秒满足空闲条件，并在启动前进行两次相隔 3 秒的最终复查。任务启动后，现有实现若发现同卡出现不属于本任务进程树的 PID，会停止自己的任务、暂停整个队列并退出。

本次修改保留启动前的全部审查，但改变任务启动后的处理方式：一旦本队列已经在某张 GPU 上启动任务，后来进入同卡的其他进程不再触发避让。调度器不停止自己的任务、不暂停队列，也不向任何其他进程发送信号，只在每分钟审计日志中记录共享状态。

## 目标

- 继续避免主动把新任务派到已经被别人占用的 GPU。
- 我方任务启动后持续运行，直到正常结束、训练自身失败或调度器无法确认进程身份。
- 每分钟记录 8 张 GPU 的状态，明确显示空闲、他人占用、我方占用和共同占用。
- 明确记录每个任务何时启动、何时结束、使用哪张物理 GPU，以及最终结果。
- 保持 Phase A 全部成功后才进入 Phase B 的阶段屏障。

## 不在本次范围

- 不实现系统级 GPU 锁、Slurm 队列、NVIDIA 独占模式、MIG 或 MPS。
- 不终止、暂停或修改其他用户的任何进程。
- 不在一张 GPU 上派发两个本队列任务。
- 不自动恢复或续训已经被旧策略终止的 A01、A02。
- 不改变 Combined 的训练命令、模型、超参数或数据集内容。
- 不把 Phase C 加入自动队列。

## 方案选择

采用“统一调度器审计日志”方案：调度器现有的 60 秒空闲采样周期同时负责写入完整 GPU 审计快照，任务启动和结束则立即写入事件日志。

不采用仅输出原始 `nvidia-smi` 的最小方案，因为原始数据无法直接说明 PID 所属用户、我方任务编号或连续空闲计数。不采用独立监控服务，因为它会引入第二份轮询状态和额外的服务管理，没有必要。

## 启动前策略

新任务启动前继续执行以下全部条件：

- GPU 没有任何 compute PID；
- 显存占用不超过 512 MiB；
- GPU 利用率不超过 5%；
- GPU UUID 与物理编号映射稳定；
- 连续 5 次、每次间隔约 60 秒通过审查；
- 派发前再做两次完整复查，两次间隔 3 秒。

任一条件不满足时，该 GPU 的连续空闲计数清零。最终复查失败时只取消本次派发，不影响其他 GPU。

## 启动后策略

调度器用任务保存的 PID、进程组、`/proc` 启动时间和命令摘要确认我方监督进程身份，并用其后代 PID 集合识别我方训练进程。

任务启动后，同一 GPU 的审计状态可以是：

- `OURS`：GPU 上只有当前我方任务进程树；
- `SHARED`：GPU 上既有当前我方任务，也有一个或多个其他 PID。

出现 `SHARED` 时：

- 保持我方任务运行；
- 不改变任务的 `running` 状态；
- 不设置 `paused_reason`；
- 不调用进程终止接口；
- 不向其他 PID 发送任何信号；
- 记录其他 PID 及其用户名，用户名无法读取时写为 `unknown`。

GPU 已分配给本队列运行任务后，即使审计状态为 `SHARED`，也不能再领取第二个本队列任务。

训练进程自身非零退出仍标记为 `failed` 并暂停新任务派发。进程身份无法确认、GPU UUID 映射改变或 `nvidia-smi` 查询失败，仍采用失败关闭策略暂停新派发。

## 每分钟 GPU 审计日志

继续使用每次运行目录中的 `dispatcher.log`。每个约 60 秒的审计周期写入一个 `GPU_AUDIT_BEGIN`、8 条按 GPU 0–7 排序的 `GPU_AUDIT`，再写入一个 `GPU_AUDIT_END`。每一行独立带本地时区时间戳，便于直接使用 `tail -f` 查看。

每张 GPU 只使用以下四种状态：

- `IDLE`：当前满足瞬时空闲条件；
- `FOREIGN`：没有本队列任务，但存在 compute PID、显存超限或利用率超限；
- `OURS`：存在本队列任务，且没有识别到其他 PID；
- `SHARED`：存在本队列任务，同时识别到其他 PID。

日志字段：

- 所有状态：`gpu`、`uuid`、`status`、`memory_mib`、`util_percent`；
- `IDLE`：增加 `idle_streak=<当前次数>/5`；
- `FOREIGN`：增加可用的 `pids`、`owners`，并记录触发不可用的指标；
- `OURS`：增加 `task`、`our_pid`、`our_pids`；
- `SHARED`：在 `OURS` 字段基础上增加 `foreign_pids`、`foreign_owners`。

字段使用无空格的 `key=value` 形式；多个值用逗号分隔；空列表写为 `-`。用户名仅来自 `/proc/<pid>` 文件属主，不记录完整命令行或环境变量。

示例：

```text
2026-07-30T10:05:00+0800 GPU_AUDIT_BEGIN
2026-07-30T10:05:00+0800 GPU_AUDIT gpu=0 uuid=GPU-... status=IDLE memory_mib=15 util_percent=0 idle_streak=3/5
2026-07-30T10:05:00+0800 GPU_AUDIT gpu=2 uuid=GPU-... status=FOREIGN memory_mib=38111 util_percent=100 pids=1810854 owners=jycheng
2026-07-30T10:05:00+0800 GPU_AUDIT gpu=4 uuid=GPU-... status=OURS memory_mib=9120 util_percent=96 task=A01 our_pid=2400123 our_pids=2400123,2400140
2026-07-30T10:05:00+0800 GPU_AUDIT gpu=5 uuid=GPU-... status=SHARED memory_mib=22000 util_percent=100 task=A02 our_pid=2400456 our_pids=2400456,2400468 foreign_pids=2400789 foreign_owners=xtchen
2026-07-30T10:05:00+0800 GPU_AUDIT_END
```

审计日志用于解释调度决策，不替代 `state.json`。调度器重启后，每次运行只追加到对应运行目录的日志。

## 任务生命周期日志

任务状态发生变化时立即记录一行事件：

- `TASK_START`：`task`、`phase`、`gpu`、`gpu_uuid`、`pid`、`log_path`；
- `TASK_END`：`task`、`phase`、`gpu`、`result`、`return_code`、`started_at`、`ended_at`、`duration_seconds`；
- `QUEUE_PAUSE`：暂停原因及相关任务；
- `QUEUE_COMPLETE`：20 个任务全部成功时记录完成时间。

`TASK_END result` 取 `SUCCEEDED`、`FAILED` 或 `INTERRUPTED`。任务训练输出继续保存在各任务目录中的原日志文件，不复制到 `dispatcher.log`。

## 状态与兼容性

新策略不再产生新的 `conflict_stopped` 状态，但为兼容旧运行目录，状态解析仍认识该值。旧运行目录
`gpu_queue_runs/20260730-091343-2279682` 中的 A01、A02 已被终止，不能通过简单 `--resume` 变成完整结果。

实施并验证后应创建新的运行目录，从 A01 开始重新派发。旧目录只保留为历史记录，不修改、不删除。

## 错误处理

- `nvidia-smi` 查询或解析失败：记录 `QUEUE_PAUSE kind=gpu_probe_error`，暂停新派发。
- 运行任务身份无法确认：标记该任务为 `interrupted`，记录 `TASK_END` 和 `QUEUE_PAUSE`。
- 训练非零退出：标记为 `failed`，记录退出码，暂停新派发。
- 写入审计日志失败：让调度器以非零状态退出，不在不可审计状态下继续派发新任务。
- 查询其他 PID 用户名失败：日志用户名使用 `unknown`，不影响任务运行。
- 其他 PID 后续消失：下一次审计从 `SHARED` 自动恢复为 `OURS`，不产生任务状态变化。

## 测试策略

使用现有依赖注入和模拟快照，不启动真实训练，按测试驱动方式覆盖：

- 运行任务所在 GPU 出现其他 PID 时，任务保持 `running`、队列不暂停、终止接口不被调用；
- `SHARED` 日志包含任务编号、我方 PID、其他 PID 和用户名；
- 其他 PID 消失后，下一次日志状态恢复为 `OURS`；
- 没有我方任务的外来 PID 仍使 GPU 为 `FOREIGN`，连续空闲计数清零且不派发；
- 每个 60 秒采样周期恰好记录 GPU 0–7 共 8 条审计行；
- `IDLE` 日志包含准确的连续空闲次数；
- 任务启动记录 `TASK_START`；
- 成功、失败和身份中断分别记录对应的 `TASK_END`；
- 训练非零退出仍暂停后续派发；
- GPU 查询失败仍暂停；
- Phase A 到 Phase B 的屏障不变；
- 完整测试套件通过。

实施完成后执行一次真实只读预检，并启动新的 systemd 用户瞬态服务。启动后观察至少一个完整的 60 秒审计周期，确认日志同时覆盖 8 张 GPU；只有通过连续 5 次空闲审查的 GPU 才允许启动任务。

## 验收标准

- 我方任务启动后，同卡后来出现其他用户进程不会导致我方任务终止或队列暂停。
- 调度器从不向其他用户进程发送信号。
- `dispatcher.log` 每分钟完整记录 8 张卡，并能明确判断每张卡当时属于 `IDLE`、`FOREIGN`、`OURS` 或 `SHARED`。
- 日志能明确定位每个任务开始和结束的时间、物理 GPU、PID 和结果。
- 启动前的五次连续空闲审查与两次最终复查保持有效。
- 普通任务失败、GPU 查询失败和进程身份无法确认仍采用失败关闭策略。
- 新队列从 A01 开始；旧冲突运行目录保持不变。
