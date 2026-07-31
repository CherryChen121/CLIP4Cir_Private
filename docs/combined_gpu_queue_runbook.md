# Combined 安全 GPU 队列运行手册

本文将该工具简称为“GPU 队列”。

## 运行边界

调度器只读取 `命令.sh` 中 Combined Phase A 和 Phase B 的 20 条命令。Phase A 的 10 条任务全部处理到终态后开始 Phase B；单条失败不会阻止进入 Phase B。Phase C 永远不会自动启动。

服务器没有 Slurm，因此用户态调度器不能提供系统级独占保证。GPU 队列最多持有 4 张卡的租约，`running` 和 `cooldown` 都计入该上限。它不会终止或抢占其他用户进程。

## 1. 只读预检

```bash
cd /data0/qrchen/projects/CLIP4Cir
./run_combined_gpu_queue.sh dry-run
```

该命令会检查 20 条训练命令、数据与权重路径，显示四租约策略和 GPU 0–7 当前不可用的原因。它不创建运行状态，也不启动训练。

## 2. 后台启动

只有准备正式开始等待空闲 GPU 时才执行：

```bash
./run_combined_gpu_queue.sh start
```

包装脚本会输出调度器 PID 和 launcher 日志路径。

GPU 首次获取必须：

- 每 60 秒采样一次，连续 5 次没有 compute PID；
- 每次显存不超过 512 MiB；
- 每次 GPU 利用率不超过 5%；
- GPU 编号与 UUID 映射稳定；
- 派发前再通过两次间隔 3 秒的完整复查。

任务成功或非零退出后，原卡租约进入 60 秒冷却。冷却结束后必须同时满足无 compute PID、显存不超过 512 MiB、利用率不超过 5%、UUID 不变，并再次通过两次间隔 3 秒的复查，才会在原卡启动下一任务。

任何条件不满足都会释放租约。该卡以后若再次候选，必须重新累计 5 个一分钟空闲样本；调度器也可以用同一策略选择新卡。多个可用租约和候选卡均按物理 GPU 编号升序处理，任务始终按当前 Phase 的编号顺序派发。

## 3. 查看状态

```bash
tail -f /data0/qrchen/projects/CLIP4Cir/gpu_queue_runs/<run-id>/dispatcher.log
```

完整状态位于同目录的 `state.json`，每个任务的输出位于 `tasks/<task-id>/`。

查看服务和调度器进程：

```bash
systemctl --user show clip4cir-combined-gpu-queue.service \
  -p ActiveState -p SubState -p MainPID --no-pager
pgrep -af "combined_gpu_queue.py|gpu_queue_worker.py"
```

`dispatcher.log` 每分钟记录一个完整的八卡审计周期：

- `GPU_AUDIT_BEGIN` / `GPU_AUDIT_END`：一分钟审计周期的边界；
- `leases=3/4`：当前持有 3 个租约，最多 4 个；
- `lease=NONE` / `RUNNING` / `COOLDOWN`：该卡的租约状态；
- `IDLE`：当前瞬时空闲，并显示 `idle_streak=<次数>/5`；
- `FOREIGN`：没有本队列任务，但存在计算 PID、显存超限或利用率超限，不能派发新任务；
- `OURS`：本队列任务正在该卡运行；
- `SHARED`：本队列任务已经启动，随后同卡出现其他 PID；双方进程都不会被调度器终止；
- `GPU_LEASE_ACQUIRED`：首次通过五分钟策略取得该卡；
- `GPU_LEASE_COOLDOWN`：任务结束，原卡进入 60 秒冷却；
- `GPU_LEASE_REUSED`：冷却和双重复查通过，在原卡启动下一任务；
- `GPU_LEASE_RELEASED`：释放租约；`reason` 会说明外来 PID、显存、利用率、UUID、启动失败、身份中断或队列完成；
- `GPU_PROBE_ERROR`：本轮 GPU 查询失败，跳过本轮并在后续周期自动重试；
- `TASK_START`：任务编号、Phase、物理 GPU、监督 PID 和训练日志路径；
- `TASK_END`：任务结果、退出码、开始/结束时间和持续秒数；
- `QUEUE_COMPLETE`：20 个任务全部到达终态，并汇总成功、失败、启动失败和中断数量。

任务状态含义：

- `pending`：尚未派发；
- `running`：本队列任务正在运行；
- `succeeded`：退出码为 0；
- `failed`：训练进程非零退出，不自动重试，队列继续；
- `launch_failed`：监督 Worker 未安全启动，不自动重试，释放租约并继续；
- `interrupted`：无法验证原监督进程身份，不自动重试，释放租约并继续。

Phase A 的 `succeeded`、`failed`、`launch_failed`、`interrupted` 都是终态。全部 A 任务进入这些状态后，GPU 队列继续 Phase B。

## 4. 停止和恢复

向调度器 PID 发送普通 `SIGTERM` 只会停止调度器本身。每个训练任务处于独立进程组，不会因此自动终止：

```bash
kill <dispatcher-pid>
```

恢复前先检查 `state.json` 和日志，然后执行：

```bash
./run_combined_gpu_queue.sh resume /data0/qrchen/projects/CLIP4Cir/gpu_queue_runs/<run-id>
```

恢复会核对命令摘要、状态模式、租约上限与唯一性、PID、进程组、`/proc` 启动时间和命令身份。不确定的运行任务会标记为 `interrupted` 并释放其租约；其他任务继续，且不会重复启动该任务。

正式服务使用 `KillMode=process`。停止用户级 systemd 服务只停止调度器主进程，不会由 systemd 沿控制组终止已经启动的训练进程：

```bash
systemctl --user show clip4cir-combined-gpu-queue.service -p KillMode
```

输出必须为 `KillMode=process`。

## 5. 失败、探测错误或共享

调度器不自动重试失败、启动失败或中断任务，但会记录后继续处理剩余任务。后续人工检查对应任务日志、`dispatcher.log` 和 `state.json`，再决定是否单独重跑。

一次或连续多次普通 `nvidia-smi` 查询/解析失败只会产生 `GPU_PROBE_ERROR`。未知状态下不会派发、复用或释放；查询恢复后自动继续。

命令摘要或状态模式不匹配、原子状态写入失败、锁冲突、租约不变量损坏、GPU 编号—UUID 全局映射异常属于完整性错误，调度器会失败关闭并非零退出。这类错误需要人工检查，不会带着不可信状态继续派发。

当日志出现 `SHARED` 时，调度器只记录外来 PID 和用户名：

- 不停止本队列任务；
- 不设置队列暂停；
- 不向外来 PID 发送信号；
- 不在该 GPU 上派发第二个本队列任务。

如果外来 PID 后续消失，下一次分钟审计会从 `SHARED` 自动恢复为 `OURS`。

当前任务结束后仍进入 60 秒冷却。冷却检查时若外来 PID 还存在，就释放该卡租约；只有四项复用条件和双重复查都通过，才会继续在原卡派发。

调度器绝不会向未知 PID 或其他用户进程发送信号，也不会修改 NVIDIA compute mode、MIG、MPS、驱动或系统服务。

## 6. 本次交付状态

代码实现、测试和只读预检不会启动 GPU 队列。本次修改完成后服务仍保持停止状态；只有收到明确的后续运行命令后才允许执行 `start` 或 `resume`。
