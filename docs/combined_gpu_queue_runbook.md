# Combined 安全 GPU 队列运行手册

## 运行边界

调度器只读取 `命令.sh` 中 Combined Phase A 和 Phase B 的 20 条命令。Phase A 全部成功后才会开始 Phase B；Phase C 永远不会自动启动。

服务器没有 Slurm，因此用户态调度器不能提供系统级独占保证。它会连续审查显卡并在派发前复查；一旦本队列任务已经启动，后来进入同卡的其他进程只会被记录，不会导致本队列任务避让或队列暂停。

## 1. 只读预检

```bash
cd /data0/qrchen/projects/CLIP4Cir
./run_combined_gpu_queue.sh dry-run
```

该命令会检查 20 条训练命令、数据与权重路径，并显示 GPU 0–7 当前不可用的原因。它不创建运行状态，也不启动训练。

## 2. 后台启动

只有准备正式开始等待空闲 GPU 时才执行：

```bash
./run_combined_gpu_queue.sh start
```

包装脚本会输出调度器 PID 和 launcher 日志路径。调度器每 60 秒累计一次空闲样本；单卡连续 5 次无计算 PID、显存不超过 512 MiB、利用率不超过 5% 后，仍需通过两次间隔 3 秒的复查。

## 3. 查看状态

```bash
tail -f /data0/qrchen/projects/CLIP4Cir/gpu_queue_runs/<run-id>/dispatcher.log
```

完整状态位于同目录的 `state.json`，每个任务的输出位于 `tasks/<task-id>/`。

`dispatcher.log` 每分钟记录一个完整的八卡审计周期：

- `GPU_AUDIT_BEGIN` / `GPU_AUDIT_END`：一分钟审计周期的边界；
- `IDLE`：当前瞬时空闲，并显示 `idle_streak=<次数>/5`；
- `FOREIGN`：没有本队列任务，但存在计算 PID、显存超限或利用率超限，不能派发新任务；
- `OURS`：本队列任务正在该卡运行；
- `SHARED`：本队列任务已经启动，随后同卡出现其他 PID；双方进程都不会被调度器终止；
- `TASK_START`：任务编号、Phase、物理 GPU、监督 PID 和训练日志路径；
- `TASK_END`：任务结果、退出码、开始/结束时间和持续秒数；
- `QUEUE_PAUSE`：普通任务失败、GPU 查询失败或进程身份无法确认，停止派发新任务；
- `QUEUE_COMPLETE`：Phase A 和 Phase B 共 20 个任务全部成功。

任务状态含义：

- `pending`：尚未派发；
- `running`：本队列任务正在运行；
- `succeeded`：退出码为 0；
- `failed`：非零退出，队列停止派发；
- `conflict_stopped`：仅用于兼容旧运行记录；新队列不会再因为后来出现其他 PID 产生该状态；
- `interrupted`：无法验证原监督进程身份，队列暂停。

## 4. 停止和恢复

向调度器 PID 发送普通 `SIGTERM` 只会停止调度器本身。每个训练任务处于独立进程组，不会因此自动终止：

```bash
kill <dispatcher-pid>
```

恢复前先检查 `state.json` 和日志，然后执行：

```bash
./run_combined_gpu_queue.sh resume /data0/qrchen/projects/CLIP4Cir/gpu_queue_runs/<run-id>
```

恢复会核对命令摘要、PID、进程组、`/proc` 启动时间和命令身份。任何不确定状态都会标记为 `interrupted` 并暂停，不会重复启动任务。

正式服务使用 `KillMode=process`。停止用户级 systemd 服务只停止调度器主进程，不会由 systemd 沿控制组终止已经启动的训练进程：

```bash
systemctl --user show clip4cir-combined-gpu-queue.service -p KillMode
```

输出必须为 `KillMode=process`。

## 5. 失败或共享

调度器不会自动重试、跳过失败任务，也不会自动解除普通失败导致的暂停。已经正常运行的其他本队列任务会继续完成。出现任务失败或身份中断后，应先检查对应任务日志、`dispatcher.log` 和 `state.json`。

当日志出现 `SHARED` 时，调度器只记录外来 PID 和用户名：

- 不停止本队列任务；
- 不设置队列暂停；
- 不向外来 PID 发送信号；
- 不在该 GPU 上派发第二个本队列任务。

如果外来 PID 后续消失，下一次分钟审计会从 `SHARED` 自动恢复为 `OURS`。

调度器绝不会向未知 PID 或其他用户进程发送信号，也不会修改 NVIDIA compute mode、MIG、MPS、驱动或系统服务。
