# Combined 安全 GPU 队列运行手册

## 运行边界

调度器只读取 `命令.sh` 中 Combined Phase A 和 Phase B 的 20 条命令。Phase A 全部成功后才会开始 Phase B；Phase C 永远不会自动启动。

服务器没有 Slurm，因此用户态调度器不能提供系统级独占保证。它会连续审查显卡、派发前复查，并在发现未知计算 PID 时只停止自己的冲突任务。

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

任务状态含义：

- `pending`：尚未派发；
- `running`：本队列任务正在运行；
- `succeeded`：退出码为 0；
- `failed`：非零退出，队列停止派发；
- `conflict_stopped`：同卡发现未知 PID，已停止自己的任务并暂停；
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

## 5. 失败或冲突

调度器不会自动重试、跳过失败任务，也不会自动解除暂停。已经正常运行的其他本队列任务会继续完成。出现失败、冲突或中断后，应先检查对应任务日志和 `state.json`，再决定后续处理。

调度器绝不会向未知 PID 或其他用户进程发送信号，也不会修改 NVIDIA compute mode、MIG、MPS、驱动或系统服务。
