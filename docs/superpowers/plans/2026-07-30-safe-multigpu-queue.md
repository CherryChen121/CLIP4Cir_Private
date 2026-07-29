# Safe Combined Multi-GPU Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, resumable user-space dispatcher that starts Combined Phase A and Phase B tasks only on GPUs proven idle, without signaling or sharing a card with another user's process.

**Architecture:** Keep parsing/policy/state logic in a side-effect-free core module and isolate `nvidia-smi`, `/proc`, process groups, locking, sleeps, and CLI behavior in a runtime module. The dispatcher reads the authoritative commands from `命令.sh`, persists every transition atomically, applies a Phase A→B barrier, and launches only after five idle samples plus two final probes.

**Tech Stack:** Python 3.9 standard library, Bash, `nvidia-smi`, Linux `/proc`, POSIX process groups and `flock`, pytest

## Global Constraints

- 使用多卡队列：多张显卡通过审查时，可以同时运行多个任务。
- Phase A 的 10 个任务按原顺序派发。
- Phase A 全部成功后，才允许派发 Phase B 的 10 个任务。
- Phase C 不加入本队列。
- 任一任务非零退出后，停止派发新任务；已经运行的任务继续完成。
- 调度器实现与验证期间不启动训练。真实派发必须显式使用 `--run`。
- 没有任何 compute PID、显存不超过 512 MiB、利用率不超过 5%，连续 5 个 60 秒样本后才成为候选。
- 派发前执行两次完整复查，两次间隔 3 秒。
- 冲突处理只能向本调度器创建且身份核验成功的进程组发送信号。
- 不修改 NVIDIA compute mode、MIG、MPS、驱动或系统服务。
- 不自动重试失败任务，不自动启动 Phase C，不覆盖已有训练日志。

---

## File Map

- `src/gpu_queue_core.py`: immutable queue/GPU records, strict command parsing, preflight validation, idle counters, state transitions, and atomic JSON persistence.
- `src/combined_gpu_queue.py`: `nvidia-smi` probing, `/proc` identity inspection, owned process groups, dispatcher loop, global lock, and CLI.
- `src/gpu_queue_worker.py`: shell-free per-task supervisor that writes an atomic exit-result file so dispatcher restart does not lose the training exit code.
- `run_combined_gpu_queue.sh`: explicit `dry-run`, `start`, and `resume` wrapper; only `start`/`resume` use `nohup`.
- `tests/test_gpu_queue_core.py`: pure unit tests for parsing, preflight, idle policy, stage gating, and state persistence.
- `tests/test_combined_gpu_queue.py`: injected-fake tests for probes, launches, final rechecks, conflict handling, locking, resume, and CLI safety.
- `.gitignore`: ignore `gpu_queue_runs/`.
- `docs/combined_gpu_queue_runbook.md`: operator commands, state meanings, stop semantics, and recovery procedure.

### Task 1: Strictly parse and preflight the authoritative queue

**Files:**
- Create: `src/gpu_queue_core.py`
- Create: `tests/test_gpu_queue_core.py`
- Read: `命令.sh`

**Interfaces:**
- Produces `TaskSpec(task_id: str, phase: str, ordinal: int, argv: tuple[str, ...], env: dict[str, str], log_name: str, source: str)`.
- Produces `QueueSpec(tasks: tuple[TaskSpec, ...], command_sha256: str)`.
- Produces `parse_combined_queue(command_file: Path, project_root: Path, python_executable: Path) -> QueueSpec`.
- Produces `preflight_queue(queue: QueueSpec, project_root: Path) -> None`, raising `PreflightError` with all discovered errors.
- Test-local helpers: `valid_combiner_line: str`, `write_command_fixture(tmp_path: Path, line: str) -> Path`, `make_task(argv: tuple[str, ...]) -> TaskSpec`, and `make_queue(*tasks: TaskSpec) -> QueueSpec`.

- [ ] **Step 1: Write parser tests that define the safe input grammar**

Add tests equivalent to:

```python
def test_real_combined_queue_contains_only_ordered_phase_a_and_b():
    queue = parse_combined_queue(COMMAND_FILE, PROJECT_ROOT, PYTHON)
    assert len(queue.tasks) == 20
    assert [task.task_id for task in queue.tasks[:2]] == ["A01", "A02"]
    assert [task.phase for task in queue.tasks] == ["A"] * 10 + ["B"] * 10
    assert all("Phase C" not in task.source for task in queue.tasks)
    assert len({task.log_name for task in queue.tasks}) == 20
    assert all(task.argv[0] == str(PYTHON) for task in queue.tasks)


def test_parser_rejects_shell_operators_outside_fixed_redirection():
    command_file = write_command_fixture(
        tmp_path,
        valid_line.replace(" --dataset", " ; touch /tmp/unsafe --dataset"),
    )
    with pytest.raises(PreflightError, match="unsupported shell token"):
        parse_combined_queue(command_file, tmp_path, PYTHON)
```

Also cover duplicate/missing markers, a phase count other than 10, duplicate logs, a non-`src/combiner_train.py` or non-`src/clip_fine_tune.py` entry point, missing `nohup`, and missing final `2>&1 &`.

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py -k 'queue or parser' -q
```

Expected: collection fails because `gpu_queue_core` does not exist.

- [ ] **Step 3: Implement the immutable records and strict parser**

Use `dataclasses.dataclass(frozen=True)`, `Path.read_text(encoding="utf-8")`, `shlex.split`, and `hashlib.sha256`. Accept only this token structure:

```text
[KEY=VALUE] [KEY=VALUE] nohup python src/<allowed-entry>.py <arguments> > <unique-log>.log 2>&1 &
```

Replace the `python` token with `python_executable`, remove `nohup`/redirection/background tokens from `argv`, and retain environment assignments except that GPU assignment remains an overridable `env` field. Never pass the source line to a shell.

- [ ] **Step 4: Write failing path and option preflight tests**

Add tests equivalent to:

```python
def test_preflight_rejects_missing_absolute_path(tmp_path):
    task = make_task(argv=(str(PYTHON), "src/combiner_train.py",
                           "--fashioniq-root", "/missing/dataset"))
    with pytest.raises(PreflightError, match="/missing/dataset"):
        preflight_queue(make_queue(task), PROJECT_ROOT)


def test_preflight_rejects_unknown_cli_option(tmp_path):
    task = make_task(argv=(str(PYTHON), "src/combiner_train.py",
                           "--definitely-unknown", "1"))
    with pytest.raises(PreflightError, match="--definitely-unknown"):
        preflight_queue(make_queue(task), PROJECT_ROOT)
```

- [ ] **Step 5: Implement aggregated preflight validation**

Parse `parser.add_argument` calls whose first positional argument is a constant string beginning with `--` from each entry point with `ast`; compare every command option against that set. Validate the Python executable, entry points, dataset roots, `--clip-model-path`, `--retizero-base-path`, `--retfound-backbone-path`, and absolute local `--blip-model-name` values. Collect every error and raise one `PreflightError` without starting either training script.

- [ ] **Step 6: Run Task 1 tests and commit**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py -q
git add src/gpu_queue_core.py tests/test_gpu_queue_core.py
git commit -m "feat: parse and preflight Combined GPU queue"
```

Expected: Task 1 tests pass; no Python training process is created.

### Task 2: Implement fail-closed GPU probing and idle qualification

**Files:**
- Modify: `src/gpu_queue_core.py`
- Create: `src/combined_gpu_queue.py`
- Modify: `tests/test_gpu_queue_core.py`
- Create: `tests/test_combined_gpu_queue.py`

**Interfaces:**
- Produces `GpuSnapshot(index: int, uuid: str, memory_used_mib: int, utilization_percent: int, compute_pids: tuple[int, ...])`.
- Produces `NvidiaSmiProbe.snapshot() -> tuple[GpuSnapshot, ...]`, raising `ProbeError`.
- Produces `IdlePolicy.observe(snapshots: tuple[GpuSnapshot, ...]) -> tuple[int, ...]`.
- Produces `IdlePolicy.is_idle_now(snapshot: GpuSnapshot) -> bool`.
- Test-local helpers: `fake_nvidia_smi(gpu_rows: list[str], process_rows: list[str])`, `eight_gpus(gpu0: tuple[int, int, tuple[int, ...]])`, and `eight_idle_gpus()`.

- [ ] **Step 1: Write failing `nvidia-smi` parsing tests**

Inject a `runner(argv) -> CompletedProcess` fake and test:

```python
def test_probe_keeps_zero_utilization_gpu_busy_when_compute_pid_exists():
    probe = NvidiaSmiProbe(runner=fake_nvidia_smi(
        gpu_rows=["0, GPU-a, 49140, 0"],
        process_rows=["GPU-a, 1790625, ray::WorkerDict, 14898"],
    ))
    gpu = probe.snapshot()[0]
    assert gpu.utilization_percent == 0
    assert gpu.compute_pids == (1790625,)
```

Also cover blank process output, `N/A`, duplicate GPU UUID, an unknown process UUID, nonzero command exit, malformed integers, and a missing GPU index.

- [ ] **Step 2: Run probe tests and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_gpu_queue.py -k probe -q
```

Expected: FAIL because `NvidiaSmiProbe` is absent.

- [ ] **Step 3: Implement `NvidiaSmiProbe`**

Issue exactly two argv-based subprocess calls with no shell:

```text
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

Require indices `0..7`, map compute PIDs by UUID, sort by index, and raise `ProbeError` for every incomplete or ambiguous result. A probe error must never return a partially usable GPU list.

- [ ] **Step 4: Write failing consecutive-idle tests**

Add tests equivalent to:

```python
@pytest.mark.parametrize(
    "memory,util,pids",
    [(513, 0, ()), (0, 6, ()), (0, 0, (42,))],
)
def test_idle_policy_fails_closed(memory, util, pids):
    policy = IdlePolicy(expected_indices=range(8))
    snapshots = eight_gpus(gpu0=(memory, util, pids))
    assert 0 not in policy.observe(snapshots)


def test_gpu_becomes_candidate_only_on_fifth_consecutive_sample():
    policy = IdlePolicy(expected_indices=range(8))
    for _ in range(4):
        assert 0 not in policy.observe(eight_idle_gpus())
    assert 0 in policy.observe(eight_idle_gpus())
```

Test that a busy sample resets the counter and that an index→UUID mapping change raises `ProbeError`.

- [ ] **Step 5: Implement `IdlePolicy` and commit**

Defaults must be `memory_limit_mib=512`, `utilization_limit_percent=5`, and `required_samples=5`. Eligibility uses `<=` for both numeric limits and requires an empty PID tuple. Keep counters by UUID and return eligible physical indices sorted ascending.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py -q
git add src/gpu_queue_core.py src/combined_gpu_queue.py tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py
git commit -m "feat: audit GPU occupancy conservatively"
```

### Task 3: Add atomic queue state and the Phase A→B barrier

**Files:**
- Modify: `src/gpu_queue_core.py`
- Modify: `tests/test_gpu_queue_core.py`

**Interfaces:**
- Produces `AtomicStateStore(path: Path).load() -> dict` and `.save(state: dict) -> None`.
- Produces `initial_state(queue: QueueSpec, run_id: str, created_at: str) -> dict`.
- Produces `next_pending_task(state: dict) -> dict | None`.
- Produces `mark_running(state: dict, task_id: str, launch: dict, gpu: GpuSnapshot, started_at: str) -> None`.
- Produces `record_exit(state: dict, task_id: str, return_code: int, ended_at: str) -> None`.
- Produces `record_conflict(state: dict, task_id: str, unknown_pids: list[int], detected_at: str) -> None`.
- Produces `record_interrupted(state: dict, task_id: str, detected_at: str) -> None`.
- Produces `validate_resume_state(state: dict, queue: QueueSpec) -> None`.
- Test-local helpers: constants `NOW` and `LATER`, plus `queue_with_two_tasks_per_phase()`, `state_with_running(*task_ids: str)`, `owned_process()`, `mark_running(state: dict, task_id: str, launch: dict)`, `task(state: dict, task_id: str)`, and `queue_with_digest(digest: str)`.

- [ ] **Step 1: Write failing state-machine tests**

Cover exact transitions:

```python
def test_phase_b_is_blocked_until_every_phase_a_task_succeeds():
    state = initial_state(queue_with_two_tasks_per_phase(), "run-1", NOW)
    assert next_pending_task(state)["task_id"] == "A01"
    mark_running(state, "A01", owned_process())
    record_exit(state, "A01", 0, LATER)
    assert next_pending_task(state)["task_id"] == "A02"
    mark_running(state, "A02", owned_process())
    record_exit(state, "A02", 0, LATER)
    assert next_pending_task(state)["task_id"] == "B01"


def test_nonzero_exit_pauses_new_dispatch_but_not_running_tasks():
    state = state_with_running("A01", "A02")
    record_exit(state, "A01", 7, LATER)
    assert state["paused_reason"]["kind"] == "task_failed"
    assert task(state, "A02")["status"] == "running"
    assert next_pending_task(state) is None
```

Also test allowed status names, no automatic retry, Phase C absence, and completion only after all B tasks succeed.

- [ ] **Step 2: Run state tests and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py -k 'state or phase or exit' -q
```

- [ ] **Step 3: Implement state construction and transitions**

Store only JSON values. Each running task records physical GPU, GPU UUID, PID, process group, `/proc` start ticks, command digest, start time, end time, return code, manifest path, result path, log path, and conflict details. `next_pending_task` returns the lowest ordinal eligible task and returns `None` while paused.

- [ ] **Step 4: Write failing persistence and resume tests**

Verify:

```python
def test_atomic_save_replaces_complete_json_and_leaves_no_temp_file(tmp_path):
    store = AtomicStateStore(tmp_path / "state.json")
    store.save({"version": 1, "tasks": []})
    assert store.load()["version"] == 1
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_resume_rejects_changed_command_digest():
    state = {"command_sha256": "old"}
    with pytest.raises(ResumeError, match="command digest"):
        validate_resume_state(state, queue_with_digest("new"))
```

Also reject unsupported schema versions, missing task IDs, and duplicate tasks.

- [ ] **Step 5: Implement atomic persistence and resume validation**

Create the temporary file in the same directory, serialize the state with `json.dump(state, handle, ensure_ascii=False, indent=2)`, flush, `os.fsync`, then `os.replace`. On load, reject malformed JSON. Resume validation must compare schema version, command digest, task IDs, phases, ordinals, argv, and log names.

- [ ] **Step 6: Run Task 3 tests and commit**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py -q
git add src/gpu_queue_core.py tests/test_gpu_queue_core.py
git commit -m "feat: persist staged GPU queue state"
```

### Task 4: Launch and terminate only identity-verified owned processes

**Files:**
- Modify: `src/combined_gpu_queue.py`
- Create: `src/gpu_queue_worker.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Interfaces:**
- Produces `ProcessIdentity(pid: int, pgid: int, start_ticks: int, command_sha256: str)`.
- Produces `LaunchRecord(identity: ProcessIdentity, manifest_path: Path, result_path: Path, log_path: Path)`.
- Produces `ProcInspector.identity(pid: int) -> ProcessIdentity | None`.
- Produces `ProcInspector.descendants(pid: int) -> set[int]`.
- Produces `OwnedProcessLauncher.start(task: TaskSpec, gpu: GpuSnapshot, task_dir: Path) -> LaunchRecord`.
- Produces `OwnedProcessLauncher.poll(identity: ProcessIdentity, result_path: Path) -> int | None`.
- Produces `OwnedProcessLauncher.terminate(identity: ProcessIdentity, grace_seconds: int = 30) -> bool`.
- Produces `gpu_queue_worker.run_manifest(manifest_path: Path) -> int`, which atomically writes `result.json`.
- Test-local helpers: pytest fixture `task`, plus `recording_popen`, `launcher_factory(popen_factory)`, `gpu(index: int)`, `changed_identity`, `record_kill`, and `original_identity`.

- [ ] **Step 1: Write failing process-identity and launch tests**

Test with an injected `popen_factory`, `/proc` reader, clock, sleeper, `killpg`, and `getpgid`:

```python
def test_launcher_uses_argv_new_session_and_selected_gpu(tmp_path):
    launcher = launcher_factory(popen_factory=recording_popen)
    launch = launcher.start(task, gpu(index=6), tmp_path / "A01")
    call = recording_popen.calls[-1]
    assert call.shell is False
    assert call.start_new_session is True
    assert call.env["CUDA_VISIBLE_DEVICES"] == "6"
    assert call.env["NCCL_P2P_DISABLE"] == "1"
    assert launch.result_path == tmp_path / "A01" / "result.json"


def test_terminate_refuses_signal_when_proc_identity_changed():
    launcher = OwnedProcessLauncher(proc_inspector=changed_identity, killpg=record_kill)
    assert launcher.terminate(original_identity) is False
    assert record_kill.calls == []
```

Also ensure an existing log file causes launch rejection and the task source is never sent through `bash -c`.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_gpu_queue.py -k 'launcher or identity or terminate' -q
```

- [ ] **Step 3: Implement `/proc` inspection and owned launch**

Read `/proc/<pid>/stat` fields for parent PID and start ticks, `/proc/<pid>/cmdline` for a SHA-256 command identity, and enumerate numeric `/proc` directories for descendants. Before launch, atomically create a per-task `manifest.json` containing the exact argv, cwd, only the `CUDA_VISIBLE_DEVICES` and `NCCL_P2P_DISABLE` environment overrides, exclusive log path, and result path. Never serialize the complete inherited environment. Start the worker with `subprocess.Popen([python, worker_script, "--manifest", manifest], shell=False, cwd=project_root, start_new_session=True)`.

- [ ] **Step 4: Write and implement worker result tests**

Run the worker only with short test commands such as:

```python
[sys.executable, "-c", "raise SystemExit(7)"]
```

Assert it uses `shell=False`, opens the log exclusively, returns 7, and atomically writes:

```json
{"return_code": 7}
```

An invalid manifest, existing log, or existing result file must fail without starting the child command.

`OwnedProcessLauncher.poll` returns the validated integer `return_code` once the atomic result file exists, returns `None` while the exact worker identity is alive, and raises `ProcessIdentityError` if the worker disappeared without a valid result.

- [ ] **Step 5: Implement identity-gated termination**

Before every signal, re-read PID identity and require exact PID, PGID, start ticks, and command digest match. Send `SIGTERM` only to the verified owned PGID, wait up to 30 seconds, then reverify before optional `SIGKILL`. If verification fails, send no signal and return `False`.

- [ ] **Step 6: Run Task 4 tests and commit**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_gpu_queue.py -q
git add src/combined_gpu_queue.py src/gpu_queue_worker.py tests/test_combined_gpu_queue.py
git commit -m "feat: manage only owned GPU process groups"
```

### Task 5: Orchestrate multi-GPU dispatch, conflict pause, and recovery

**Files:**
- Modify: `src/combined_gpu_queue.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Interfaces:**
- Produces `Dispatcher.run_cycle(sample_idle: bool) -> None`.
- Produces `Dispatcher.run_forever() -> int`.
- Consumes injected `probe`, `idle_policy`, `launcher`, `state_store`, `clock`, and `sleep`.
- Test-local helpers: `make_dispatcher`, `pending_phase_a_state`, `all_idle`, `gpu2_with_foreign_pid`, `running_dispatcher`, and `dispatcher_with_exit_codes`.

- [ ] **Step 1: Write failing dispatch-order and final-recheck tests**

Use fakes; no real subprocesses:

```python
def test_multiple_idle_gpus_receive_tasks_in_gpu_and_queue_order():
    dispatcher = make_dispatcher(
        eligible_gpus=[gpu(2), gpu(5)],
        state=pending_phase_a_state(),
        final_probe_sequences=[all_idle(), all_idle()],
    )
    dispatcher.run_cycle(sample_idle=True)
    assert dispatcher.launcher.assignments == [("A01", 2), ("A02", 5)]


def test_second_final_probe_becoming_busy_cancels_launch():
    dispatcher = make_dispatcher(
        eligible_gpus=[gpu(2)],
        final_probe_sequences=[all_idle(), gpu2_with_foreign_pid(9001)],
    )
    dispatcher.run_cycle(sample_idle=True)
    assert dispatcher.launcher.assignments == []
```

Assert the final probes are separated by an injected `sleep(3)`.

- [ ] **Step 2: Run dispatch tests and verify RED**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_gpu_queue.py -k 'dispatch or final_probe' -q
```

- [ ] **Step 3: Implement ordered multi-GPU dispatch**

On a 60-second idle-sampling cycle, call `IdlePolicy.observe`. Remove GPUs already assigned to running tasks. For each eligible GPU in ascending order, re-fetch the next eligible task, run two `is_idle_now` probes separated by 3 seconds, and launch only if both snapshots preserve the same UUID and remain idle. Persist after every assignment.

- [ ] **Step 4: Write failing monitor and conflict tests**

Cover:

```python
def test_unknown_compute_pid_pauses_queue_and_stops_only_owned_task():
    dispatcher = running_dispatcher(
        gpu_compute_pids={3: {owned_pid, 7777}},
        owned_descendants={owned_pid, worker_pid},
    )
    dispatcher.run_cycle(sample_idle=False)
    assert dispatcher.launcher.terminated == [owned_identity]
    assert dispatcher.state["paused_reason"]["kind"] == "foreign_gpu_process"
    assert dispatcher.state["paused_reason"]["unknown_pids"] == [7777]
    assert task(dispatcher.state, "A01")["status"] == "conflict_stopped"


def test_failed_task_pauses_but_other_owned_jobs_keep_running():
    dispatcher = dispatcher_with_exit_codes({"A01": 1, "A02": None})
    dispatcher.run_cycle(sample_idle=False)
    assert task(dispatcher.state, "A01")["status"] == "failed"
    assert task(dispatcher.state, "A02")["status"] == "running"
    assert dispatcher.launcher.terminated == []
```

Also cover successful exits, all-A barrier, all-B completion, probe failure pause, unverified termination pause, and 30-second monitor versus 60-second idle-sample cadence.

- [ ] **Step 5: Implement monitoring and fail-closed behavior**

Poll each worker/result pair before dispatch. Compare each GPU's compute PIDs against `ProcInspector.descendants(root_pid) | {root_pid}`. Log PID owners with `pwd.getpwuid(os.stat("/proc/<pid>").st_uid)`; an unreadable owner is `"unknown"`. On conflict, pause first, persist, then call identity-gated termination. Never call termination for a normal nonzero exit or another running task.

- [ ] **Step 6: Write and implement recovery reconciliation**

Tests must prove:

- exact live identity remains `running`;
- missing or mismatched identity becomes `interrupted` and pauses;
- a valid resumed run never starts a duplicate process;
- a changed command digest is rejected before probing GPUs.

Implement `reconcile_resume()` before entering the loop and persist any `interrupted` transition.

- [ ] **Step 7: Run Task 5 tests and commit**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py -q
git add src/combined_gpu_queue.py tests/test_combined_gpu_queue.py
git commit -m "feat: dispatch Combined jobs with safety barriers"
```

### Task 6: Add the locked CLI, nohup wrapper, and operator runbook

**Files:**
- Modify: `src/combined_gpu_queue.py`
- Create: `run_combined_gpu_queue.sh`
- Modify: `.gitignore`
- Create: `docs/combined_gpu_queue_runbook.md`
- Modify: `tests/test_combined_gpu_queue.py`

**Interfaces:**
- CLI modes: `--dry-run`, `--run`, and `--resume RUN_DIR`, mutually exclusive.
- Runtime root: `gpu_queue_runs/<YYYYmmdd-HHMMSS-pid>/`.
- Global lock: `gpu_queue_runs/dispatcher.lock`.
- Test-local helpers: `fake_dependencies(tmp_path: Path)` and `DispatcherLock`.

- [ ] **Step 1: Write failing CLI and lock tests**

Test `build_parser()` mutual exclusion and injected `main()` dependencies:

```python
def test_dry_run_never_calls_launcher_or_creates_run_state(tmp_path):
    dependencies = fake_dependencies(tmp_path)
    result = main(["--dry-run"], dependencies=dependencies)
    assert result == 0
    assert dependencies.launcher.calls == []
    assert not list(tmp_path.glob("*/state.json"))


def test_second_dispatcher_lock_is_rejected(tmp_path):
    first = DispatcherLock(tmp_path / "dispatcher.lock")
    first.acquire()
    with pytest.raises(AlreadyRunningError):
        DispatcherLock(tmp_path / "dispatcher.lock").acquire()
```

Also test that `--run` creates a unique directory, `--resume` requires an existing `state.json`, and dry-run reports all eight occupied simulated GPUs as unavailable.

- [ ] **Step 2: Implement the CLI and lock**

Use `argparse` mutually exclusive required modes. Acquire a nonblocking `fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)` for `--run` and `--resume`; hold the file descriptor for the dispatcher lifetime and write its PID to the lock file. Dry-run parses, preflights, probes once, prints a table with reasons, and exits without creating state or calling the launcher.

- [ ] **Step 3: Create the explicit Bash wrapper**

Implement these accepted forms:

```bash
./run_combined_gpu_queue.sh dry-run
./run_combined_gpu_queue.sh start
./run_combined_gpu_queue.sh resume /absolute/path/to/gpu_queue_runs/<run-id>
```

The wrapper must use `/data0/qrchen/miniconda3/envs/clip4cir/bin/python`, `cd` to the project root, create `gpu_queue_runs/`, and use `nohup` only for `start` and `resume`. It prints the dispatcher PID and launcher log path. Reject every other argument shape. Do not source `命令.sh`.

- [ ] **Step 4: Ignore runtime artifacts and write the runbook**

Add exactly `gpu_queue_runs/` to `.gitignore`. Document:

- dry-run review;
- background start;
- `tail -f` for `dispatcher.log` and task logs;
- `state.json` status meanings;
- stopping the dispatcher with `SIGTERM` leaves already running training process groups alone;
- resume validation;
- failure/conflict requires manual review;
- Phase C remains manual;
- user-space race limitation.

- [ ] **Step 5: Run CLI, shell, and full regression tests**

```bash
bash -n run_combined_gpu_queue.sh
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py -q
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass and no training process starts.

- [ ] **Step 6: Commit the operator interface**

```bash
git add src/combined_gpu_queue.py run_combined_gpu_queue.sh .gitignore docs/combined_gpu_queue_runbook.md tests/test_combined_gpu_queue.py
git commit -m "feat: expose safe Combined GPU queue controls"
```

### Task 7: Perform a real read-only safety audit and integrate

**Files:**
- Verify: `命令.sh`
- Verify: `gpu_queue_runs/` remains untracked/ignored

**Interfaces:**
- Consumes: completed CLI and current `nvidia-smi` state.
- Produces: evidence that dry-run rejects occupied GPUs without launching training.

- [ ] **Step 1: Capture the process baseline**

Run:

```bash
pgrep -af 'src/(combiner_train|clip_fine_tune)\\.py' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

Save the output for comparison; do not signal any PID.

- [ ] **Step 2: Run the real dry-run only**

```bash
./run_combined_gpu_queue.sh dry-run
```

Expected with the currently observed server state: 20 commands pass preflight, all GPUs with compute PIDs report unavailable, no run state is created, and no training command starts.

- [ ] **Step 3: Compare the process baseline and inspect state**

Re-run the baseline commands. Assert no new `combiner_train.py` or `clip_fine_tune.py` PID appeared, `gpu_queue_runs/` contains no `state.json`, and no other user's PID received a signal.

- [ ] **Step 4: Run verification-before-completion**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
bash -n run_combined_gpu_queue.sh
git diff --check
git status --short
```

Review the branch diff and verify only the planned files changed.

- [ ] **Step 5: Finish the branch using the user's established integration choice**

Use `superpowers:finishing-a-development-branch`. The user previously chose local merge to `main`; merge only after the suite is green, verify again on merged `main`, preserve the two existing untracked user files, then remove the owned worktree and merged feature branch.

Do not start `--run` as part of integration. Starting the real dispatcher requires a separate explicit user instruction after the completed dry-run report.
