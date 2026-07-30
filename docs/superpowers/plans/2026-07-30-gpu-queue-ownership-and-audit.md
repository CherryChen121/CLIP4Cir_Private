# GPU Queue Ownership and Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Combined training jobs running after they acquire a GPU, while recording a complete, readable audit of all eight GPUs every minute and explicit task start/end events.

**Architecture:** Add a small pure formatting module for GPU and task audit lines, expose the existing idle streak from `IdlePolicy`, and let `Dispatcher` emit one audited snapshot on every 60-second sample cycle. Remove runtime conflict termination from the dispatcher while retaining startup admission checks, process identity checks, task-failure pauses, and compatibility with historical `conflict_stopped` state.

**Tech Stack:** Python 3.9, dataclasses, `/proc`, `nvidia-smi`, pytest, user-level systemd.

## Global Constraints

- A new task still requires no compute PID, at most 512 MiB used memory, at most 5% utilization, five consecutive 60-second idle samples, and two final probes three seconds apart.
- A GPU can run at most one task from this queue.
- A foreign process arriving after our task starts must not pause the queue, change the task from `running`, or trigger any signal.
- The dispatcher must never signal a foreign process.
- Each 60-second audit contains GPU 0–7 exactly once and classifies each as `IDLE`, `FOREIGN`, `OURS`, or `SHARED`.
- Phase A must fully succeed before Phase B starts; Phase C remains excluded.
- A nonzero training exit, GPU probe failure, or unverifiable owned-process identity still pauses new dispatch.
- Do not modify or delete the historical run directory `gpu_queue_runs/20260730-091343-2279682`.
- Do not stage or modify the user-owned untracked files `IDRiD平均召回率汇总.xlsx` and `Related_Work_组合示例查询与医学跨模态检索.md`.

---

### Task 1: Pure GPU Audit Formatting and Idle-Streak Visibility

**Files:**
- Create: `src/gpu_queue_audit.py`
- Modify: `src/gpu_queue_core.py`
- Create: `tests/test_gpu_queue_audit.py`
- Modify: `tests/test_gpu_queue_core.py`

**Interfaces:**
- Consumes: `gpu_queue_core.GpuSnapshot`.
- Produces: `IdlePolicy.idle_streak(gpu_uuid: str) -> int`.
- Produces: `format_gpu_audit(snapshot: GpuSnapshot, *, idle_now: bool, idle_streak: int, running_task: Optional[dict], owned_compute_pids: Iterable[int], owner_lookup: Callable[[int], str]) -> str`.
- Produces: `format_task_start(task: dict) -> str`.
- Produces: `format_task_end(task: dict) -> str`.
- Produces: `format_queue_pause(reason: dict) -> str`.

- [ ] **Step 1: Write failing idle-streak tests**

Add to `tests/test_gpu_queue_core.py`:

```python
def test_idle_policy_exposes_streak_by_gpu_uuid():
    policy = IdlePolicy(expected_indices=range(8))
    snapshots = _snapshots()

    policy.observe(snapshots)
    policy.observe(snapshots)

    assert policy.idle_streak("GPU-0") == 2
    assert policy.idle_streak("GPU-never-seen") == 0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_core.py::test_idle_policy_exposes_streak_by_gpu_uuid -q
```

Expected: fail because `IdlePolicy` has no `idle_streak` method.

- [ ] **Step 3: Implement the minimal streak accessor**

Add to `IdlePolicy`:

```python
def idle_streak(self, gpu_uuid: str) -> int:
    return int(self._counts.get(gpu_uuid, 0))
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: one passing test.

- [ ] **Step 5: Write failing pure-format tests**

Create `tests/test_gpu_queue_audit.py` with focused cases that assert exact tokens:

```python
from src.gpu_queue_audit import (
    format_gpu_audit,
    format_queue_pause,
    format_task_end,
    format_task_start,
)
from src.gpu_queue_core import GpuSnapshot


def test_formats_idle_and_foreign_gpu_lines():
    idle = format_gpu_audit(
        GpuSnapshot(0, "GPU-0", 15, 0, ()),
        idle_now=True,
        idle_streak=3,
        running_task=None,
        owned_compute_pids=(),
        owner_lookup=lambda pid: "unused",
    )
    foreign = format_gpu_audit(
        GpuSnapshot(2, "GPU-2", 38111, 100, (1810854,)),
        idle_now=False,
        idle_streak=0,
        running_task=None,
        owned_compute_pids=(),
        owner_lookup=lambda pid: "jycheng",
    )

    assert idle == (
        "GPU_AUDIT gpu=0 uuid=GPU-0 status=IDLE "
        "memory_mib=15 util_percent=0 idle_streak=3/5"
    )
    assert "status=FOREIGN" in foreign
    assert "pids=1810854" in foreign
    assert "owners=jycheng" in foreign


def test_formats_ours_and_shared_gpu_lines():
    task = {"task_id": "A01", "pid": 101}
    ours = format_gpu_audit(
        GpuSnapshot(4, "GPU-4", 9000, 95, (202,)),
        idle_now=False,
        idle_streak=0,
        running_task=task,
        owned_compute_pids=(202,),
        owner_lookup=lambda pid: "qrchen",
    )
    shared = format_gpu_audit(
        GpuSnapshot(4, "GPU-4", 12000, 100, (202, 7777)),
        idle_now=False,
        idle_streak=0,
        running_task=task,
        owned_compute_pids=(202,),
        owner_lookup=lambda pid: "xtchen",
    )

    assert "status=OURS" in ours
    assert "task=A01 our_pid=101 our_pids=202" in ours
    assert "status=SHARED" in shared
    assert "foreign_pids=7777 foreign_owners=xtchen" in shared


def test_formats_task_and_pause_events():
    task = {
        "task_id": "A01",
        "phase": "A",
        "gpu_index": 4,
        "gpu_uuid": "GPU-4",
        "pid": 101,
        "log_path": "/tmp/A01/train.log",
        "status": "succeeded",
        "return_code": 0,
        "started_at": "2026-07-30T10:00:00+0800",
        "ended_at": "2026-07-30T10:02:30+0800",
    }

    assert format_task_start(task).startswith("TASK_START task=A01 phase=A gpu=4")
    assert "result=SUCCEEDED return_code=0" in format_task_end(task)
    assert "duration_seconds=150" in format_task_end(task)
    assert format_queue_pause({"kind": "task_failed", "task_id": "A01"}) == (
        "QUEUE_PAUSE kind=task_failed task=A01"
    )
```

- [ ] **Step 6: Run the audit tests and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_audit.py -q
```

Expected: collection fails because `src.gpu_queue_audit` does not exist.

- [ ] **Step 7: Implement the pure formatter**

Create `src/gpu_queue_audit.py`. It must:

- build deterministic `key=value` tokens in the order shown by the tests;
- use `IDLE` only when `running_task is None and idle_now`;
- use `FOREIGN` when no queue task owns the GPU and it is not idle;
- use `OURS` when all snapshot compute PIDs are in `owned_compute_pids`;
- use `SHARED` when at least one snapshot compute PID is outside `owned_compute_pids`;
- sort and deduplicate PID lists;
- catch owner lookup failures and emit `unknown`;
- emit `-` for an empty PID/owner list where that field is required;
- parse timestamps with `%Y-%m-%dT%H:%M:%S%z`, and emit `duration_seconds=-` if either timestamp is absent or malformed;
- map task states `succeeded`, `failed`, and `interrupted` to uppercase result values.

- [ ] **Step 8: Run focused and core tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_gpu_queue_audit.py tests/test_gpu_queue_core.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/gpu_queue_audit.py src/gpu_queue_core.py tests/test_gpu_queue_audit.py tests/test_gpu_queue_core.py
git commit -m "feat: format minute-level GPU audit records"
```

### Task 2: Warning-Only Sharing and Dispatcher Lifecycle Events

**Files:**
- Modify: `src/combined_gpu_queue.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Interfaces:**
- Consumes: Task 1 formatters and `IdlePolicy.idle_streak`.
- Produces: `Dispatcher._audit_snapshots(snapshots: Tuple[GpuSnapshot, ...]) -> None`.
- Produces: one `GPU_AUDIT_BEGIN`, eight `GPU_AUDIT`, and one `GPU_AUDIT_END` event per `sample_idle=True` cycle.
- Produces: immediate `TASK_START`, `TASK_END`, `QUEUE_PAUSE`, and `QUEUE_COMPLETE` events.

- [ ] **Step 1: Replace the old conflict test with a failing warning-only test**

Replace `test_unknown_compute_pid_pauses_and_stops_only_owned_task` in `tests/test_combined_gpu_queue.py`:

```python
def test_foreign_pid_after_launch_is_logged_without_stopping_owned_task(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    gpu = GpuSnapshot(3, "GPU-3", 2000, 80, (202, 7777))
    mark_running(
        state,
        "A01",
        {
            "pid": 101,
            "pgid": 101,
            "start_ticks": 10,
            "command_sha256": "owned",
            "manifest_path": str(tmp_path / "manifest.json"),
            "result_path": str(tmp_path / "result.json"),
            "log_path": str(tmp_path / "task.log"),
        },
        gpu,
        "start",
    )
    launcher = DispatcherLauncher()
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots({3: gpu})),
        idle_policy=EligiblePolicy([]),
        launcher=launcher,
        proc_inspector=Descendants({101: {202}}),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:00:00+0800",
        owner_lookup=lambda pid: "xtchen",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=True)

    assert launcher.terminated == []
    assert state["paused_reason"] is None
    assert next(t for t in state["tasks"] if t["task_id"] == "A01")["status"] == "running"
    assert any(
        "GPU_AUDIT gpu=3" in event
        and "status=SHARED" in event
        and "foreign_pids=7777" in event
        and "foreign_owners=xtchen" in event
        for event in events
    )
```

- [ ] **Step 2: Add failing complete-cycle audit tests**

Add tests that call `run_cycle(sample_idle=True)` with eight snapshots and assert:

```python
audit_lines = [event for event in events if event.startswith("GPU_AUDIT gpu=")]
assert len(audit_lines) == 8
assert [line.split("gpu=", 1)[1].split(" ", 1)[0] for line in audit_lines] == [
    str(index) for index in range(8)
]
assert events.count("GPU_AUDIT_BEGIN") == 1
assert events.count("GPU_AUDIT_END") == 1
```

Call `run_cycle(sample_idle=False)` in a separate test and assert it emits no `GPU_AUDIT_BEGIN` or `GPU_AUDIT` lines, proving the audit cadence remains 60 seconds while worker polling remains 30 seconds.

- [ ] **Step 3: Add failing lifecycle event tests**

Extend the existing launch, successful exit, failed exit, interrupted identity, and all-tasks-complete tests to assert:

```python
assert any(event.startswith("TASK_START task=A01 phase=A gpu=2") for event in events)
assert any("TASK_END task=A01" in event and "result=SUCCEEDED" in event for event in events)
assert any("TASK_END task=A01" in event and "result=FAILED" in event for event in events)
assert any(event.startswith("QUEUE_PAUSE kind=task_failed task=A01") for event in events)
assert events.count("QUEUE_COMPLETE") == 1
```

For `ProcessIdentityError`, assert `TASK_END ... result=INTERRUPTED` and `QUEUE_PAUSE kind=process_identity_unverified`.

- [ ] **Step 4: Run the new dispatcher tests and verify RED**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_combined_gpu_queue.py::test_foreign_pid_after_launch_is_logged_without_stopping_owned_task \
  tests/test_combined_gpu_queue.py -q
```

Expected: failures show that the current dispatcher still terminates the owned task and does not emit audit/lifecycle events.

- [ ] **Step 5: Implement warning-only sharing**

In `src/combined_gpu_queue.py`:

- remove the call to `_monitor_conflicts` from `run_cycle`;
- remove runtime use of `record_conflict` and `OwnedProcessLauncher.terminate`;
- retain `terminate` itself for backward API compatibility and its existing identity-safety tests;
- add `_owned_compute_pids(task, snapshot)` using `ProcInspector.descendants(root_pid) | {root_pid}`, intersected with `snapshot.compute_pids`;
- add `_audit_snapshots` that maps one running task to each assigned GPU, calls `format_gpu_audit`, and logs GPU 0–7 in order between begin/end markers;
- call `IdlePolicy.observe` before `_audit_snapshots` when dispatch is not paused, so the current minute's streak is visible;
- when already paused but another owned task is still running, continue minute audits without dispatching.

- [ ] **Step 6: Implement lifecycle events exactly once**

Use the existing `event_logger` injection:

- after `mark_running` and state persistence, emit `format_task_start`;
- after `record_exit` or `record_interrupted` and state persistence, emit `format_task_end`;
- whenever code first sets `paused_reason`, emit `format_queue_pause`;
- immediately before `run_forever` returns 0, emit `QUEUE_COMPLETE`;
- do not emit duplicate pause/end/complete events on later cycles.

Let logger write failures propagate so the dispatcher exits rather than continue unaudited dispatch.

- [ ] **Step 7: Run focused dispatcher tests and verify GREEN**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_gpu_queue.py -q
```

Expected: all dispatcher tests pass.

- [ ] **Step 8: Run the complete test suite**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
```

Expected: all tests pass; only the repository's pre-existing dependency warnings may remain.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/combined_gpu_queue.py tests/test_combined_gpu_queue.py
git commit -m "feat: keep acquired GPU jobs running with audit logs"
```

### Task 3: Runbook, Verification, and Fresh Production Queue

**Files:**
- Modify: `docs/combined_gpu_queue_runbook.md`
- Runtime only: the new timestamped directory created below `gpu_queue_runs/`

**Interfaces:**
- Consumes: Task 2 dispatcher behavior and log format.
- Produces: operator instructions for minute audits and shared-GPU semantics.
- Produces: a fresh user-level systemd transient service named `clip4cir-combined-gpu-queue.service`.

- [ ] **Step 1: Update the runbook**

Document:

- startup admission remains conservative;
- `OURS` and `SHARED` jobs continue running;
- `FOREIGN` GPUs cannot receive a new task;
- exact `GPU_AUDIT`, `TASK_START`, `TASK_END`, `QUEUE_PAUSE`, and `QUEUE_COMPLETE` meanings;
- `tail -f "$QUEUE_RUN_DIR/dispatcher.log"` after resolving `QUEUE_RUN_DIR` to the selected run directory;
- `conflict_stopped` is historical-only compatibility;
- normal task failures and unverifiable process identity still pause;
- the current service uses `KillMode=process`.

- [ ] **Step 2: Verify documentation and repository scope**

Run:

```bash
git diff --check
rg -n "SHARED|GPU_AUDIT|TASK_START|TASK_END|KillMode=process|historical" \
  docs/combined_gpu_queue_runbook.md
git status --short
```

Expected: no whitespace errors; the runbook contains every required term; the two user-owned untracked files remain unmodified.

- [ ] **Step 3: Commit the runbook**

```bash
git add docs/combined_gpu_queue_runbook.md
git commit -m "docs: explain persistent GPU ownership auditing"
```

- [ ] **Step 4: Run final automated verification**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
./run_combined_gpu_queue.sh dry-run
```

Expected: all tests pass, preflight reports 20 commands, all eight GPUs are listed, and dry-run launches no training process.

- [ ] **Step 5: Confirm no old dispatcher or owned training process is live**

Run:

```bash
systemctl --user show clip4cir-combined-gpu-queue.service \
  -p ActiveState -p SubState -p MainPID -p KillMode
pgrep -af '^/data0/qrchen/miniconda3/envs/clip4cir/bin/python /data0/qrchen/projects/CLIP4Cir/src/combined_gpu_queue\.py (--run|--resume)' || true
pgrep -af '^.*python .*src/(combiner_train|clip_fine_tune)\.py' || true
```

Expected: no active old dispatcher and no Combined training process owned by this queue.

- [ ] **Step 6: Start one fresh transient service**

Clear the failed transient unit, then create one new run:

```bash
systemctl --user reset-failed clip4cir-combined-gpu-queue.service
systemd-run --user \
  --unit=clip4cir-combined-gpu-queue.service \
  --description='CLIP4Cir Combined GPU queue with minute audit' \
  --property=KillMode=process \
  /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  /data0/qrchen/projects/CLIP4Cir/src/combined_gpu_queue.py --run
```

Do not start a second dispatcher if the read-only checks in Step 5 show an active instance.

- [ ] **Step 7: Verify service identity and first complete audit cycle**

Resolve the new run directory from the service command or newest `state.json`, then run:

```bash
QUEUE_RUN_DIR="$(find gpu_queue_runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
  | sort -nr | sed -n '1s/^[^ ]* //p')"
test -f "$QUEUE_RUN_DIR/state.json"
systemctl --user show clip4cir-combined-gpu-queue.service \
  -p ActiveState -p SubState -p MainPID -p KillMode -p ExecMainStatus
jq '{run_id, paused_reason, counts: (.tasks | group_by(.status) | map({status: .[0].status, count: length}))}' \
  "$QUEUE_RUN_DIR/state.json"
tail -n 40 "$QUEUE_RUN_DIR/dispatcher.log"
```

Wait only until one 60-second audit window has completed. Verify exactly one `GPU_AUDIT` line for each GPU 0–7 and confirm no task was launched unless its GPU had reached `idle_streak=5/5` and passed the two final probes.

- [ ] **Step 8: Record the production handoff**

Report:

- new run directory;
- service `ActiveState`, `MainPID`, and `KillMode`;
- current task counts and any running task/GPU mapping;
- dispatcher log path and task log paths;
- the first eight-card audit result;
- current GPU occupancy from `nvidia-smi`;
- confirmation that the old historical run was not modified.
