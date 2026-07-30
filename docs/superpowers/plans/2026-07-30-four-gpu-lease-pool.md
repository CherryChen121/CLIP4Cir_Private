# Four-GPU Lease Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the Combined Phase A/B scheduler to a resilient pool of at most four persistent GPU leases, with safe first acquisition, 60-second same-GPU reuse, isolated task failures, and partial-success completion.

**Architecture:** Persist lease state beside task state and make the dispatcher process transitions in a fixed order: reconcile running jobs, probe once, audit, process ready cooldown leases, then acquire new leases from five-sample candidates. Core helpers own task/lease invariants; the dispatcher owns I/O and transition orchestration; audit helpers own stable, inspectable log lines.

**Tech Stack:** Python 3, pytest, `nvidia-smi`, JSON atomic state, systemd user service wrapper. Tests use fake probes, clocks, process inspectors, launchers, and state stores; they must never launch training.

## Global constraints

- Do not modify `命令.sh`; it contains the user's current hyperparameter edits.
- Do not start or resume the GPU queue, systemd service, or any training command.
- Never terminate or signal another user's process.
- Preserve the existing five one-minute samples and two final checks three seconds apart for first acquisition.
- Count both `running` and `cooldown` leases toward the four-GPU limit.
- Keep Phase C outside this queue.
- Use a new schema version and reject old state rather than guessing a migration.
- Keep atomic state write, command digest, lock, mapping, and lease-invariant errors fail-closed.
- Commit each coherent task and run the listed focused tests before continuing.

### Task 1: Encode terminal task states and lease invariants

**Files:**

- Modify: `src/gpu_queue_core.py`
- Modify: `tests/test_gpu_queue_core.py`

**Interfaces:**

```python
SCHEMA_VERSION = 2
MAX_GPU_LEASES = 4
PHASE_TERMINAL_STATUSES = {
    "succeeded", "failed", "launch_failed", "interrupted",
}

def lease_for_gpu(state: dict, gpu_index: int) -> Optional[dict]: ...
def acquire_lease(state: dict, gpu: GpuSnapshot, task_id: str, acquired_at: str) -> dict: ...
def mark_lease_cooldown(
    state: dict,
    task_id: str,
    previous_task_id: str,
    cooldown_ready_at: float,
    updated_at: str,
) -> dict: ...
def release_lease(state: dict, gpu_index: int) -> dict: ...
def record_launch_failed(
    state: dict,
    task_id: str,
    detail: str,
    ended_at: str,
    gpu: GpuSnapshot,
) -> None: ...
def validate_lease_state(state: dict) -> None: ...
def all_tasks_terminal(state: dict) -> bool: ...
def terminal_summary(state: dict) -> dict[str, object]: ...
```

- [ ] Write failing core state tests.

Add tests that assert:

- initial state has schema version 2, an empty `leases` list, and per-task `error=None`;
- a nonzero exit becomes `failed` without setting `paused_reason`;
- `record_launch_failed` records terminal status, error, GPU identity, and end time;
- `record_interrupted` is terminal without setting `paused_reason`;
- Phase B opens when every Phase A task is any allowed terminal status;
- Phase B remains blocked while any Phase A task is pending or running;
- acquiring a fifth lease raises `ResumeError`;
- duplicate lease GPU index, UUID, or task association raises `ResumeError`;
- running leases and running tasks must be one-to-one;
- cooldown timestamps must be numeric and finite;
- summary counts each terminal state and reports task IDs.

- [ ] Run the focused tests and confirm RED.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py -q
```

Expected: failures for schema 2, lease helpers, and non-pausing failure behavior.

- [ ] Implement the minimal state machine.

Implementation details:

- Add `leases=[]` in `initial_state`.
- Add `error=None` to each task.
- Keep `paused_reason` only for fatal scheduler conditions and compatibility with fail-closed exit.
- Make `record_exit` and `record_interrupted` update only the task.
- Make `record_launch_failed` accept only a pending task and store a single-line error detail.
- Make `next_pending_task` choose Phase A until all A tasks are terminal, then Phase B.
- Represent cooldown deadlines as Unix epoch seconds to allow exact clock injection.
- Validate lease count, legal states, unique indices/UUIDs/task IDs, required fields, and running task/lease correspondence.
- Call `validate_lease_state` from `validate_resume_state`.

- [ ] Run focused tests and confirm GREEN.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py -q
```

- [ ] Commit.

```bash
git add src/gpu_queue_core.py tests/test_gpu_queue_core.py
git commit -m "feat: add persistent GPU lease state"
```

### Task 2: Add four-slot acquisition and 60-second same-GPU reuse

**Files:**

- Modify: `src/combined_gpu_queue.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Dispatcher additions:**

```python
Dispatcher(
    ...,
    clock=time.time,
    cooldown_seconds=60,
    max_gpu_leases=MAX_GPU_LEASES,
)

def _cooldown_leases(self) -> list[dict]: ...
def _process_ready_cooldowns(self, snapshots: tuple[GpuSnapshot, ...]) -> None: ...
def _dispatch_on_gpu(
    self,
    pending: dict,
    gpu: GpuSnapshot,
    *,
    reuse: bool,
) -> bool: ...
def _release(self, lease: dict, reason: str) -> None: ...
```

- [ ] Write failing scheduler tests for the lease cap and acquisition order.

Add tests that prove:

- eight eligible GPUs start only four tasks and persist exactly four running leases;
- candidates are considered in GPU index order and tasks in queue order;
- running plus cooldown leases together enforce the cap;
- leased GPUs are excluded from fresh five-minute acquisition;
- a failed final check does not create a lease;
- an available fifth card may be acquired only after an existing lease is released.

- [ ] Run the cap tests and confirm RED.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_combined_gpu_queue.py -q -k "lease or four or fifth or final"
```

- [ ] Implement capped first acquisition.

Implementation details:

- Preserve the existing `IdlePolicy.observe` five-sample qualification.
- After the two final checks, launch the task and atomically persist its running task plus lease.
- Stop acquiring when `len(state["leases"]) == 4`.
- Never create a lease for a GPU already leased by index or UUID.
- If launcher startup raises an expected launch exception, mark the task `launch_failed`, log its end, and continue to another pending task/candidate without pausing.
- Avoid creating the task directory twice if a launch attempt fails; launch failures are terminal and not retried.

- [ ] Write failing cooldown and reuse tests.

Use a mutable fake epoch clock and sequenced snapshots to prove:

- success or nonzero task exit moves its lease to cooldown at `clock()+60`;
- at 59 seconds no task launches;
- at 60 seconds an idle, UUID-stable card passes two final checks and receives the next task;
- cooldown reuse does not require five new observations;
- ready cooldown leases are processed in physical GPU index order;
- foreign compute PID, memory above 512 MiB, utilization above 5%, missing GPU, or UUID change releases the lease;
- a released GPU does not reuse its old idle streak and must requalify;
- launch failure during reuse releases the lease;
- process identity loss marks the task interrupted and immediately releases its lease;
- foreign PID during a running task does not stop it, but causes release after its task ends and cooldown expires.

- [ ] Run cooldown tests and confirm RED.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_combined_gpu_queue.py -q -k "cooldown or reuse or foreign or interrupted or launch_failure"
```

- [ ] Implement cooldown, reuse, and release transitions.

Implementation order inside `run_cycle`:

1. Poll running jobs and persist task result plus lease transition.
2. Take one normal snapshot; on transient probe failure log and return.
3. On audit cycles, observe fresh candidates and write all eight audit lines.
4. Process all cooldown leases whose deadline has passed, in GPU index order.
5. For each ready lease, require no compute PID, memory at most 512 MiB, utilization at most 5%, and matching UUID, followed by the existing double final check.
6. Reuse a passing lease for the next phase-eligible pending task.
7. Release a failing lease with a specific reason and reset that GPU's idle qualification.
8. Use remaining eligible unleased candidates for first acquisition, up to four leases.

When no task is currently eligible because the active phase still has running work, keep the cooldown lease instead of releasing it. This allows direct reuse when the phase barrier opens.

- [ ] Run all scheduler tests and confirm GREEN.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_combined_gpu_queue.py -q
```

- [ ] Commit.

```bash
git add src/combined_gpu_queue.py tests/test_combined_gpu_queue.py
git commit -m "feat: schedule through a four-GPU lease pool"
```

### Task 3: Make failures non-blocking and GPU probe errors transient

**Files:**

- Modify: `src/gpu_queue_core.py`
- Modify: `src/combined_gpu_queue.py`
- Modify: `tests/test_gpu_queue_core.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Error distinction:**

```python
class ProbeError(RuntimeError):
    """Transient nvidia-smi execution or parsing failure."""

class GpuMappingError(ProbeError):
    """Fatal global GPU index/UUID identity inconsistency."""
```

- [ ] Write failing robustness tests.

Add tests that prove:

- one `ProbeError` logs `GPU_PROBE_ERROR`, leaves state unpaused, makes no dispatch/reuse/release transition, and a later cycle recovers;
- repeated ordinary probe failures remain retryable;
- an `IdlePolicy` UUID mapping change raises `GpuMappingError`;
- a fatal mapping error makes `run_forever` exit nonzero without launching;
- one task's nonzero exit does not block pending tasks on another lease;
- after all Phase A tasks reach mixed terminal states, Phase B dispatches;
- launch failure and interruption do not block unrelated running jobs or pending work;
- all terminal tasks make `run_forever` return success even when some failed;
- completion releases remaining cooldown leases and writes one partial-success summary.

- [ ] Run robustness tests and confirm RED.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py -q \
  -k "probe or mapping or failed or interrupted or phase_b or completed"
```

- [ ] Implement failure isolation and fatal mapping handling.

Implementation details:

- Ordinary `NvidiaSmiProbe` command and row parsing failures remain `ProbeError`.
- `IdlePolicy` raises `GpuMappingError` only when the established global index-to-UUID map changes.
- Dispatcher catches ordinary `ProbeError`, logs a sanitized one-line `GPU_PROBE_ERROR detail=...`, and skips the cycle without persisting a pause.
- Let `GpuMappingError`, atomic state errors, resume errors, and lock errors fail closed to the CLI's nonzero error path.
- Remove task-failure and process-identity pause logging from normal result polling.
- Completion means all tasks are in allowed terminal states, not all succeeded.
- On completion release all remaining leases with reason `queue_complete`, persist, log one summary, and return zero.

- [ ] Run focused tests and confirm GREEN.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py -q
```

- [ ] Commit.

```bash
git add src/gpu_queue_core.py src/combined_gpu_queue.py \
  tests/test_gpu_queue_core.py tests/test_combined_gpu_queue.py
git commit -m "fix: isolate task and probe failures in GPU queue"
```

### Task 4: Expose lease state and partial-success results in audit logs

**Files:**

- Modify: `src/gpu_queue_audit.py`
- Modify: `src/combined_gpu_queue.py`
- Modify: `tests/test_gpu_queue_audit.py`
- Modify: `tests/test_combined_gpu_queue.py`

**Formatting interfaces:**

```python
def format_gpu_audit(
    snapshot: GpuSnapshot,
    *,
    idle_now: bool,
    idle_streak: int,
    running_task: Optional[dict],
    lease: Optional[dict],
    active_lease_count: int,
    max_gpu_leases: int,
    owned_compute_pids: Iterable[int],
    owner_lookup: Callable[[int], str],
) -> str: ...

def format_lease_event(
    event: str,
    lease: dict,
    *,
    reason: Optional[str] = None,
) -> str: ...

def format_queue_complete(summary: dict[str, object]) -> str: ...
```

- [ ] Write failing audit-format tests.

Assert stable output for:

- `lease=NONE`, `lease=RUNNING`, and `lease=COOLDOWN`;
- every audit line includes `leases=x/4`;
- running lease includes task ID;
- cooldown lease includes previous task ID and formatted cooldown deadline;
- acquired, cooldown, reused, and released events;
- released event always includes a reason;
- task end output for `FAILED`, `LAUNCH_FAILED`, and `INTERRUPTED`;
- completion summary includes total and all four terminal counts.

- [ ] Run audit tests and confirm RED.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_audit.py tests/test_combined_gpu_queue.py -q \
  -k "audit or lease_event or completion or task_end"
```

- [ ] Implement and wire the log formatters.

Implementation details:

- Keep each event on one line and sanitize exception/reason text.
- Add lease fields without logging commands or environment values.
- Preserve `OURS`/`SHARED` ownership reporting for running jobs.
- Log transition events immediately after the corresponding state save.
- Format cooldown epoch in the local timezone used by dispatcher timestamps.
- Produce exactly one final `QUEUE_COMPLETE total=...` summary.

- [ ] Run audit and dispatcher tests and confirm GREEN.

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_audit.py tests/test_combined_gpu_queue.py -q
```

- [ ] Commit.

```bash
git add src/gpu_queue_audit.py src/combined_gpu_queue.py \
  tests/test_gpu_queue_audit.py tests/test_combined_gpu_queue.py
git commit -m "feat: log GPU lease lifecycle and queue summary"
```

### Task 5: Document operations and verify without starting training

**Files:**

- Modify: `docs/combined_gpu_queue_runbook.md`
- Modify: `tests/test_combined_gpu_queue.py`

- [ ] Add runbook coverage.

Document:

- “GPU 队列” as the short operator name;
- maximum four held leases, including cooldown;
- first-use five-minute qualification;
- 60-second reuse criteria and double final check;
- release/requalification rules;
- foreign PID behavior while running and after cooldown;
- terminal task statuses and no automatic retry;
- Phase A-to-B mixed-result transition;
- transient probe retry versus fatal integrity exit;
- commands to inspect state/log/service;
- explicit statement that implementation did not start the service.

- [ ] Add or update CLI dry-run assertions.

The dry run must still:

- parse exactly 20 Phase A/B commands;
- inspect all eight GPUs;
- create no runtime state;
- launch no worker;
- report the four-lease policy in its informational output.

- [ ] Run all queue-focused tests.

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py \
  tests/test_gpu_queue_audit.py \
  tests/test_gpu_queue_worker.py \
  tests/test_combined_gpu_queue.py -q
```

- [ ] Run the complete suite in the isolated worktree.

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
```

- [ ] Run syntax and shell checks.

```bash
/data0/qrchen/miniconda3/envs/clip4cir/bin/python -m py_compile \
  src/gpu_queue_core.py \
  src/gpu_queue_audit.py \
  src/gpu_queue_worker.py \
  src/combined_gpu_queue.py
bash -n run_combined_gpu_queue.sh
git diff --check
```

- [ ] Run only the read-only CLI preflight.

```bash
bash run_combined_gpu_queue.sh dry-run
```

Expected: preflight and eight-GPU status output only. It must state that no state or training process was created.

- [ ] Prove that nothing was started.

Run:

```bash
systemctl --user show clip4cir-combined-gpu-queue.service \
  -p ActiveState -p SubState -p MainPID --no-pager
pgrep -af "combined_gpu_queue.py|gpu_queue_worker.py" || true
```

Expected: no active service process and no queue/worker process other than the inspection command itself.

- [ ] Commit documentation.

```bash
git add docs/combined_gpu_queue_runbook.md tests/test_combined_gpu_queue.py
git commit -m "docs: update resilient GPU queue operations"
```

### Task 6: Review, merge locally, and re-verify the delivered tree

**Files:**

- Review only: all files changed by Tasks 1–5
- Preserve untouched: `命令.sh`, `IDRiD平均召回率汇总.xlsx`, `Related_Work_组合示例查询与医学跨模态检索.md`

- [ ] Review against the design and inspect the diff.

```bash
git status --short
git diff --stat main...HEAD
git diff --check main...HEAD
git log --oneline main..HEAD
```

Verify every acceptance criterion in
`docs/superpowers/specs/2026-07-30-four-gpu-lease-pool-design.md`.

- [ ] Use the `superpowers:requesting-code-review` skill and address valid findings.

- [ ] Re-run queue-focused tests after review changes.

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest \
  tests/test_gpu_queue_core.py \
  tests/test_gpu_queue_audit.py \
  tests/test_gpu_queue_worker.py \
  tests/test_combined_gpu_queue.py -q
```

- [ ] Use `superpowers:verification-before-completion`.

Run fresh tests and checks immediately before claiming success.

- [ ] Use `superpowers:finishing-a-development-branch` and merge locally into `main`.

Before merging, confirm the main worktree's user-owned changes are unchanged. Merge the feature branch without touching those files. Resolve no conflict by discarding user changes.

- [ ] Verify the merged queue files on `main`.

Run the queue-focused suite on `main`. If a pre-existing user edit to `命令.sh` intentionally changes command text and causes only command-fixture assertions to differ, report that separately rather than modifying the file.

- [ ] Confirm again that the queue remains stopped.

Do not call `systemctl start`, `systemctl restart`, `bash run_combined_gpu_queue.sh run`, or `resume`. Report the implementation and verification results, current service state, and exact list of failed/interrupted tasks only if a real queue is later started by the user.
