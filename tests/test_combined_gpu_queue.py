import json
import os
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from combined_gpu_queue import (
    AlreadyRunningError,
    Dispatcher,
    DispatcherLock,
    LaunchRecord,
    NvidiaSmiProbe,
    OwnedProcessLauncher,
    ProcessIdentity,
    ProcessIdentityError,
    build_parser,
    main,
)
from gpu_queue_core import (
    GpuSnapshot,
    IdlePolicy,
    ProbeError,
    QueueSpec,
    TaskSpec,
    acquire_lease,
    initial_state,
    mark_running,
)
from gpu_queue_worker import WorkerError, run_manifest


def _gpu_rows():
    return [f"{index}, GPU-{index}, 0, 0" for index in range(8)]


class FakeRunner:
    def __init__(self, gpu_rows=None, process_rows=None, returncode=0):
        self.gpu_rows = _gpu_rows() if gpu_rows is None else gpu_rows
        self.process_rows = [] if process_rows is None else process_rows
        self.returncode = returncode
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        rows = self.gpu_rows if "--query-gpu=" in argv[1] else self.process_rows
        return CompletedProcess(argv, self.returncode, "\n".join(rows), "probe failed")


def test_probe_keeps_zero_utilization_gpu_busy_when_compute_pid_exists():
    runner = FakeRunner(
        process_rows=["GPU-0, 1790625, ray::WorkerDict, 14898"]
    )

    gpu = NvidiaSmiProbe(runner=runner).snapshot()[0]

    assert gpu.utilization_percent == 0
    assert gpu.compute_pids == (1790625,)
    assert all(call[1]["shell"] is False for call in runner.calls)


def test_probe_accepts_blank_compute_process_output():
    snapshots = NvidiaSmiProbe(runner=FakeRunner()).snapshot()

    assert len(snapshots) == 8
    assert all(gpu.compute_pids == () for gpu in snapshots)


def test_probe_rejects_unknown_process_uuid():
    runner = FakeRunner(process_rows=["GPU-unknown, 12, python, 100"])

    with pytest.raises(ProbeError, match="unknown GPU UUID"):
        NvidiaSmiProbe(runner=runner).snapshot()


def test_probe_rejects_nonzero_nvidia_smi_exit():
    with pytest.raises(ProbeError, match="probe failed"):
        NvidiaSmiProbe(runner=FakeRunner(returncode=1)).snapshot()


def test_probe_rejects_missing_gpu_index():
    with pytest.raises(ProbeError, match="indices 0..7"):
        NvidiaSmiProbe(runner=FakeRunner(gpu_rows=_gpu_rows()[:-1])).snapshot()


def test_worker_records_child_exit_code_without_shell(tmp_path):
    log_path = tmp_path / "task.log"
    result_path = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                "cwd": str(tmp_path),
                "env_overrides": {
                    "CUDA_VISIBLE_DEVICES": "6",
                    "NCCL_P2P_DISABLE": "1",
                },
                "log_path": str(log_path),
                "result_path": str(result_path),
            }
        ),
        encoding="utf-8",
    )

    assert run_manifest(manifest) == 7
    assert json.loads(result_path.read_text()) == {"return_code": 7}
    assert log_path.exists()


def test_worker_rejects_existing_log_without_starting_child(tmp_path):
    marker = tmp_path / "child-started"
    log_path = tmp_path / "task.log"
    log_path.write_text("existing", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                "cwd": str(tmp_path),
                "env_overrides": {"CUDA_VISIBLE_DEVICES": "0"},
                "log_path": str(log_path),
                "result_path": str(tmp_path / "result.json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="already exists"):
        run_manifest(manifest)
    assert not marker.exists()


class FakePopen:
    def __init__(self):
        self.calls = []
        self.pid = 4321

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return type("Process", (), {"pid": self.pid})()


class FakeInspector:
    def __init__(self, identity):
        self.current = identity

    def identity(self, pid):
        return self.current if self.current and self.current.pid == pid else None


def _task():
    return TaskSpec(
        task_id="A01",
        phase="A",
        ordinal=1,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
        log_name="a01.log",
        source="fixture",
    )


def test_launcher_uses_worker_new_session_and_persists_only_safe_env(tmp_path):
    identity = ProcessIdentity(4321, 4321, 99, "worker-digest")
    popen = FakePopen()
    launcher = OwnedProcessLauncher(
        project_root=tmp_path,
        python_executable=Path(sys.executable),
        worker_script=tmp_path / "gpu_queue_worker.py",
        proc_inspector=FakeInspector(identity),
        popen_factory=popen,
    )

    launch = launcher.start(
        _task(),
        GpuSnapshot(6, "GPU-6", 0, 0, ()),
        tmp_path / "A01",
    )

    argv, kwargs = popen.calls[-1]
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert argv[0] == sys.executable
    manifest = json.loads(launch.manifest_path.read_text())
    assert manifest["env_overrides"] == {
        "CUDA_VISIBLE_DEVICES": "6",
        "NCCL_P2P_DISABLE": "1",
    }
    assert "PATH" not in manifest["env_overrides"]


def test_terminate_refuses_signal_when_process_identity_changed(tmp_path):
    original = ProcessIdentity(4321, 4321, 99, "original")
    changed = ProcessIdentity(4321, 4321, 100, "different")
    signals = []
    launcher = OwnedProcessLauncher(
        project_root=tmp_path,
        python_executable=Path(sys.executable),
        worker_script=tmp_path / "gpu_queue_worker.py",
        proc_inspector=FakeInspector(changed),
        killpg=lambda pgid, signal_number: signals.append((pgid, signal_number)),
    )

    assert launcher.terminate(original, grace_seconds=0) is False
    assert signals == []


def _queue(phase_size=2):
    tasks = []
    for phase in ("A", "B"):
        for ordinal in range(1, phase_size + 1):
            tasks.append(
                TaskSpec(
                    task_id=f"{phase}{ordinal:02d}",
                    phase=phase,
                    ordinal=ordinal,
                    argv=(sys.executable, "-c", "raise SystemExit(0)"),
                    env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
                    log_name=f"{phase.lower()}{ordinal}.log",
                    source=f"{phase}{ordinal}",
                )
            )
    return QueueSpec(tuple(tasks), "digest")


def _eight_snapshots(overrides=None):
    overrides = overrides or {}
    return tuple(
        overrides.get(index, GpuSnapshot(index, f"GPU-{index}", 0, 0, ()))
        for index in range(8)
    )


class StaticProbe:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return self.snapshots


class EligiblePolicy:
    def __init__(self, eligible):
        self.eligible = tuple(eligible)

    def observe(self, snapshots):
        return self.eligible

    def is_idle_now(self, snapshot):
        return not snapshot.compute_pids and snapshot.memory_used_mib <= 512 and snapshot.utilization_percent <= 5

    def idle_streak(self, gpu_uuid):
        return 5 if int(gpu_uuid.rsplit("-", 1)[1]) in self.eligible else 0


class MemoryStore:
    def __init__(self):
        self.saved = []

    def save(self, state):
        self.saved.append(json.loads(json.dumps(state)))


class DispatcherLauncher:
    def __init__(self, exit_codes=None):
        self.assignments = []
        self.terminated = []
        self.exit_codes = exit_codes or {}

    def start(self, task, gpu, task_dir):
        self.assignments.append((task.task_id, gpu.index))
        pid = 1000 + len(self.assignments)
        return LaunchRecord(
            ProcessIdentity(pid, pid, pid * 10, f"digest-{pid}"),
            task_dir / "manifest.json",
            task_dir / "result.json",
            task_dir / task.log_name,
        )

    def poll(self, identity, result_path):
        return self.exit_codes.get(identity.pid)

    def terminate(self, identity, grace_seconds=30):
        self.terminated.append(identity)
        return True


class Descendants:
    def __init__(self, values=None):
        self.values = values or {}

    def descendants(self, pid):
        return set(self.values.get(pid, set()))

    def identity(self, pid):
        return None


class MutableClock:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self):
        return self.value


def _seed_running(state, task_id, gpu_index, pid, tmp_path):
    gpu = GpuSnapshot(gpu_index, f"GPU-{gpu_index}", 100, 10, (pid,))
    mark_running(
        state,
        task_id,
        {
            "pid": pid,
            "pgid": pid,
            "start_ticks": pid,
            "command_sha256": f"digest-{pid}",
            "manifest_path": str(tmp_path / task_id / "manifest.json"),
            "result_path": str(tmp_path / task_id / "result.json"),
            "log_path": str(tmp_path / task_id / "task.log"),
        },
        gpu,
        "start",
    )
    acquire_lease(state, gpu, task_id, "acquired")


def test_multiple_idle_gpus_receive_tasks_in_gpu_and_queue_order(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    launcher = DispatcherLauncher()
    sleeps = []
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([2, 5]),
        launcher=launcher,
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=sleeps.append,
        now=lambda: "2026-07-30T10:00:00+0800",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=True)

    assert launcher.assignments == [("A01", 2), ("A02", 5)]
    assert sleeps == [3, 3]
    assert any(event.startswith("TASK_START task=A01 phase=A gpu=2") for event in events)
    assert any(event.startswith("TASK_START task=A02 phase=A gpu=5") for event in events)


def test_eight_eligible_gpus_create_only_four_running_leases(tmp_path):
    queue = _queue(phase_size=6)
    state = initial_state(queue, "run-1", "created")
    launcher = DispatcherLauncher()
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy(range(8)),
        launcher=launcher,
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "now",
    )

    dispatcher.run_cycle(sample_idle=True)

    assert launcher.assignments == [
        ("A01", 0),
        ("A02", 1),
        ("A03", 2),
        ("A04", 3),
    ]
    assert [(lease["gpu_index"], lease["state"]) for lease in state["leases"]] == [
        (0, "running"),
        (1, "running"),
        (2, "running"),
        (3, "running"),
    ]


def test_running_and_cooldown_leases_together_block_a_fifth_gpu(tmp_path):
    queue = _queue(phase_size=6)
    state = initial_state(queue, "run-1", "created")
    for task_id, gpu_index, pid in (
        ("A01", 0, 101),
        ("A02", 1, 102),
        ("A03", 2, 103),
        ("A04", 3, 104),
    ):
        _seed_running(state, task_id, gpu_index, pid, tmp_path)
    state["tasks"][0]["status"] = "succeeded"
    state["leases"][0].update(
        state="cooldown",
        task_id=None,
        previous_task_id="A01",
        cooldown_ready_at=999.0,
    )
    launcher = DispatcherLauncher()
    dispatcher = Dispatcher(
        queue,
        state,
        tmp_path,
        StaticProbe(_eight_snapshots()),
        EligiblePolicy([4]),
        launcher,
        Descendants(),
        MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "now",
        clock=lambda: 100.0,
    )

    dispatcher.run_cycle(sample_idle=True)

    assert launcher.assignments == []
    assert len(state["leases"]) == 4


def test_successful_task_waits_sixty_seconds_then_reuses_same_gpu(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    _seed_running(state, "A01", 2, 101, tmp_path)
    clock = MutableClock(100)
    launcher = DispatcherLauncher(exit_codes={101: 0})
    dispatcher = Dispatcher(
        queue,
        state,
        tmp_path,
        StaticProbe(_eight_snapshots()),
        EligiblePolicy([]),
        launcher,
        Descendants(),
        MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "now",
        clock=clock,
    )

    dispatcher.run_cycle(sample_idle=False)
    assert state["leases"][0]["state"] == "cooldown"
    assert state["leases"][0]["cooldown_ready_at"] == 160.0
    assert launcher.assignments == []

    launcher.exit_codes = {}
    clock.value = 159
    dispatcher.run_cycle(sample_idle=False)
    assert launcher.assignments == []

    clock.value = 160
    dispatcher.run_cycle(sample_idle=False)
    assert launcher.assignments == [("A02", 2)]
    assert state["leases"][0]["state"] == "running"
    assert state["leases"][0]["task_id"] == "A02"


@pytest.mark.parametrize(
    "replacement",
    [
        GpuSnapshot(2, "GPU-2", 0, 0, (7777,)),
        GpuSnapshot(2, "GPU-2", 513, 0, ()),
        GpuSnapshot(2, "GPU-2", 0, 6, ()),
        GpuSnapshot(2, "GPU-changed", 0, 0, ()),
    ],
)
def test_ready_cooldown_releases_gpu_that_is_not_safe_to_reuse(
    tmp_path, replacement
):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    _seed_running(state, "A01", 2, 101, tmp_path)
    state["tasks"][0]["status"] = "succeeded"
    state["leases"][0].update(
        state="cooldown",
        task_id=None,
        previous_task_id="A01",
        cooldown_ready_at=100.0,
    )
    launcher = DispatcherLauncher()
    dispatcher = Dispatcher(
        queue,
        state,
        tmp_path,
        StaticProbe(_eight_snapshots({2: replacement})),
        EligiblePolicy([]),
        launcher,
        Descendants(),
        MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "now",
        clock=lambda: 100.0,
    )

    dispatcher.run_cycle(sample_idle=False)

    assert state["leases"] == []
    assert launcher.assignments == []


def test_second_final_probe_becoming_busy_cancels_launch(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    idle = _eight_snapshots()
    busy = _eight_snapshots({2: GpuSnapshot(2, "GPU-2", 100, 0, (7777,))})

    class SequencedProbe:
        def __init__(self):
            self.responses = iter((idle, idle, busy))

        def snapshot(self):
            return next(self.responses)

    launcher = DispatcherLauncher()
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=SequencedProbe(),
        idle_policy=EligiblePolicy([2]),
        launcher=launcher,
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "now",
    )

    dispatcher.run_cycle(sample_idle=True)

    assert launcher.assignments == []


def test_launch_failure_is_terminal_and_does_not_pause_queue(tmp_path):
    class FailingLauncher(DispatcherLauncher):
        def start(self, task, gpu, task_dir):
            raise ProcessIdentityError("worker disappeared")

    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([2]),
        launcher=FailingLauncher(),
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "launch-time",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=True)

    task = next(task for task in state["tasks"] if task["task_id"] == "A01")
    assert state["paused_reason"] is None
    assert task["status"] == "launch_failed"
    assert task["error"] == "worker disappeared"
    assert task["gpu_index"] == 2
    assert state["leases"] == []
    assert any("TASK_END task=A01" in event and "LAUNCH_FAILED" in event for event in events)


def test_foreign_pid_after_launch_is_logged_without_stopping_owned_task(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    gpu = GpuSnapshot(3, "GPU-3", 2000, 80, (202, 7777))
    launch = {
        "pid": 101,
        "pgid": 101,
        "start_ticks": 10,
        "command_sha256": "owned",
        "manifest_path": str(tmp_path / "manifest.json"),
        "result_path": str(tmp_path / "result.json"),
        "log_path": str(tmp_path / "task.log"),
    }
    mark_running(state, "A01", launch, gpu, "start")
    acquire_lease(state, gpu, "A01", "acquired")
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
    assert next(task for task in state["tasks"] if task["task_id"] == "A01")["status"] == "running"
    assert any(
        "GPU_AUDIT gpu=3" in event
        and "status=SHARED" in event
        and "foreign_pids=7777" in event
        and "foreign_owners=xtchen" in event
        for event in events
    )


def test_minute_cycle_logs_all_eight_gpus_once_in_index_order(tmp_path):
    queue = _queue()
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=initial_state(queue, "run-1", "created"),
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([]),
        launcher=DispatcherLauncher(),
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:00:00+0800",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=True)

    audit_lines = [event for event in events if event.startswith("GPU_AUDIT gpu=")]
    assert len(audit_lines) == 8
    assert [line.split("gpu=", 1)[1].split(" ", 1)[0] for line in audit_lines] == [
        str(index) for index in range(8)
    ]
    assert events.count("GPU_AUDIT_BEGIN") == 1
    assert events.count("GPU_AUDIT_END") == 1


def test_half_minute_cycle_does_not_log_gpu_audit(tmp_path):
    queue = _queue()
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=initial_state(queue, "run-1", "created"),
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([]),
        launcher=DispatcherLauncher(),
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:00:30+0800",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=False)

    assert not any(event.startswith("GPU_AUDIT") for event in events)


def test_failed_task_enters_cooldown_while_other_owned_job_keeps_running(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    for task_id, pid, gpu_index in (("A01", 101, 0), ("A02", 102, 1)):
        _seed_running(state, task_id, gpu_index, pid, tmp_path)
    launcher = DispatcherLauncher(exit_codes={101: 1, 102: None})
    snapshots = _eight_snapshots(
        {
            0: GpuSnapshot(0, "GPU-0", 0, 0, ()),
            1: GpuSnapshot(1, "GPU-1", 100, 10, (102,)),
        }
    )
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(snapshots),
        idle_policy=EligiblePolicy([]),
        launcher=launcher,
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:02:30+0800",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=False)

    assert state["paused_reason"] is None
    assert next(task for task in state["tasks"] if task["task_id"] == "A02")["status"] == "running"
    assert next(
        lease for lease in state["leases"] if lease["gpu_index"] == 0
    )["state"] == "cooldown"
    assert launcher.terminated == []
    assert any("TASK_END task=A01" in event and "result=FAILED" in event for event in events)
    assert not any(event.startswith("QUEUE_PAUSE") for event in events)


def test_successful_task_exit_logs_task_end(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    _seed_running(state, "A01", 0, 101, tmp_path)
    next(task for task in state["tasks"] if task["task_id"] == "A01")[
        "started_at"
    ] = "2026-07-30T10:00:00+0800"
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([]),
        launcher=DispatcherLauncher(exit_codes={101: 0}),
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:02:30+0800",
        event_logger=events.append,
    )

    dispatcher.run_cycle(sample_idle=False)

    assert any("TASK_END task=A01" in event and "result=SUCCEEDED" in event for event in events)
    assert state["paused_reason"] is None


def test_resume_keeps_exact_live_worker_running(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    _seed_running(state, "A01", 0, 101, tmp_path)
    dispatcher = Dispatcher(
        queue, state, tmp_path, StaticProbe(_eight_snapshots()),
        EligiblePolicy([]), DispatcherLauncher(), Descendants(), MemoryStore(),
        sleeper=lambda seconds: None, now=lambda: "resume",
    )

    dispatcher.reconcile_resume()

    assert next(task for task in state["tasks"] if task["task_id"] == "A01")["status"] == "running"
    assert state["paused_reason"] is None


def test_resume_marks_missing_worker_interrupted_and_releases_lease(tmp_path):
    class MissingLauncher(DispatcherLauncher):
        def poll(self, identity, result_path):
            raise ProcessIdentityError("missing")

    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    _seed_running(state, "A01", 0, 101, tmp_path)
    events = []
    dispatcher = Dispatcher(
        queue, state, tmp_path, StaticProbe(_eight_snapshots()),
        EligiblePolicy([]), MissingLauncher(), Descendants(), MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:02:30+0800",
        event_logger=events.append,
    )

    dispatcher.reconcile_resume()

    assert next(task for task in state["tasks"] if task["task_id"] == "A01")["status"] == "interrupted"
    assert state["paused_reason"] is None
    assert state["leases"] == []
    assert any("TASK_END task=A01" in event and "result=INTERRUPTED" in event for event in events)
    assert not any(event.startswith("QUEUE_PAUSE") for event in events)


def test_completed_queue_logs_completion_once(tmp_path):
    queue = _queue()
    state = initial_state(queue, "run-1", "created")
    for task in state["tasks"]:
        task["status"] = "succeeded"
    events = []
    dispatcher = Dispatcher(
        queue=queue,
        state=state,
        run_dir=tmp_path,
        probe=StaticProbe(_eight_snapshots()),
        idle_policy=EligiblePolicy([]),
        launcher=DispatcherLauncher(),
        proc_inspector=Descendants(),
        state_store=MemoryStore(),
        sleeper=lambda seconds: None,
        now=lambda: "2026-07-30T10:02:30+0800",
        event_logger=events.append,
    )

    assert dispatcher.run_forever() == 0

    assert events.count("QUEUE_COMPLETE") == 1


def test_cli_requires_exactly_one_mode():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run", "--run"])


def test_second_dispatcher_lock_is_rejected(tmp_path):
    first = DispatcherLock(tmp_path / "dispatcher.lock")
    second = DispatcherLock(tmp_path / "dispatcher.lock")
    first.acquire()
    try:
        with pytest.raises(AlreadyRunningError):
            second.acquire()
    finally:
        first.release()


def test_dry_run_reports_all_occupied_gpus_without_creating_state(tmp_path):
    occupied = tuple(
        GpuSnapshot(index, f"GPU-{index}", 16000, 0, (9000 + index,))
        for index in range(8)
    )
    output = []
    dependencies = SimpleNamespace(
        project_root=Path(__file__).resolve().parents[1],
        command_file=Path(__file__).resolve().parents[1] / "命令.sh",
        python_executable=Path(sys.executable),
        runtime_root=tmp_path / "gpu_queue_runs",
        probe=StaticProbe(occupied),
        output=output.append,
    )

    assert main(["--dry-run"], dependencies=dependencies) == 0

    assert not dependencies.runtime_root.exists()
    assert len([line for line in output if "compute_pids=" in line]) == 8
    assert all("unavailable" in line for line in output if "compute_pids=" in line)


def test_shell_wrapper_rejects_unknown_mode():
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["bash", "run_combined_gpu_queue.sh", "unknown"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Usage:" in completed.stderr


def test_script_entrypoint_defines_probe_before_running_dry_run(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
  for index in 0 1 2 3 4 5 6 7; do
    echo "${index}, GPU-${index}, 0, 0"
  done
fi
""",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [sys.executable, "src/combined_gpu_queue.py", "--dry-run"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dry-run only" in completed.stdout
    assert "NameError" not in completed.stderr
