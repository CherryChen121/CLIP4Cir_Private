from pathlib import Path

import pytest

from gpu_queue_core import (
    AtomicStateStore,
    PreflightError,
    GpuSnapshot,
    IdlePolicy,
    MAX_GPU_LEASES,
    ProbeError,
    QueueSpec,
    ResumeError,
    TaskSpec,
    acquire_lease,
    all_tasks_terminal,
    initial_state,
    mark_running,
    mark_lease_cooldown,
    next_pending_task,
    parse_combined_queue,
    preflight_queue,
    record_exit,
    record_interrupted,
    record_launch_failed,
    release_lease,
    terminal_summary,
    validate_lease_state,
    validate_resume_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "命令.sh"
PYTHON = Path("/data0/qrchen/miniconda3/envs/clip4cir/bin/python")


def _line(log_name="job.log", extra=""):
    return (
        "CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 nohup python "
        "src/combiner_train.py --dataset FashionIQ "
        "--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset "
        f"{extra}> {log_name} 2>&1 &"
    )


def _command_file(tmp_path, phase_a=None, phase_b=None):
    phase_a = phase_a or [_line(f"a{i}.log") for i in range(10)]
    phase_b = phase_b or [_line(f"b{i}.log") for i in range(10)]
    path = tmp_path / "commands.sh"
    path.write_text(
        "\n".join(
            [
                "# COMBINED_COMMANDS_BEGIN",
                "# Phase A:",
                *phase_a,
                "# Phase B:",
                *phase_b,
                "# Phase C:",
                _line("c.log"),
                "# COMBINED_COMMANDS_END",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _task(argv):
    return TaskSpec(
        task_id="A01",
        phase="A",
        ordinal=1,
        argv=tuple(argv),
        env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
        log_name="job.log",
        source="fixture",
    )


def _queue(task):
    return QueueSpec(tasks=(task,), command_sha256="fixture")


def test_real_combined_queue_contains_only_ordered_phase_a_and_b():
    queue = parse_combined_queue(COMMAND_FILE, PROJECT_ROOT, PYTHON)

    assert len(queue.tasks) == 20
    assert [task.task_id for task in queue.tasks[:2]] == ["A01", "A02"]
    assert [task.phase for task in queue.tasks] == ["A"] * 10 + ["B"] * 10
    assert len({task.log_name for task in queue.tasks}) == 20
    assert all(task.argv[0] == str(PYTHON) for task in queue.tasks)
    assert all("Phase C" not in task.source for task in queue.tasks)


def test_parser_rejects_shell_operators_outside_fixed_redirection(tmp_path):
    unsafe = _line("unsafe.log", extra="; touch /tmp/unsafe ")
    path = _command_file(tmp_path, phase_a=[unsafe] + [_line(f"a{i}.log") for i in range(1, 10)])

    with pytest.raises(PreflightError, match="unsupported shell token"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_parser_rejects_wrong_phase_count(tmp_path):
    path = _command_file(tmp_path, phase_a=[_line(f"a{i}.log") for i in range(9)])

    with pytest.raises(PreflightError, match="Phase A.*10"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_parser_rejects_duplicate_log_names(tmp_path):
    phase_a = [_line("duplicate.log")] + [_line(f"a{i}.log") for i in range(1, 10)]
    phase_b = [_line("duplicate.log")] + [_line(f"b{i}.log") for i in range(1, 10)]
    path = _command_file(tmp_path, phase_a=phase_a, phase_b=phase_b)

    with pytest.raises(PreflightError, match="duplicate log"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_preflight_rejects_missing_absolute_path():
    task = _task(
        (
            str(PYTHON),
            "src/combiner_train.py",
            "--dataset",
            "FashionIQ",
            "--fashioniq-root",
            "/definitely/missing/dataset",
        )
    )

    with pytest.raises(PreflightError, match="/definitely/missing/dataset"):
        preflight_queue(_queue(task), PROJECT_ROOT)


def test_preflight_rejects_unknown_cli_option():
    task = _task(
        (
            str(PYTHON),
            "src/combiner_train.py",
            "--dataset",
            "FashionIQ",
            "--definitely-unknown",
            "1",
        )
    )

    with pytest.raises(PreflightError, match="--definitely-unknown"):
        preflight_queue(_queue(task), PROJECT_ROOT)


def _snapshots(gpu0=(0, 0, ())):
    memory, utilization, pids = gpu0
    return tuple(
        GpuSnapshot(
            index=index,
            uuid=f"GPU-{index}",
            memory_used_mib=memory if index == 0 else 0,
            utilization_percent=utilization if index == 0 else 0,
            compute_pids=tuple(pids) if index == 0 else (),
        )
        for index in range(8)
    )


@pytest.mark.parametrize(
    "gpu0",
    [
        (513, 0, ()),
        (0, 6, ()),
        (0, 0, (42,)),
    ],
)
def test_idle_policy_fails_closed(gpu0):
    policy = IdlePolicy(expected_indices=range(8))

    assert 0 not in policy.observe(_snapshots(gpu0))


def test_gpu_becomes_candidate_only_on_fifth_consecutive_sample():
    policy = IdlePolicy(expected_indices=range(8))

    for _ in range(4):
        assert 0 not in policy.observe(_snapshots())
    assert 0 in policy.observe(_snapshots())


def test_idle_policy_exposes_streak_by_gpu_uuid():
    policy = IdlePolicy(expected_indices=range(8))
    snapshots = _snapshots()

    policy.observe(snapshots)
    policy.observe(snapshots)

    assert policy.idle_streak("GPU-0") == 2
    assert policy.idle_streak("GPU-never-seen") == 0


def test_busy_sample_resets_consecutive_idle_count():
    policy = IdlePolicy(expected_indices=range(8))
    for _ in range(4):
        policy.observe(_snapshots())

    policy.observe(_snapshots((0, 0, (99,))))

    for _ in range(4):
        assert 0 not in policy.observe(_snapshots())
    assert 0 in policy.observe(_snapshots())


def test_released_gpu_must_accumulate_five_new_idle_samples():
    policy = IdlePolicy(expected_indices=range(8))
    for _ in range(5):
        policy.observe(_snapshots())
    assert policy.idle_streak("GPU-0") == 5

    policy.reset("GPU-0")

    for _ in range(4):
        assert 0 not in policy.observe(_snapshots())
    assert 0 in policy.observe(_snapshots())


def test_idle_policy_rejects_gpu_uuid_mapping_change():
    policy = IdlePolicy(expected_indices=range(8))
    policy.observe(_snapshots())
    changed = list(_snapshots())
    changed[0] = GpuSnapshot(0, "GPU-changed", 0, 0, ())

    with pytest.raises(ProbeError, match="UUID mapping"):
        policy.observe(tuple(changed))


def _small_queue(digest="queue-digest"):
    tasks = []
    for phase in ("A", "B"):
        for ordinal in (1, 2):
            tasks.append(
                TaskSpec(
                    task_id=f"{phase}{ordinal:02d}",
                    phase=phase,
                    ordinal=ordinal,
                    argv=(str(PYTHON), "src/combiner_train.py", "--dataset", "FashionIQ"),
                    env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
                    log_name=f"{phase.lower()}{ordinal}.log",
                    source=f"{phase}{ordinal}",
                )
            )
    return QueueSpec(tuple(tasks), digest)


def _launch():
    return {
        "pid": 101,
        "pgid": 101,
        "start_ticks": 12345,
        "command_sha256": "worker-command",
        "manifest_path": "/tmp/A01/manifest.json",
        "result_path": "/tmp/A01/result.json",
        "log_path": "/tmp/A01/task.log",
    }


def _task_state(state, task_id):
    return next(task for task in state["tasks"] if task["task_id"] == task_id)


def test_initial_state_has_empty_lease_pool_and_task_error_fields():
    state = initial_state(_small_queue(), "run-1", "created")

    assert state["schema_version"] == 2
    assert state["leases"] == []
    assert all(task["error"] is None for task in state["tasks"])
    assert MAX_GPU_LEASES == 4


def test_phase_b_opens_after_every_phase_a_task_is_terminal():
    state = initial_state(_small_queue(), "run-1", "2026-07-30T00:00:00")
    gpu = GpuSnapshot(0, "GPU-0", 0, 0, ())

    assert next_pending_task(state)["task_id"] == "A01"
    mark_running(state, "A01", _launch(), gpu, "start")
    record_exit(state, "A01", 7, "end")
    assert next_pending_task(state)["task_id"] == "A02"
    record_launch_failed(state, "A02", "worker disappeared", "end", gpu)

    assert next_pending_task(state)["task_id"] == "B01"


def test_phase_b_remains_blocked_while_a_phase_a_task_is_running():
    state = initial_state(_small_queue(), "run-1", "created")
    gpu = GpuSnapshot(0, "GPU-0", 0, 0, ())
    mark_running(state, "A01", _launch(), gpu, "start")
    record_launch_failed(state, "A02", "launch failed", "end", gpu)

    assert next_pending_task(state) is None


def test_nonzero_exit_does_not_pause_new_dispatch_or_stop_other_task():
    state = initial_state(_small_queue(), "run-1", "created")
    gpu0 = GpuSnapshot(0, "GPU-0", 0, 0, ())
    gpu1 = GpuSnapshot(1, "GPU-1", 0, 0, ())
    mark_running(state, "A01", _launch(), gpu0, "start")
    second_launch = dict(_launch(), pid=202, pgid=202)
    mark_running(state, "A02", second_launch, gpu1, "start")

    record_exit(state, "A01", 7, "end")

    assert state["paused_reason"] is None
    assert _task_state(state, "A02")["status"] == "running"
    assert next_pending_task(state) is None


def test_launch_failure_and_interruption_are_terminal_without_pausing():
    state = initial_state(_small_queue(), "run-1", "created")
    gpu = GpuSnapshot(2, "GPU-2", 0, 0, ())

    record_launch_failed(state, "A01", "bad\nlaunch", "end-1", gpu)
    mark_running(state, "A02", _launch(), gpu, "start")
    record_interrupted(state, "A02", "end-2")

    failed = _task_state(state, "A01")
    interrupted = _task_state(state, "A02")
    assert failed["status"] == "launch_failed"
    assert failed["error"] == "bad launch"
    assert failed["gpu_index"] == 2
    assert failed["gpu_uuid"] == "GPU-2"
    assert interrupted["status"] == "interrupted"
    assert state["paused_reason"] is None


def test_lease_lifecycle_enforces_four_slot_limit():
    state = initial_state(_small_queue(), "run-1", "created")
    for index, task_id in enumerate(("A01", "A02", "B01", "B02")):
        acquire_lease(
            state,
            GpuSnapshot(index, f"GPU-{index}", 0, 0, ()),
            task_id,
            "acquired",
        )

    with pytest.raises(ResumeError, match="at most 4"):
        acquire_lease(
            state,
            GpuSnapshot(4, "GPU-4", 0, 0, ()),
            "extra",
            "acquired",
        )

    lease = mark_lease_cooldown(state, "A01", 60.0, "updated")
    assert lease["state"] == "cooldown"
    assert lease["previous_task_id"] == "A01"
    assert lease["task_id"] is None
    assert lease["cooldown_ready_at"] == 60.0
    assert release_lease(state, 0)["gpu_uuid"] == "GPU-0"
    assert len(state["leases"]) == 3


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda leases: leases.append(dict(leases[0], task_id="B01")),
            "duplicate GPU index",
        ),
        (
            lambda leases: leases.append(
                dict(leases[0], gpu_index=1, task_id="B01")
            ),
            "duplicate GPU UUID",
        ),
        (
            lambda leases: leases.append(
                dict(leases[0], gpu_index=1, gpu_uuid="GPU-1")
            ),
            "duplicate lease task",
        ),
        (
            lambda leases: leases.__setitem__(
                0, dict(leases[0], cooldown_ready_at=float("inf"))
            ),
            "cooldown_ready_at",
        ),
    ],
)
def test_lease_validation_rejects_duplicate_or_invalid_state(mutate, match):
    state = initial_state(_small_queue(), "run-1", "created")
    acquire_lease(state, GpuSnapshot(0, "GPU-0", 0, 0, ()), "A01", "acquired")
    mutate(state["leases"])

    with pytest.raises(ResumeError, match=match):
        validate_lease_state(state)


def test_resume_requires_running_tasks_and_leases_to_match():
    state = initial_state(_small_queue(), "run-1", "created")
    gpu = GpuSnapshot(0, "GPU-0", 0, 0, ())
    mark_running(state, "A01", _launch(), gpu, "start")

    with pytest.raises(ResumeError, match="running task.*lease"):
        validate_resume_state(state, _small_queue())


def test_terminal_summary_reports_partial_success_task_ids():
    state = initial_state(_small_queue(), "run-1", "created")
    statuses = ("succeeded", "failed", "launch_failed", "interrupted")
    for task, status in zip(state["tasks"], statuses):
        task["status"] = status

    assert all_tasks_terminal(state)
    assert terminal_summary(state) == {
        "total": 4,
        "succeeded": 1,
        "failed": 1,
        "launch_failed": 1,
        "interrupted": 1,
        "task_ids": {
            "succeeded": ["A01"],
            "failed": ["A02"],
            "launch_failed": ["B01"],
            "interrupted": ["B02"],
        },
    }


def test_atomic_state_store_replaces_complete_json(tmp_path):
    path = tmp_path / "state.json"
    store = AtomicStateStore(path)

    store.save({"schema_version": 1, "tasks": []})

    assert store.load() == {"schema_version": 1, "tasks": []}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_resume_rejects_changed_command_digest():
    state = initial_state(_small_queue("old"), "run-1", "created")

    with pytest.raises(ResumeError, match="command digest"):
        validate_resume_state(state, _small_queue("new"))
