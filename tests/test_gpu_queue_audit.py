import pytest

from gpu_queue_audit import (
    format_gpu_audit,
    format_lease_event,
    format_queue_complete,
    format_queue_pause,
    format_task_end,
    format_task_start,
)
from gpu_queue_core import GpuSnapshot


def test_formats_idle_and_foreign_gpu_lines():
    idle = format_gpu_audit(
        GpuSnapshot(0, "GPU-0", 15, 0, ()),
        idle_now=True,
        idle_streak=3,
        running_task=None,
        lease=None,
        active_lease_count=0,
        max_gpu_leases=4,
        owned_compute_pids=(),
        owner_lookup=lambda pid: "unused",
    )
    foreign = format_gpu_audit(
        GpuSnapshot(2, "GPU-2", 38111, 100, (1810854,)),
        idle_now=False,
        idle_streak=0,
        running_task=None,
        lease=None,
        active_lease_count=0,
        max_gpu_leases=4,
        owned_compute_pids=(),
        owner_lookup=lambda pid: "jycheng",
    )

    assert idle == (
        "GPU_AUDIT gpu=0 uuid=GPU-0 leases=0/4 lease=NONE status=IDLE "
        "memory_mib=15 util_percent=0 idle_streak=3/5"
    )
    assert foreign == (
        "GPU_AUDIT gpu=2 uuid=GPU-2 leases=0/4 lease=NONE status=FOREIGN "
        "memory_mib=38111 util_percent=100 pids=1810854 owners=jycheng"
    )


def test_formats_ours_and_shared_gpu_lines():
    task = {"task_id": "A01", "pid": 101}
    ours = format_gpu_audit(
        GpuSnapshot(4, "GPU-4", 9000, 95, (202,)),
        idle_now=False,
        idle_streak=0,
        running_task=task,
        lease={
            "state": "running",
            "task_id": "A01",
            "previous_task_id": None,
            "cooldown_ready_at": None,
        },
        active_lease_count=1,
        max_gpu_leases=4,
        owned_compute_pids=(202,),
        owner_lookup=lambda pid: "qrchen",
    )
    shared = format_gpu_audit(
        GpuSnapshot(4, "GPU-4", 12000, 100, (202, 7777)),
        idle_now=False,
        idle_streak=0,
        running_task=task,
        lease={
            "state": "running",
            "task_id": "A01",
            "previous_task_id": None,
            "cooldown_ready_at": None,
        },
        active_lease_count=1,
        max_gpu_leases=4,
        owned_compute_pids=(202,),
        owner_lookup=lambda pid: "xtchen",
    )

    assert ours == (
        "GPU_AUDIT gpu=4 uuid=GPU-4 leases=1/4 lease=RUNNING status=OURS "
        "memory_mib=9000 util_percent=95 task=A01 our_pid=101 our_pids=202"
    )
    assert shared == (
        "GPU_AUDIT gpu=4 uuid=GPU-4 leases=1/4 lease=RUNNING status=SHARED "
        "memory_mib=12000 util_percent=100 task=A01 our_pid=101 our_pids=202 "
        "foreign_pids=7777 foreign_owners=xtchen"
    )


def test_owner_lookup_failure_is_reported_as_unknown():
    def unreadable_owner(pid):
        raise OSError(f"missing PID {pid}")

    line = format_gpu_audit(
        GpuSnapshot(3, "GPU-3", 1000, 10, (7001,)),
        idle_now=False,
        idle_streak=0,
        running_task=None,
        lease=None,
        active_lease_count=0,
        max_gpu_leases=4,
        owned_compute_pids=(),
        owner_lookup=unreadable_owner,
    )

    assert line.endswith("pids=7001 owners=unknown")


def test_formats_cooldown_lease_and_lifecycle_events():
    lease = {
        "gpu_index": 2,
        "gpu_uuid": "GPU-2",
        "state": "cooldown",
        "task_id": None,
        "previous_task_id": "A01",
        "cooldown_ready_at": 0.0,
    }
    audit = format_gpu_audit(
        GpuSnapshot(2, "GPU-2", 0, 0, ()),
        idle_now=True,
        idle_streak=0,
        running_task=None,
        lease=lease,
        active_lease_count=3,
        max_gpu_leases=4,
        owned_compute_pids=(),
        owner_lookup=lambda pid: "unused",
    )

    assert (
        "leases=3/4 lease=COOLDOWN previous_task=A01 "
        "cooldown_ready_at=1970-01-01T08:00:00+0800"
    ) in audit
    assert format_lease_event("GPU_LEASE_COOLDOWN", lease) == (
        "GPU_LEASE_COOLDOWN gpu=2 gpu_uuid=GPU-2 previous_task=A01 "
        "cooldown_ready_at=1970-01-01T08:00:00+0800"
    )
    assert format_lease_event(
        "GPU_LEASE_RELEASED", lease, reason="foreign_pid"
    ).endswith("reason=foreign_pid")
    with pytest.raises(ValueError, match="reason"):
        format_lease_event("GPU_LEASE_RELEASED", lease)


def test_formats_partial_success_completion_summary():
    summary = {
        "total": 20,
        "succeeded": 17,
        "failed": 2,
        "launch_failed": 1,
        "interrupted": 0,
    }

    assert format_queue_complete(summary) == (
        "QUEUE_COMPLETE total=20 succeeded=17 failed=2 "
        "launch_failed=1 interrupted=0"
    )


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

    assert format_task_start(task) == (
        "TASK_START task=A01 phase=A gpu=4 gpu_uuid=GPU-4 "
        "pid=101 log_path=/tmp/A01/train.log"
    )
    assert format_task_end(task) == (
        "TASK_END task=A01 phase=A gpu=4 result=SUCCEEDED return_code=0 "
        "started_at=2026-07-30T10:00:00+0800 "
        "ended_at=2026-07-30T10:02:30+0800 duration_seconds=150"
    )
    assert format_queue_pause({"kind": "task_failed", "task_id": "A01"}) == (
        "QUEUE_PAUSE kind=task_failed task=A01"
    )


def test_malformed_task_times_use_unknown_duration():
    task = {
        "task_id": "A01",
        "phase": "A",
        "gpu_index": 4,
        "status": "interrupted",
        "return_code": None,
        "started_at": "not-a-time",
        "ended_at": "also-not-a-time",
    }

    assert format_task_end(task).endswith("duration_seconds=-")
