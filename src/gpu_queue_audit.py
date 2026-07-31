from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Optional

try:
    from gpu_queue_core import GpuSnapshot
except ImportError:
    from src.gpu_queue_core import GpuSnapshot


def _csv(values: Iterable[object]) -> str:
    items = [str(value) for value in values]
    return ",".join(items) if items else "-"


def _owners(pids: Iterable[int], owner_lookup: Callable[[int], str]) -> str:
    owners = []
    for pid in pids:
        try:
            owners.append(owner_lookup(pid))
        except Exception:
            owners.append("unknown")
    return _csv(owners)


def _timestamp(epoch: object) -> str:
    try:
        return datetime.fromtimestamp(float(epoch)).astimezone().strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return "-"


def format_lease_event(event: str, lease: dict, *, reason: Optional[str] = None) -> str:
    fields = [
        event,
        f"gpu={lease['gpu_index']}",
        f"gpu_uuid={lease['gpu_uuid']}",
    ]
    if event in {"GPU_LEASE_ACQUIRED", "GPU_LEASE_REUSED"}:
        fields.append(f"task={lease.get('task_id') or '-'}")
    if event == "GPU_LEASE_COOLDOWN":
        fields.extend(
            (
                f"previous_task={lease.get('previous_task_id') or '-'}",
                f"cooldown_ready_at={_timestamp(lease.get('cooldown_ready_at'))}",
            )
        )
    if event == "GPU_LEASE_RELEASED":
        if not reason:
            raise ValueError("GPU_LEASE_RELEASED requires a reason")
        fields.append(f"reason={_single_line(reason)}")
    return " ".join(fields)


def format_queue_complete(summary: dict) -> str:
    return (
        f"QUEUE_COMPLETE total={summary['total']} "
        f"succeeded={summary['succeeded']} failed={summary['failed']} "
        f"launch_failed={summary['launch_failed']} "
        f"interrupted={summary['interrupted']}"
    )


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
) -> str:
    lease_state = str(lease["state"]).upper() if lease is not None else "NONE"
    lease_detail = (
        f"leases={active_lease_count}/{max_gpu_leases} lease={lease_state}"
    )
    if lease is not None and lease.get("state") == "cooldown":
        lease_detail += (
            f" previous_task={lease.get('previous_task_id') or '-'}"
            f" cooldown_ready_at={_timestamp(lease.get('cooldown_ready_at'))}"
        )
    prefix = (
        f"GPU_AUDIT gpu={snapshot.index} uuid={snapshot.uuid} "
        f"{lease_detail} status={{status}} memory_mib={snapshot.memory_used_mib} "
        f"util_percent={snapshot.utilization_percent}"
    )
    compute_pids = tuple(sorted(set(snapshot.compute_pids)))
    if running_task is None:
        if idle_now:
            return prefix.format(status="IDLE") + f" idle_streak={idle_streak}/5"
        return (
            prefix.format(status="FOREIGN")
            + f" pids={_csv(compute_pids)} owners={_owners(compute_pids, owner_lookup)}"
        )

    owned = set(owned_compute_pids)
    our_pids = tuple(pid for pid in compute_pids if pid in owned)
    foreign_pids = tuple(pid for pid in compute_pids if pid not in owned)
    base = (
        prefix.format(status="SHARED" if foreign_pids else "OURS")
        + f" task={running_task['task_id']} our_pid={running_task['pid']}"
        + f" our_pids={_csv(our_pids)}"
    )
    if not foreign_pids:
        return base
    return (
        base
        + f" foreign_pids={_csv(foreign_pids)}"
        + f" foreign_owners={_owners(foreign_pids, owner_lookup)}"
    )


def format_task_start(task: dict) -> str:
    return (
        f"TASK_START task={task['task_id']} phase={task['phase']} "
        f"gpu={task['gpu_index']} gpu_uuid={task['gpu_uuid']} "
        f"pid={task['pid']} log_path={task['log_path']}"
    )


def _duration_seconds(started_at: object, ended_at: object) -> str:
    try:
        started = datetime.strptime(str(started_at), "%Y-%m-%dT%H:%M:%S%z")
        ended = datetime.strptime(str(ended_at), "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return "-"
    seconds = int((ended - started).total_seconds())
    return str(seconds) if seconds >= 0 else "-"


def format_task_end(task: dict) -> str:
    return_code = task.get("return_code")
    return (
        f"TASK_END task={task['task_id']} phase={task['phase']} "
        f"gpu={task.get('gpu_index', '-')} result={str(task['status']).upper()} "
        f"return_code={return_code if return_code is not None else '-'} "
        f"started_at={task.get('started_at') or '-'} "
        f"ended_at={task.get('ended_at') or '-'} "
        f"duration_seconds={_duration_seconds(task.get('started_at'), task.get('ended_at'))}"
    )


def _single_line(value: object) -> str:
    return "_".join(str(value).split())


def format_queue_pause(reason: dict) -> str:
    fields = [f"QUEUE_PAUSE kind={_single_line(reason.get('kind', 'unknown'))}"]
    ordered = (
        ("task_id", "task"),
        ("return_code", "return_code"),
        ("detail", "detail"),
        ("detected_at", "detected_at"),
    )
    for source, target in ordered:
        if source in reason:
            fields.append(f"{target}={_single_line(reason[source])}")
    if "unknown_pids" in reason:
        fields.append(f"unknown_pids={_csv(reason['unknown_pids'])}")
    return " ".join(fields)
