from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

try:
    from gpu_queue_core import (
        AtomicStateStore,
        GpuSnapshot,
        GpuMappingError,
        IdlePolicy,
        MAX_GPU_LEASES,
        ProbeError,
        QueueSpec,
        ResumeError,
        TaskSpec,
        acquire_lease,
        all_tasks_terminal,
        initial_state,
        lease_for_gpu,
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
        validate_resume_state,
    )
    from gpu_queue_audit import (
        format_gpu_audit,
        format_lease_event,
        format_queue_complete,
        format_task_end,
        format_task_start,
    )
except ImportError:
    from src.gpu_queue_core import (
        AtomicStateStore,
        GpuSnapshot,
        GpuMappingError,
        IdlePolicy,
        MAX_GPU_LEASES,
        ProbeError,
        QueueSpec,
        ResumeError,
        TaskSpec,
        acquire_lease,
        all_tasks_terminal,
        initial_state,
        lease_for_gpu,
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
        validate_resume_state,
    )
    from src.gpu_queue_audit import (
        format_gpu_audit,
        format_lease_event,
        format_queue_complete,
        format_task_end,
        format_task_start,
    )


class ProcessIdentityError(RuntimeError):
    pass


class AlreadyRunningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int
    command_sha256: str


@dataclass(frozen=True)
class LaunchRecord:
    identity: ProcessIdentity
    manifest_path: Path
    result_path: Path
    log_path: Path

    def state_fields(self) -> dict:
        return {
            "pid": self.identity.pid,
            "pgid": self.identity.pgid,
            "start_ticks": self.identity.start_ticks,
            "command_sha256": self.identity.command_sha256,
            "manifest_path": str(self.manifest_path),
            "result_path": str(self.result_path),
            "log_path": str(self.log_path),
        }


class ProcInspector:
    def __init__(self, proc_root: Path = Path("/proc")):
        self.proc_root = Path(proc_root)

    def _stat(self, pid: int):
        text = (self.proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        closing = text.rfind(")")
        if closing < 0:
            raise ProcessIdentityError(f"malformed /proc stat for PID {pid}")
        fields = text[closing + 2 :].split()
        return int(fields[1]), int(fields[2]), int(fields[19])

    def identity(self, pid: int) -> Optional[ProcessIdentity]:
        try:
            _, pgid, start_ticks = self._stat(pid)
            command = (self.proc_root / str(pid) / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, ValueError, IndexError) as exc:
            raise ProcessIdentityError(f"cannot inspect PID {pid}: {exc}") from exc
        return ProcessIdentity(
            pid=pid,
            pgid=pgid,
            start_ticks=start_ticks,
            command_sha256=hashlib.sha256(command).hexdigest(),
        )

    def descendants(self, root_pid: int) -> set[int]:
        parent_by_pid = {}
        for entry in self.proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                parent, _, _ = self._stat(pid)
            except (OSError, ValueError, IndexError, ProcessIdentityError):
                continue
            parent_by_pid[pid] = parent
        descendants = set()
        changed = True
        while changed:
            changed = False
            for pid, parent in parent_by_pid.items():
                if pid not in descendants and (parent == root_pid or parent in descendants):
                    descendants.add(pid)
                    changed = True
        return descendants


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class OwnedProcessLauncher:
    def __init__(
        self,
        project_root: Path,
        python_executable: Path,
        worker_script: Path,
        proc_inspector=None,
        popen_factory=subprocess.Popen,
        killpg=os.killpg,
        sleeper=time.sleep,
        monotonic=time.monotonic,
    ):
        self.project_root = Path(project_root)
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.proc_inspector = proc_inspector or ProcInspector()
        self.popen_factory = popen_factory
        self.killpg = killpg
        self.sleeper = sleeper
        self.monotonic = monotonic

    def start(self, task: TaskSpec, gpu: GpuSnapshot, task_dir: Path) -> LaunchRecord:
        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = task_dir / "manifest.json"
        result_path = task_dir / "result.json"
        log_path = task_dir / task.log_name
        overrides = {
            "CUDA_VISIBLE_DEVICES": str(gpu.index),
            "NCCL_P2P_DISABLE": task.env.get("NCCL_P2P_DISABLE", "1"),
        }
        _atomic_json(
            manifest_path,
            {
                "argv": list(task.argv),
                "cwd": str(self.project_root),
                "env_overrides": overrides,
                "log_path": str(log_path),
                "result_path": str(result_path),
            },
        )
        environment = os.environ.copy()
        environment.update(overrides)
        process = self.popen_factory(
            [
                str(self.python_executable),
                str(self.worker_script),
                "--manifest",
                str(manifest_path),
            ],
            cwd=str(self.project_root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
        )
        identity = self.proc_inspector.identity(process.pid)
        if identity is None:
            raise ProcessIdentityError(f"worker PID {process.pid} disappeared during launch")
        return LaunchRecord(identity, manifest_path, result_path, log_path)

    def poll(self, identity: ProcessIdentity, result_path: Path) -> Optional[int]:
        result_path = Path(result_path)
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                return_code = payload["return_code"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ProcessIdentityError(f"invalid worker result: {exc}") from exc
            if not isinstance(return_code, int):
                raise ProcessIdentityError("worker result return_code is not an integer")
            return return_code
        if self.proc_inspector.identity(identity.pid) == identity:
            return None
        raise ProcessIdentityError("worker disappeared without a valid result")

    def terminate(self, identity: ProcessIdentity, grace_seconds: int = 30) -> bool:
        if self.proc_inspector.identity(identity.pid) != identity:
            return False
        self.killpg(identity.pgid, signal.SIGTERM)
        deadline = self.monotonic() + max(0, grace_seconds)
        while self.monotonic() < deadline:
            if self.proc_inspector.identity(identity.pid) is None:
                return True
            self.sleeper(0.25)
        current = self.proc_inspector.identity(identity.pid)
        if current is None:
            return True
        if current != identity:
            return False
        self.killpg(identity.pgid, signal.SIGKILL)
        return True


def _default_owner_lookup(pid: int) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (KeyError, OSError):
        return "unknown"


class Dispatcher:
    def __init__(
        self,
        queue: QueueSpec,
        state: dict,
        run_dir: Path,
        probe,
        idle_policy,
        launcher: OwnedProcessLauncher,
        proc_inspector,
        state_store,
        sleeper=time.sleep,
        now=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        clock=time.time,
        cooldown_seconds=60,
        max_gpu_leases=MAX_GPU_LEASES,
        owner_lookup=_default_owner_lookup,
        event_logger=lambda message: None,
    ):
        self.queue = queue
        self.state = state
        self.run_dir = Path(run_dir)
        self.probe = probe
        self.idle_policy = idle_policy
        self.launcher = launcher
        self.proc_inspector = proc_inspector
        self.state_store = state_store
        self.sleeper = sleeper
        self.now = now
        self.clock = clock
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_gpu_leases = int(max_gpu_leases)
        self.owner_lookup = owner_lookup
        self.event_logger = event_logger
        self._task_specs = {task.task_id: task for task in queue.tasks}
        self._completion_logged = False

    def _running_tasks(self):
        return [task for task in self.state["tasks"] if task["status"] == "running"]

    @staticmethod
    def _identity(task: dict) -> ProcessIdentity:
        return ProcessIdentity(
            pid=int(task["pid"]),
            pgid=int(task["pgid"]),
            start_ticks=int(task["start_ticks"]),
            command_sha256=str(task["process_command_sha256"]),
        )

    def _save(self):
        self.state_store.save(self.state)

    def _log_probe_error(self, exc: Exception) -> None:
        detail = "_".join(str(exc).split()) or "unknown"
        self.event_logger(f"GPU_PROBE_ERROR detail={detail}")

    def _poll_results(self):
        for task in list(self._running_tasks()):
            identity = self._identity(task)
            try:
                return_code = self.launcher.poll(identity, Path(task["result_path"]))
            except ProcessIdentityError as exc:
                record_interrupted(self.state, task["task_id"], self.now())
                task["error"] = " ".join(str(exc).split())
                lease = lease_for_gpu(self.state, int(task["gpu_index"]))
                if lease is not None:
                    self._release(lease, "interrupted", save=False)
                self._save()
                self.event_logger(format_task_end(task))
                continue
            if return_code is not None:
                record_exit(self.state, task["task_id"], return_code, self.now())
                lease = mark_lease_cooldown(
                    self.state,
                    task["task_id"],
                    self.clock() + self.cooldown_seconds,
                    self.now(),
                )
                self._save()
                self.event_logger(format_task_end(task))
                self.event_logger(format_lease_event("GPU_LEASE_COOLDOWN", lease))

    @staticmethod
    def _gpu(snapshots: Tuple[GpuSnapshot, ...], index: int) -> Optional[GpuSnapshot]:
        return next((gpu for gpu in snapshots if gpu.index == index), None)

    def _final_idle_check(self, index: int, expected_uuid: str) -> Optional[GpuSnapshot]:
        first = self._gpu(self.probe.snapshot(), index)
        if first is None or first.uuid != expected_uuid or not self.idle_policy.is_idle_now(first):
            return None
        self.sleeper(3)
        second = self._gpu(self.probe.snapshot(), index)
        if second is None or second.uuid != expected_uuid or not self.idle_policy.is_idle_now(second):
            return None
        return second

    @staticmethod
    def _reuse_rejection(
        snapshot: Optional[GpuSnapshot],
        expected_uuid: str,
    ) -> Optional[str]:
        if snapshot is None:
            return "gpu_missing"
        if snapshot.uuid != expected_uuid:
            return "uuid_changed"
        if snapshot.compute_pids:
            return "foreign_pid"
        if snapshot.memory_used_mib > 512:
            return "memory_above_limit"
        if snapshot.utilization_percent > 5:
            return "utilization_above_limit"
        return None

    def _release(self, lease: dict, reason: str, *, save: bool = True) -> None:
        released = release_lease(self.state, int(lease["gpu_index"]))
        reset = getattr(self.idle_policy, "reset", None)
        if reset is not None:
            reset(released["gpu_uuid"])
        if save:
            self._save()
        self.event_logger(
            format_lease_event(
                "GPU_LEASE_RELEASED",
                released,
                reason=reason,
            )
        )

    def _dispatch_on_gpu(
        self,
        pending: dict,
        gpu: GpuSnapshot,
        *,
        reuse: bool,
    ) -> bool:
        spec = self._task_specs[pending["task_id"]]
        try:
            launch = self.launcher.start(
                spec,
                gpu,
                self.run_dir / "tasks" / spec.task_id,
            )
        except (OSError, ProcessIdentityError, ValueError) as exc:
            record_launch_failed(
                self.state,
                spec.task_id,
                str(exc),
                self.now(),
                gpu,
            )
            if reuse:
                lease = lease_for_gpu(self.state, gpu.index)
                if lease is not None:
                    self._release(lease, "launch_failed", save=False)
            self._save()
            self.event_logger(format_task_end(pending))
            return False

        mark_running(
            self.state,
            spec.task_id,
            launch.state_fields(),
            gpu,
            self.now(),
        )
        if reuse:
            lease = lease_for_gpu(self.state, gpu.index)
            if lease is None or lease.get("state") != "cooldown":
                raise ResumeError(f"GPU {gpu.index} has no reusable cooldown lease")
            lease.update(
                {
                    "state": "running",
                    "task_id": spec.task_id,
                    "cooldown_ready_at": None,
                    "updated_at": self.now(),
                }
            )
            event = "GPU_LEASE_REUSED"
        else:
            lease = acquire_lease(self.state, gpu, spec.task_id, self.now())
            event = "GPU_LEASE_ACQUIRED"
        self._save()
        self.event_logger(format_lease_event(event, lease))
        self.event_logger(format_task_start(pending))
        return True

    def _process_ready_cooldowns(
        self,
        snapshots: Tuple[GpuSnapshot, ...],
    ) -> None:
        ready = sorted(
            (
                lease
                for lease in self.state.get("leases", [])
                if lease.get("state") == "cooldown"
                and float(lease["cooldown_ready_at"]) <= self.clock()
            ),
            key=lambda lease: int(lease["gpu_index"]),
        )
        for lease in ready:
            pending = next_pending_task(self.state)
            if pending is None:
                continue
            snapshot = self._gpu(snapshots, int(lease["gpu_index"]))
            reason = self._reuse_rejection(snapshot, lease["gpu_uuid"])
            if reason is not None:
                self._release(lease, reason)
                continue
            try:
                final_gpu = self._final_idle_check(
                    int(lease["gpu_index"]),
                    lease["gpu_uuid"],
                )
            except ProbeError:
                raise
            if final_gpu is None:
                self._release(lease, "final_check_failed")
                continue
            self._dispatch_on_gpu(pending, final_gpu, reuse=True)

    def _audit_snapshots(self, snapshots: Tuple[GpuSnapshot, ...]) -> None:
        running_by_gpu = {
            int(task["gpu_index"]): task
            for task in self._running_tasks()
            if task.get("gpu_index") is not None
        }
        self.event_logger("GPU_AUDIT_BEGIN")
        for snapshot in sorted(snapshots, key=lambda item: item.index):
            task = running_by_gpu.get(snapshot.index)
            owned_compute_pids = ()
            if task is not None:
                identity = self._identity(task)
                owned_process_tree = self.proc_inspector.descendants(identity.pid) | {
                    identity.pid
                }
                owned_compute_pids = tuple(
                    sorted(set(snapshot.compute_pids) & owned_process_tree)
                )
            self.event_logger(
                format_gpu_audit(
                    snapshot,
                    idle_now=self.idle_policy.is_idle_now(snapshot),
                    idle_streak=self.idle_policy.idle_streak(snapshot.uuid),
                    running_task=task,
                    lease=lease_for_gpu(self.state, snapshot.index),
                    active_lease_count=len(self.state.get("leases", [])),
                    max_gpu_leases=self.max_gpu_leases,
                    owned_compute_pids=owned_compute_pids,
                    owner_lookup=self.owner_lookup,
                )
            )
        self.event_logger("GPU_AUDIT_END")

    def run_cycle(self, sample_idle: bool) -> None:
        self._poll_results()
        try:
            snapshots = self.probe.snapshot()
        except GpuMappingError:
            raise
        except ProbeError as exc:
            self._log_probe_error(exc)
            return
        candidates = ()
        if sample_idle and not self.state.get("paused_reason"):
            try:
                candidates = self.idle_policy.observe(snapshots)
            except GpuMappingError:
                raise
            except ProbeError as exc:
                self._log_probe_error(exc)
                return
        if sample_idle:
            self._audit_snapshots(snapshots)
        if self.state.get("paused_reason"):
            return
        try:
            self._process_ready_cooldowns(snapshots)
        except GpuMappingError:
            raise
        except ProbeError as exc:
            self._log_probe_error(exc)
            return
        if not sample_idle:
            return
        leased = {
            int(lease["gpu_index"]) for lease in self.state.get("leases", [])
        }
        for index in candidates:
            if len(self.state.get("leases", [])) >= self.max_gpu_leases:
                break
            if index in leased:
                continue
            pending = next_pending_task(self.state)
            if pending is None:
                break
            initial_gpu = self._gpu(snapshots, index)
            if initial_gpu is None:
                continue
            try:
                final_gpu = self._final_idle_check(index, initial_gpu.uuid)
            except GpuMappingError:
                raise
            except ProbeError as exc:
                self._log_probe_error(exc)
                return
            if final_gpu is None:
                continue
            self._dispatch_on_gpu(pending, final_gpu, reuse=False)
            if lease_for_gpu(self.state, index) is not None:
                leased.add(index)

    def reconcile_resume(self) -> None:
        self._poll_results()

    def run_forever(self) -> int:
        cycle = 0
        while True:
            if all_tasks_terminal(self.state):
                leases = list(self.state.get("leases", []))
                for lease in leases:
                    self._release(lease, "queue_complete", save=False)
                if leases:
                    self._save()
                if not self._completion_logged:
                    summary = terminal_summary(self.state)
                    self.event_logger(format_queue_complete(summary))
                    self._completion_logged = True
                return 0
            self.run_cycle(sample_idle=(cycle % 2 == 0))
            running = self._running_tasks()
            if self.state.get("paused_reason") and not running:
                return 2
            cycle += 1
            self.sleeper(30)


class DispatcherLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown"
            handle.close()
            raise AlreadyRunningError(f"dispatcher already running (PID {owner})") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely dispatch Combined Phase A/B jobs to audited idle GPUs."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--resume", type=Path, metavar="RUN_DIR")
    return parser


def _default_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        project_root=project_root,
        command_file=project_root / "命令.sh",
        python_executable=Path("/data0/qrchen/miniconda3/envs/clip4cir/bin/python"),
        runtime_root=project_root / "gpu_queue_runs",
        probe=NvidiaSmiProbe(),
        output=print,
    )


def _audit_line(gpu: GpuSnapshot) -> str:
    reasons = []
    if gpu.compute_pids:
        reasons.append(f"compute_pids={','.join(str(pid) for pid in gpu.compute_pids)}")
    if gpu.memory_used_mib > 512:
        reasons.append(f"memory={gpu.memory_used_mib}MiB")
    if gpu.utilization_percent > 5:
        reasons.append(f"utilization={gpu.utilization_percent}%")
    status = "unavailable" if reasons else "idle-now (needs 5 consecutive samples)"
    detail = "; ".join(reasons) if reasons else "no compute PID; memory/utilization within limits"
    return f"GPU {gpu.index} {gpu.uuid}: {status}; {detail}"


def _event_logger(path: Path, output):
    def log(message: str):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"{timestamp} {message}"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        output(line)

    return log


def main(argv=None, dependencies=None) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    queue = parse_combined_queue(
        deps.command_file,
        deps.project_root,
        deps.python_executable,
    )
    preflight_queue(queue, deps.project_root)

    if args.dry_run:
        snapshots = deps.probe.snapshot()
        deps.output(
            f"preflight ok: {len(queue.tasks)} commands "
            f"(Phase A=10, Phase B=10), digest={queue.command_sha256}"
        )
        for gpu in snapshots:
            deps.output(_audit_line(gpu))
        deps.output("dry-run only: no state created and no training process launched")
        return 0

    runtime_root = Path(deps.runtime_root)
    lock = DispatcherLock(runtime_root / "dispatcher.lock")
    with lock:
        if args.run:
            run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
            run_dir = runtime_root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            state = initial_state(
                queue,
                run_id,
                time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            store = AtomicStateStore(run_dir / "state.json")
            store.save(state)
        else:
            run_dir = args.resume.resolve()
            store = AtomicStateStore(run_dir / "state.json")
            state = store.load()
            validate_resume_state(state, queue)

        logger = _event_logger(run_dir / "dispatcher.log", deps.output)
        inspector = ProcInspector()
        launcher = OwnedProcessLauncher(
            project_root=deps.project_root,
            python_executable=deps.python_executable,
            worker_script=deps.project_root / "src/gpu_queue_worker.py",
            proc_inspector=inspector,
        )
        dispatcher = Dispatcher(
            queue=queue,
            state=state,
            run_dir=run_dir,
            probe=deps.probe,
            idle_policy=IdlePolicy(expected_indices=range(8)),
            launcher=launcher,
            proc_inspector=inspector,
            state_store=store,
            event_logger=logger,
        )
        if args.resume:
            dispatcher.reconcile_resume()
        logger(f"dispatcher started in {run_dir}")
        return dispatcher.run_forever()


class NvidiaSmiProbe:
    GPU_QUERY = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    PROCESS_QUERY = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]

    def __init__(self, runner: Callable = subprocess.run):
        self.runner = runner

    def _run(self, argv) -> str:
        result = self.runner(
            argv,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
            raise ProbeError(detail)
        return result.stdout.strip()

    @staticmethod
    def _integer(value: str, field: str) -> int:
        value = value.strip()
        if not value.isdigit():
            raise ProbeError(f"invalid {field}: {value}")
        return int(value)

    def snapshot(self) -> Tuple[GpuSnapshot, ...]:
        gpu_output = self._run(self.GPU_QUERY)
        process_output = self._run(self.PROCESS_QUERY)
        gpu_rows = {}
        uuid_to_index = {}
        for line in gpu_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                raise ProbeError(f"malformed GPU row: {line}")
            index = self._integer(parts[0], "GPU index")
            uuid = parts[1]
            if index in gpu_rows or uuid in uuid_to_index or not uuid:
                raise ProbeError("duplicate GPU index or UUID")
            memory = self._integer(parts[2], "memory.used")
            utilization = self._integer(parts[3], "utilization.gpu")
            gpu_rows[index] = (uuid, memory, utilization)
            uuid_to_index[uuid] = index

        if tuple(sorted(gpu_rows)) != tuple(range(8)):
            raise ProbeError(f"expected GPU indices 0..7, found {tuple(sorted(gpu_rows))}")

        pids = {index: [] for index in gpu_rows}
        if process_output:
            for line in process_output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 4:
                    raise ProbeError(f"malformed compute process row: {line}")
                uuid = parts[0]
                if uuid not in uuid_to_index:
                    raise ProbeError(f"compute process references unknown GPU UUID: {uuid}")
                pid = self._integer(parts[1], "compute PID")
                pids[uuid_to_index[uuid]].append(pid)

        return tuple(
            GpuSnapshot(
                index=index,
                uuid=gpu_rows[index][0],
                memory_used_mib=gpu_rows[index][1],
                utilization_percent=gpu_rows[index][2],
                compute_pids=tuple(sorted(set(pids[index]))),
            )
            for index in range(8)
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AlreadyRunningError, ProbeError, ResumeError) as exc:
        print(f"dispatcher error: {exc}")
        raise SystemExit(2)
