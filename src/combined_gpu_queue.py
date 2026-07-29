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
        IdlePolicy,
        ProbeError,
        QueueSpec,
        ResumeError,
        TaskSpec,
        initial_state,
        mark_running,
        next_pending_task,
        parse_combined_queue,
        preflight_queue,
        record_conflict,
        record_exit,
        record_interrupted,
        validate_resume_state,
    )
except ImportError:
    from src.gpu_queue_core import (
        AtomicStateStore,
        GpuSnapshot,
        IdlePolicy,
        ProbeError,
        QueueSpec,
        ResumeError,
        TaskSpec,
        initial_state,
        mark_running,
        next_pending_task,
        parse_combined_queue,
        preflight_queue,
        record_conflict,
        record_exit,
        record_interrupted,
        validate_resume_state,
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
        self.owner_lookup = owner_lookup
        self.event_logger = event_logger
        self._task_specs = {task.task_id: task for task in queue.tasks}

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

    def _pause_probe_error(self, exc: Exception):
        if not self.state.get("paused_reason"):
            self.state["paused_reason"] = {
                "kind": "gpu_probe_error",
                "detail": str(exc),
                "detected_at": self.now(),
            }
            self._save()

    def _poll_results(self):
        for task in list(self._running_tasks()):
            identity = self._identity(task)
            try:
                return_code = self.launcher.poll(identity, Path(task["result_path"]))
            except ProcessIdentityError:
                record_interrupted(self.state, task["task_id"], self.now())
                self._save()
                continue
            if return_code is not None:
                record_exit(self.state, task["task_id"], return_code, self.now())
                self._save()

    def _monitor_conflicts(self, snapshots: Tuple[GpuSnapshot, ...]):
        by_index = {gpu.index: gpu for gpu in snapshots}
        for task in list(self._running_tasks()):
            gpu = by_index.get(task["gpu_index"])
            if gpu is None or gpu.uuid != task["gpu_uuid"]:
                record_interrupted(self.state, task["task_id"], self.now())
                self._save()
                continue
            identity = self._identity(task)
            owned_pids = self.proc_inspector.descendants(identity.pid) | {identity.pid}
            unknown = sorted(set(gpu.compute_pids) - owned_pids)
            if not unknown:
                continue
            owners = {pid: self.owner_lookup(pid) for pid in unknown}
            record_conflict(self.state, task["task_id"], unknown, self.now(), owners)
            self._save()
            self.launcher.terminate(identity, grace_seconds=30)

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

    def run_cycle(self, sample_idle: bool) -> None:
        self._poll_results()
        try:
            snapshots = self.probe.snapshot()
        except ProbeError as exc:
            self._pause_probe_error(exc)
            return
        self._monitor_conflicts(snapshots)
        if self.state.get("paused_reason") or not sample_idle:
            return
        try:
            candidates = self.idle_policy.observe(snapshots)
        except ProbeError as exc:
            self._pause_probe_error(exc)
            return
        assigned = {task["gpu_index"] for task in self._running_tasks()}
        for index in candidates:
            if index in assigned:
                continue
            pending = next_pending_task(self.state)
            if pending is None:
                break
            initial_gpu = self._gpu(snapshots, index)
            if initial_gpu is None:
                continue
            try:
                final_gpu = self._final_idle_check(index, initial_gpu.uuid)
            except ProbeError as exc:
                self._pause_probe_error(exc)
                return
            if final_gpu is None:
                continue
            spec = self._task_specs[pending["task_id"]]
            launch = self.launcher.start(
                spec,
                final_gpu,
                self.run_dir / "tasks" / spec.task_id,
            )
            mark_running(
                self.state,
                spec.task_id,
                launch.state_fields(),
                final_gpu,
                self.now(),
            )
            self._save()
            assigned.add(index)

    def reconcile_resume(self) -> None:
        self._poll_results()

    def run_forever(self) -> int:
        cycle = 0
        while True:
            self.run_cycle(sample_idle=(cycle % 2 == 0))
            running = self._running_tasks()
            if self.state.get("paused_reason") and not running:
                return 2
            if all(task["status"] == "succeeded" for task in self.state["tasks"]):
                return 0
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AlreadyRunningError, ProbeError, ResumeError) as exc:
        print(f"dispatcher error: {exc}")
        raise SystemExit(2)


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
