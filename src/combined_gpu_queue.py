from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

try:
    from gpu_queue_core import GpuSnapshot, ProbeError, TaskSpec
except ImportError:
    from src.gpu_queue_core import GpuSnapshot, ProbeError, TaskSpec


class ProcessIdentityError(RuntimeError):
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
