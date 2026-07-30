from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


BEGIN = "# COMBINED_COMMANDS_BEGIN"
END = "# COMBINED_COMMANDS_END"
ALLOWED_ENTRY_POINTS = {"src/combiner_train.py", "src/clip_fine_tune.py"}
ALLOWED_ENV = {"CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE"}
PATH_OPTIONS = {
    "--fashioniq-root",
    "--clip-model-path",
    "--retizero-base-path",
    "--retfound-backbone-path",
    "--blip-model-name",
}


class PreflightError(RuntimeError):
    pass


class ProbeError(RuntimeError):
    pass


class ResumeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    phase: str
    ordinal: int
    argv: Tuple[str, ...]
    env: Dict[str, str]
    log_name: str
    source: str


@dataclass(frozen=True)
class QueueSpec:
    tasks: Tuple[TaskSpec, ...]
    command_sha256: str


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    memory_used_mib: int
    utilization_percent: int
    compute_pids: Tuple[int, ...] = ()


class IdlePolicy:
    def __init__(
        self,
        expected_indices: Iterable[int],
        memory_limit_mib: int = 512,
        utilization_limit_percent: int = 5,
        required_samples: int = 5,
    ):
        self.expected_indices = tuple(expected_indices)
        self.memory_limit_mib = memory_limit_mib
        self.utilization_limit_percent = utilization_limit_percent
        self.required_samples = required_samples
        self._uuid_by_index: Optional[Dict[int, str]] = None
        self._counts: Dict[str, int] = {}

    def is_idle_now(self, snapshot: GpuSnapshot) -> bool:
        return (
            not snapshot.compute_pids
            and snapshot.memory_used_mib <= self.memory_limit_mib
            and snapshot.utilization_percent <= self.utilization_limit_percent
        )

    def idle_streak(self, gpu_uuid: str) -> int:
        return int(self._counts.get(gpu_uuid, 0))

    def observe(self, snapshots: Sequence[GpuSnapshot]) -> Tuple[int, ...]:
        by_index = {snapshot.index: snapshot for snapshot in snapshots}
        if tuple(sorted(by_index)) != tuple(sorted(self.expected_indices)):
            raise ProbeError(
                f"expected GPU indices {self.expected_indices}, found {tuple(sorted(by_index))}"
            )
        mapping = {index: by_index[index].uuid for index in self.expected_indices}
        if self._uuid_by_index is None:
            self._uuid_by_index = mapping
        elif mapping != self._uuid_by_index:
            raise ProbeError("GPU UUID mapping changed")

        eligible = []
        for index in self.expected_indices:
            snapshot = by_index[index]
            if self.is_idle_now(snapshot):
                self._counts[snapshot.uuid] = self._counts.get(snapshot.uuid, 0) + 1
            else:
                self._counts[snapshot.uuid] = 0
            if self._counts[snapshot.uuid] >= self.required_samples:
                eligible.append(index)
        return tuple(sorted(eligible))


def _phase_lines(combined: str, phase: str, next_phase: str) -> list[str]:
    marker = f"# Phase {phase}:"
    next_marker = f"# Phase {next_phase}:"
    if combined.count(marker) != 1 or combined.count(next_marker) != 1:
        raise PreflightError(f"missing or duplicate phase marker: {marker}")
    section = combined.split(marker, 1)[1].split(next_marker, 1)[0]
    return [line for line in section.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]


def _parse_line(
    line: str,
    phase: str,
    ordinal: int,
    python_executable: Path,
) -> TaskSpec:
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError as exc:
        raise PreflightError(f"invalid shell quoting in {phase}{ordinal:02d}: {exc}") from exc

    forbidden = {";", "&&", "||", "|", "<", "<<", ">>"}
    if any(token in forbidden or "$(" in token or "`" in token for token in tokens):
        raise PreflightError(f"unsupported shell token in {phase}{ordinal:02d}")
    if len(tokens) < 8 or tokens[-4] != ">" or tokens[-2:] != ["2>&1", "&"]:
        raise PreflightError(f"invalid redirection/background syntax in {phase}{ordinal:02d}")

    try:
        nohup_index = tokens.index("nohup")
    except ValueError as exc:
        raise PreflightError(f"missing nohup in {phase}{ordinal:02d}") from exc
    env: Dict[str, str] = {}
    for token in tokens[:nohup_index]:
        if "=" not in token:
            raise PreflightError(f"invalid environment assignment in {phase}{ordinal:02d}")
        key, value = token.split("=", 1)
        if key not in ALLOWED_ENV or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
            raise PreflightError(f"unsupported environment assignment: {key}")
        env[key] = value
    if set(env) != ALLOWED_ENV:
        raise PreflightError(f"required environment assignments missing in {phase}{ordinal:02d}")

    command = tokens[nohup_index + 1 : -4]
    if len(command) < 2 or command[0] != "python":
        raise PreflightError(f"expected nohup python in {phase}{ordinal:02d}")
    entry_point = command[1]
    if entry_point not in ALLOWED_ENTRY_POINTS:
        raise PreflightError(f"unsupported entry point: {entry_point}")

    log_name = tokens[-3]
    if Path(log_name).is_absolute() or Path(log_name).name != log_name or not log_name.endswith(".log"):
        raise PreflightError(f"unsafe log name: {log_name}")
    argv = (str(python_executable), entry_point, *command[2:])
    return TaskSpec(
        task_id=f"{phase}{ordinal:02d}",
        phase=phase,
        ordinal=ordinal,
        argv=tuple(argv),
        env=env,
        log_name=log_name,
        source=line,
    )


def parse_combined_queue(
    command_file: Path,
    project_root: Path,
    python_executable: Path,
) -> QueueSpec:
    del project_root
    text = Path(command_file).read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise PreflightError("Combined begin/end markers must each appear exactly once")
    combined = text.split(BEGIN, 1)[1].split(END, 1)[0]
    lines_by_phase = {
        "A": _phase_lines(combined, "A", "B"),
        "B": _phase_lines(combined, "B", "C"),
    }
    for phase, lines in lines_by_phase.items():
        if len(lines) != 10:
            raise PreflightError(f"Phase {phase} must contain exactly 10 commands; found {len(lines)}")

    tasks = tuple(
        _parse_line(line, phase, ordinal, python_executable)
        for phase in ("A", "B")
        for ordinal, line in enumerate(lines_by_phase[phase], 1)
    )
    logs = [task.log_name for task in tasks]
    if len(set(logs)) != len(logs):
        raise PreflightError("duplicate log names in Combined queue")
    digest_source = "\n".join(task.source for task in tasks).encode("utf-8")
    return QueueSpec(tasks=tasks, command_sha256=hashlib.sha256(digest_source).hexdigest())


def _accepted_options(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
            options.add(first.value)
    return options


def preflight_queue(queue: QueueSpec, project_root: Path) -> None:
    errors = []
    option_cache: Dict[str, set[str]] = {}
    for task in queue.tasks:
        python_path = Path(task.argv[0])
        script_path = project_root / task.argv[1]
        if not python_path.is_file():
            errors.append(f"missing Python executable: {python_path}")
        if not script_path.is_file():
            errors.append(f"missing entry point: {script_path}")
            accepted = set()
        else:
            accepted = option_cache.setdefault(task.argv[1], _accepted_options(script_path))

        for token in task.argv[2:]:
            if token.startswith("--") and token not in accepted:
                errors.append(f"{task.task_id}: unsupported option {token}")

        for index, token in enumerate(task.argv[:-1]):
            if token not in PATH_OPTIONS:
                continue
            value = task.argv[index + 1]
            path = Path(value)
            if path.is_absolute() and not path.exists():
                errors.append(f"{task.task_id}: missing path {value}")

    if errors:
        raise PreflightError("\n".join(errors))


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"succeeded", "failed", "conflict_stopped", "interrupted"}


def initial_state(
    queue: QueueSpec,
    run_id: str,
    created_at: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "command_sha256": queue.command_sha256,
        "paused_reason": None,
        "tasks": [
            {
                "task_id": task.task_id,
                "phase": task.phase,
                "ordinal": task.ordinal,
                "argv": list(task.argv),
                "env": dict(task.env),
                "log_name": task.log_name,
                "status": "pending",
                "gpu_index": None,
                "gpu_uuid": None,
                "pid": None,
                "pgid": None,
                "start_ticks": None,
                "process_command_sha256": None,
                "manifest_path": None,
                "result_path": None,
                "log_path": None,
                "started_at": None,
                "ended_at": None,
                "return_code": None,
                "conflict": None,
            }
            for task in queue.tasks
        ],
    }


def _state_task(state: dict, task_id: str) -> dict:
    matches = [task for task in state["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ResumeError(f"state has invalid task id: {task_id}")
    return matches[0]


def next_pending_task(state: dict) -> Optional[dict]:
    if state.get("paused_reason"):
        return None
    phase_a = [task for task in state["tasks"] if task["phase"] == "A"]
    phase_b = [task for task in state["tasks"] if task["phase"] == "B"]
    if not all(task["status"] == "succeeded" for task in phase_a):
        candidates = [task for task in phase_a if task["status"] == "pending"]
    else:
        candidates = [task for task in phase_b if task["status"] == "pending"]
    return min(candidates, key=lambda item: item["ordinal"]) if candidates else None


def mark_running(
    state: dict,
    task_id: str,
    launch: dict,
    gpu: GpuSnapshot,
    started_at: str,
) -> None:
    task = _state_task(state, task_id)
    if task["status"] != "pending":
        raise ResumeError(f"task {task_id} is not pending")
    task.update(
        {
            "status": "running",
            "gpu_index": gpu.index,
            "gpu_uuid": gpu.uuid,
            "pid": launch["pid"],
            "pgid": launch["pgid"],
            "start_ticks": launch["start_ticks"],
            "process_command_sha256": launch["command_sha256"],
            "manifest_path": launch["manifest_path"],
            "result_path": launch["result_path"],
            "log_path": launch["log_path"],
            "started_at": started_at,
        }
    )


def record_exit(state: dict, task_id: str, return_code: int, ended_at: str) -> None:
    task = _state_task(state, task_id)
    if task["status"] != "running":
        raise ResumeError(f"task {task_id} is not running")
    task["return_code"] = int(return_code)
    task["ended_at"] = ended_at
    task["status"] = "succeeded" if return_code == 0 else "failed"
    if return_code != 0 and not state.get("paused_reason"):
        state["paused_reason"] = {
            "kind": "task_failed",
            "task_id": task_id,
            "return_code": int(return_code),
            "detected_at": ended_at,
        }


def record_conflict(
    state: dict,
    task_id: str,
    unknown_pids: Sequence[int],
    detected_at: str,
    owners: Optional[Dict[int, str]] = None,
) -> None:
    task = _state_task(state, task_id)
    task["status"] = "conflict_stopped"
    task["ended_at"] = detected_at
    task["conflict"] = {
        "unknown_pids": sorted(int(pid) for pid in unknown_pids),
        "owners": {str(pid): owner for pid, owner in (owners or {}).items()},
    }
    state["paused_reason"] = {
        "kind": "foreign_gpu_process",
        "task_id": task_id,
        "unknown_pids": sorted(int(pid) for pid in unknown_pids),
        "detected_at": detected_at,
    }


def record_interrupted(state: dict, task_id: str, detected_at: str) -> None:
    task = _state_task(state, task_id)
    task["status"] = "interrupted"
    task["ended_at"] = detected_at
    state["paused_reason"] = {
        "kind": "process_identity_unverified",
        "task_id": task_id,
        "detected_at": detected_at,
    }


def validate_resume_state(state: dict, queue: QueueSpec) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ResumeError("unsupported state schema version")
    if state.get("command_sha256") != queue.command_sha256:
        raise ResumeError("command digest changed")
    expected = {
        task.task_id: (
            task.phase,
            task.ordinal,
            list(task.argv),
            task.env,
            task.log_name,
        )
        for task in queue.tasks
    }
    actual = {}
    for task in state.get("tasks", []):
        task_id = task.get("task_id")
        if task_id in actual:
            raise ResumeError(f"duplicate task in state: {task_id}")
        actual[task_id] = (
            task.get("phase"),
            task.get("ordinal"),
            task.get("argv"),
            task.get("env"),
            task.get("log_name"),
        )
    if actual != expected:
        raise ResumeError("state task definitions do not match queue")


class AtomicStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeError(f"cannot load state: {exc}") from exc

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
