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
