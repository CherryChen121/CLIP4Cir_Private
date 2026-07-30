from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct
from typing import Callable, Optional, Tuple
import zipfile

from output_paths import slugify_component


CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})
LEGACY_RUN_RE = re.compile(
    r"^(clip_finetuned|combiner_trained)_on_(fiq|cirr)_(.+)_"
    r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"
    r"(?:_\d+)?(?:_pid\d+)?)$"
)
PID_RE = re.compile(r"_pid(\d+)$")


class LegacyPathError(ValueError):
    pass


@dataclass(frozen=True)
class LegacyRun:
    source: Path
    destination: Path
    dataset_slug: str
    stage: str
    model_name: str
    model_slug: str
    run_id: str
    checkpoints: Tuple[Path, ...]
    metrics: Tuple[Path, ...]
    newest_mtime_ns: int
    pid: Optional[int]
    classification: str
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ScanResult:
    runs: Tuple[LegacyRun, ...]
    reports: Tuple[Path, ...]
    unknown_paths: Tuple[Path, ...]
    source_root: Path
    output_root: Path


def _legacy_run_id(stamp: str) -> Tuple[str, Optional[int]]:
    pid_match = PID_RE.search(stamp)
    pid = int(pid_match.group(1)) if pid_match else None
    time_part = PID_RE.sub("", stamp)
    parsed = None
    for format_string in ("%Y-%m-%d_%H:%M:%S_%f", "%Y-%m-%d_%H:%M:%S"):
        try:
            parsed = datetime.strptime(time_part, format_string)
            break
        except ValueError:
            continue
    if parsed is None:
        raise LegacyPathError(f"invalid legacy timestamp: {stamp}")
    if parsed.microsecond:
        run_id = f"{parsed:%Y%m%d-%H%M%S-%f}"
    else:
        run_id = f"{parsed:%Y%m%d-%H%M%S}"
    if pid is not None:
        run_id = f"{run_id}-p{pid}"
    return run_id, pid


def _newest_mtime_ns(path: Path) -> int:
    newest = path.stat().st_mtime_ns
    for item in path.rglob("*"):
        newest = max(newest, item.lstat().st_mtime_ns)
    return newest


def parse_legacy_run(
    source_root: Path,
    run_root: Path,
    output_root: Path,
) -> LegacyRun:
    source_root = source_root.resolve()
    run_root = run_root.resolve()
    relative = run_root.relative_to(source_root).as_posix()
    match = LEGACY_RUN_RE.fullmatch(relative)
    if match is None:
        raise LegacyPathError(f"unrecognized legacy run path: {relative}")

    prefix, dataset_token, model_name, stamp = match.groups()
    dataset_slug = "fashioniq" if dataset_token == "fiq" else "cirr"
    stage = "clip-finetune" if prefix == "clip_finetuned" else "combiner"
    model_slug = slugify_component(model_name)
    run_id, pid = _legacy_run_id(stamp)
    checkpoints = tuple(
        sorted(
            path
            for path in run_root.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        )
    )
    metrics = tuple(
        sorted(
            path
            for path in run_root.glob("*metrics*.csv")
            if path.is_file()
        )
    )
    destination = (
        output_root.resolve()
        / dataset_slug
        / stage
        / model_slug
        / run_id
    )
    return LegacyRun(
        source=run_root,
        destination=destination,
        dataset_slug=dataset_slug,
        stage=stage,
        model_name=model_name,
        model_slug=model_slug,
        run_id=run_id,
        checkpoints=checkpoints,
        metrics=metrics,
        newest_mtime_ns=_newest_mtime_ns(run_root),
        pid=pid,
        classification="unclassified",
        reasons=(),
    )


def _valid_safetensors_container(path: Path) -> bool:
    file_size = path.stat().st_size
    if file_size < 8:
        return False
    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > file_size - 8:
            return False
        try:
            header = json.loads(file.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
    if not isinstance(header, dict):
        return False
    data_size = file_size - 8 - header_size
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            return False
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            return False
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            return False
    return True


def checkpoint_container_is_valid(path: Path) -> bool:
    if path.stat().st_size <= 0:
        return False
    if path.suffix.lower() == ".safetensors":
        return _valid_safetensors_container(path)
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _classify_run(
    run: LegacyRun,
    *,
    now: datetime,
    pid_is_alive: Callable[[int], bool],
    legacy_writer_pids: Tuple[int, ...],
) -> LegacyRun:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_seconds = now.timestamp() - run.newest_mtime_ns / 1_000_000_000
    nonempty_checkpoints = tuple(
        path for path in run.checkpoints if path.stat().st_size > 0
    )
    nonempty_metrics = tuple(
        path for path in run.metrics if path.stat().st_size > 0
    )
    pid_alive = run.pid is not None and pid_is_alive(run.pid)

    if legacy_writer_pids or pid_alive:
        reasons = []
        if legacy_writer_pids:
            reasons.append("legacy-writer-active")
        if pid_alive:
            reasons.append("pid-alive")
        return replace(
            run,
            classification="active",
            reasons=tuple(reasons),
        )

    if nonempty_checkpoints:
        if all(checkpoint_container_is_valid(path) for path in nonempty_checkpoints):
            return replace(run, classification="valid", reasons=())
        return replace(
            run,
            classification="unresolved",
            reasons=("checkpoint-format-invalid",),
        )

    if nonempty_metrics:
        return replace(
            run,
            classification="unresolved",
            reasons=("nonempty-metrics-without-checkpoint",),
        )

    if age_seconds < 24 * 60 * 60:
        return replace(
            run,
            classification="unresolved",
            reasons=("modified-within-24-hours",),
        )

    return replace(
        run,
        classification="failed",
        reasons=(
            "no-nonempty-checkpoint",
            "no-nonempty-metrics",
            "pid-not-alive",
            "older-than-24-hours",
        ),
    )


def _candidate_run_roots(source_root: Path) -> Tuple[Path, ...]:
    roots = {
        directory.parent.resolve()
        for directory in source_root.rglob("saved_models")
        if directory.is_dir()
    }
    roots.update(
        file.parent.resolve()
        for file in source_root.rglob("training_hyperparameters.json")
        if file.is_file()
    )
    return tuple(sorted(roots))


def _is_owned(path: Path, run_roots: Tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in run_roots)


def scan_legacy_outputs(
    source_root: Path,
    output_root: Path,
    *,
    now: datetime,
    pid_is_alive: Callable[[int], bool],
    legacy_writer_pids: Tuple[int, ...],
) -> ScanResult:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.exists():
        return ScanResult(
            runs=(),
            reports=(),
            unknown_paths=(),
            source_root=source_root,
            output_root=output_root,
        )

    candidate_roots = _candidate_run_roots(source_root)
    runs = []
    recognized_roots = []
    for run_root in candidate_roots:
        try:
            run = parse_legacy_run(source_root, run_root, output_root)
        except (LegacyPathError, ValueError):
            continue
        recognized_roots.append(run_root)
        runs.append(
            _classify_run(
                run,
                now=now,
                pid_is_alive=pid_is_alive,
                legacy_writer_pids=legacy_writer_pids,
            )
        )

    recognized_root_tuple = tuple(recognized_roots)
    reports = []
    unknown_paths = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or _is_owned(path, recognized_root_tuple):
            continue
        if path.suffix.lower() == ".xlsx":
            reports.append(path)
        else:
            unknown_paths.append(path)

    return ScanResult(
        runs=tuple(sorted(runs, key=lambda run: str(run.source))),
        reports=tuple(reports),
        unknown_paths=tuple(unknown_paths),
        source_root=source_root,
        output_root=output_root,
    )
