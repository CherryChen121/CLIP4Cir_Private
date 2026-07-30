from collections import Counter
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Callable, Dict, Optional, Tuple
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


class SourceChangedError(RuntimeError):
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


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class MigrationAction:
    source: Path
    destination: Optional[Path]
    kind: str
    dataset: str
    stage: str
    model_slug: str
    run_id: str
    status: str
    size: int
    mtime_ns: int
    sha256: str
    duplicate_group: str
    canonical: str
    reason: str


@dataclass(frozen=True)
class MigrationPlan:
    scan: ScanResult
    actions: Tuple[MigrationAction, ...]
    duplicate_groups: Dict[str, Tuple[Path, ...]]
    logical_bytes: int
    physical_bytes_before: int

    @property
    def has_blockers(self) -> bool:
        blocking_statuses = {"skipped-active", "unresolved", "error"}
        return any(
            action.status in blocking_statuses for action in self.actions
        )


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SourceChangedError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def _snapshot(path: Path) -> FileSnapshot:
    stat_result = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        stat_result.st_size != after.st_size
        or stat_result.st_mtime_ns != after.st_mtime_ns
    ):
        raise SourceChangedError(f"source changed while snapshotting: {path}")
    return FileSnapshot(
        path=path,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def _run_file_destination(run: LegacyRun, path: Path) -> Path:
    relative = path.relative_to(run.source)
    if relative.parts and relative.parts[0] == "saved_models":
        relative = Path("checkpoints", *relative.parts[1:])
    return run.destination / relative


def _action_for_snapshot(
    run: LegacyRun,
    snapshot: FileSnapshot,
) -> MigrationAction:
    destination = _run_file_destination(run, snapshot.path)
    kind = (
        "checkpoint"
        if snapshot.path.suffix.lower() in CHECKPOINT_SUFFIXES
        else "file"
    )
    status = "error" if destination.exists() else "planned-move"
    reason = "destination-exists" if destination.exists() else ""
    return MigrationAction(
        source=snapshot.path,
        destination=destination,
        kind=kind,
        dataset=run.dataset_slug,
        stage=run.stage,
        model_slug=run.model_slug,
        run_id=run.run_id,
        status=status,
        size=snapshot.size,
        mtime_ns=snapshot.mtime_ns,
        sha256=snapshot.sha256,
        duplicate_group="",
        canonical="",
        reason=reason,
    )


def _tree_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def build_migration_plan(scan: ScanResult) -> MigrationPlan:
    actions = []
    logical_bytes = 0
    physical_bytes_before = 0

    for run in scan.runs:
        if run.classification == "valid":
            for path in sorted(
                item for item in run.source.rglob("*") if item.is_file()
            ):
                snapshot = _snapshot(path)
                action = _action_for_snapshot(run, snapshot)
                actions.append(action)
                logical_bytes += snapshot.size
                physical_bytes_before += path.stat().st_blocks * 512
        elif run.classification == "failed":
            size = _tree_size(run.source)
            actions.append(
                MigrationAction(
                    source=run.source,
                    destination=None,
                    kind="failed-run",
                    dataset=run.dataset_slug,
                    stage=run.stage,
                    model_slug=run.model_slug,
                    run_id=run.run_id,
                    status="planned-delete-failed",
                    size=size,
                    mtime_ns=run.newest_mtime_ns,
                    sha256="",
                    duplicate_group="",
                    canonical="",
                    reason=";".join(run.reasons),
                )
            )
        else:
            status = (
                "skipped-active"
                if run.classification == "active"
                else "unresolved"
            )
            actions.append(
                MigrationAction(
                    source=run.source,
                    destination=run.destination,
                    kind="run",
                    dataset=run.dataset_slug,
                    stage=run.stage,
                    model_slug=run.model_slug,
                    run_id=run.run_id,
                    status=status,
                    size=_tree_size(run.source),
                    mtime_ns=run.newest_mtime_ns,
                    sha256="",
                    duplicate_group="",
                    canonical="",
                    reason=";".join(run.reasons),
                )
            )

    for report in scan.reports:
        snapshot = _snapshot(report)
        destination = scan.output_root / "reports" / "legacy" / report.name
        status = "error" if destination.exists() else "planned-move"
        actions.append(
            MigrationAction(
                source=report,
                destination=destination,
                kind="report",
                dataset="",
                stage="",
                model_slug="",
                run_id="",
                status=status,
                size=snapshot.size,
                mtime_ns=snapshot.mtime_ns,
                sha256=snapshot.sha256,
                duplicate_group="",
                canonical="",
                reason="destination-exists" if destination.exists() else "",
            )
        )
        logical_bytes += snapshot.size
        physical_bytes_before += report.stat().st_blocks * 512

    for unknown in scan.unknown_paths:
        stat_result = unknown.stat()
        actions.append(
            MigrationAction(
                source=unknown,
                destination=None,
                kind="unknown",
                dataset="",
                stage="",
                model_slug="",
                run_id="",
                status="unresolved",
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                sha256="",
                duplicate_group="",
                canonical="",
                reason="unknown-path",
            )
        )

    destination_counts = Counter(
        str(action.destination)
        for action in actions
        if action.destination is not None
    )
    actions = [
        replace(
            action,
            status="error",
            reason="destination-collision",
        )
        if (
            action.destination is not None
            and destination_counts[str(action.destination)] > 1
        )
        else action
        for action in actions
    ]

    duplicate_candidates = {}
    for index, action in enumerate(actions):
        if action.kind != "checkpoint" or action.status != "planned-move":
            continue
        duplicate_candidates.setdefault(
            (action.size, action.sha256),
            [],
        ).append((index, action))

    duplicate_groups = {}
    for (_, digest), members in sorted(duplicate_candidates.items()):
        if len(members) < 2:
            continue
        sorted_members = sorted(
            members,
            key=lambda member: str(member[1].destination),
        )
        group_id = digest
        canonical = sorted_members[0][1].destination
        group_paths = tuple(
            member[1].destination for member in sorted_members
            if member[1].destination is not None
        )
        duplicate_groups[group_id] = group_paths
        for position, (index, action) in enumerate(sorted_members):
            actions[index] = replace(
                action,
                status=(
                    "planned-move"
                    if position == 0
                    else "planned-deduplicate"
                ),
                duplicate_group=group_id,
                canonical=str(canonical),
            )

    return MigrationPlan(
        scan=scan,
        actions=tuple(
            sorted(actions, key=lambda action: str(action.source))
        ),
        duplicate_groups=duplicate_groups,
        logical_bytes=logical_bytes,
        physical_bytes_before=physical_bytes_before,
    )


CSV_FIELDS = (
    "old_path",
    "new_path",
    "dataset",
    "stage",
    "model_slug",
    "run_id",
    "status",
    "size",
    "sha256",
    "duplicate_group",
    "canonical",
    "reason",
)


def migration_report_payload(plan: MigrationPlan) -> dict:
    status_counts = Counter(action.status for action in plan.actions)
    duplicate_files = sum(
        len(paths) - 1 for paths in plan.duplicate_groups.values()
    )
    reclaimable_bytes = sum(
        action.size
        for action in plan.actions
        if action.status == "planned-deduplicate"
    )
    return {
        "total_runs": len(plan.scan.runs),
        "valid_runs": sum(
            run.classification == "valid" for run in plan.scan.runs
        ),
        "failed_runs": sum(
            run.classification == "failed" for run in plan.scan.runs
        ),
        "active_runs": sum(
            run.classification == "active" for run in plan.scan.runs
        ),
        "unresolved_runs": sum(
            run.classification == "unresolved" for run in plan.scan.runs
        ),
        "unknown_paths": len(plan.scan.unknown_paths),
        "total_files": sum(
            action.kind in {"file", "checkpoint", "report"}
            for action in plan.actions
        ),
        "logical_bytes": plan.logical_bytes,
        "physical_bytes_before": plan.physical_bytes_before,
        "duplicate_groups": len(plan.duplicate_groups),
        "duplicate_files": duplicate_files,
        "reclaimable_bytes": reclaimable_bytes,
        "collisions": sum(
            action.reason in {"destination-exists", "destination-collision"}
            for action in plan.actions
        ),
        "errors": status_counts.get("error", 0),
        "status_counts": dict(sorted(status_counts.items())),
    }


def write_migration_reports(
    plan: MigrationPlan,
    output_root: Path,
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "migration_manifest.csv"
    json_path = output_root / "migration_report.json"
    csv_temporary = output_root / ".migration_manifest.csv.tmp"
    json_temporary = output_root / ".migration_report.json.tmp"

    with csv_temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for action in plan.actions:
            writer.writerow(
                {
                    "old_path": str(action.source),
                    "new_path": (
                        str(action.destination)
                        if action.destination is not None
                        else ""
                    ),
                    "dataset": action.dataset,
                    "stage": action.stage,
                    "model_slug": action.model_slug,
                    "run_id": action.run_id,
                    "status": action.status,
                    "size": action.size,
                    "sha256": action.sha256,
                    "duplicate_group": action.duplicate_group,
                    "canonical": action.canonical,
                    "reason": action.reason,
                }
            )
    json_temporary.write_text(
        json.dumps(
            migration_report_payload(plan),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_temporary.replace(csv_path)
    json_temporary.replace(json_path)
    return csv_path, json_path
