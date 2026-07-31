import csv
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_identity import (
    FASHIONIQ_TYPES,
    IDRID_TYPES,
    ROOT_NAME_TO_DATASET,
    UWF_TYPES,
)
from model_output_migration import SourceChangedError, sha256_file
from output_paths import VALID_STAGES, resolve_output_root


class ReclassificationBlockedError(RuntimeError):
    pass


class TransactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunClassification:
    dataset_slug: str
    dataset_format: str
    root_requested: Optional[str]
    root_resolved: Optional[str]
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: Path
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ReclassificationAction:
    source: Path
    staging: Path
    destination: Path
    classification: RunClassification
    files: Tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class ReclassificationPlan:
    output_root: Path
    actions: Tuple[ReclassificationAction, ...]
    unresolved: Tuple[str, ...]
    collisions: Tuple[str, ...]

    @property
    def dataset_counts(self) -> dict:
        return dict(
            sorted(
                Counter(
                    action.classification.dataset_slug
                    for action in self.actions
                ).items()
            )
        )


@dataclass(frozen=True)
class VerificationResult:
    run_counts: Dict[str, int]
    checkpoint_count: int
    retained_audit_files: int
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _classify_historical_categories(
    values: Optional[Sequence[str]],
) -> Optional[str]:
    if not values:
        return None
    normalized = frozenset(str(value).casefold() for value in values)
    if normalized == IDRID_TYPES:
        return "idrid"
    if normalized == UWF_TYPES:
        return "uwf"
    if normalized == frozenset({"internal"}):
        return "combined-fundus-cir"
    if normalized == FASHIONIQ_TYPES:
        return "fashioniq"
    return None


def _classify_historical_root(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return ROOT_NAME_TO_DATASET.get(Path(value).expanduser().name.casefold())


def _metric_header_dataset(header: str) -> Optional[str]:
    normalized = header.casefold()
    if normalized.startswith("idrid_recall_"):
        return "idrid"
    if normalized.startswith("internal_recall_"):
        return "combined-fundus-cir"
    for category in UWF_TYPES:
        if normalized.startswith(f"{category}_recall_"):
            return f"uwf:{category}"
    for category in FASHIONIQ_TYPES:
        if normalized.startswith(f"{category}_recall_"):
            return f"fashioniq:{category}"
    return None


def _classify_metric_headers(headers: Sequence[str]) -> Optional[str]:
    raw_evidence = {
        value
        for value in (_metric_header_dataset(header) for header in headers)
        if value is not None
    }
    candidates = {
        value
        for value in raw_evidence
        if value in {"idrid", "combined-fundus-cir"}
    }
    uwf_categories = {
        value.split(":", 1)[1]
        for value in raw_evidence
        if value.startswith("uwf:")
    }
    fashioniq_categories = {
        value.split(":", 1)[1]
        for value in raw_evidence
        if value.startswith("fashioniq:")
    }
    if uwf_categories == UWF_TYPES:
        candidates.add("uwf")
    elif uwf_categories:
        raise ReclassificationBlockedError(
            "incomplete UWF validation metric evidence: "
            + ",".join(sorted(uwf_categories))
        )
    if fashioniq_categories == FASHIONIQ_TYPES:
        candidates.add("fashioniq")
    elif fashioniq_categories:
        raise ReclassificationBlockedError(
            "incomplete FashionIQ validation metric evidence: "
            + ",".join(sorted(fashioniq_categories))
        )
    if len(candidates) > 1:
        raise ReclassificationBlockedError(
            "conflicting validation metric evidence: "
            + ",".join(sorted(candidates))
        )
    return next(iter(candidates), None)


def _matching_metric_headers(
    headers: Sequence[str], dataset_slug: str
) -> Iterable[str]:
    for header in headers:
        classified = _metric_header_dataset(header)
        if classified == dataset_slug or (
            classified is not None
            and classified.startswith(f"{dataset_slug}:")
        ):
            yield header


def classify_run(run_root: Path) -> RunClassification:
    hyperparameters_path = run_root / "training_hyperparameters.json"
    validation_path = run_root / "validation_metrics.csv"
    if not validation_path.is_file():
        raise ReclassificationBlockedError(
            f"missing validation_metrics.csv for {run_root}"
        )

    hyperparameters = (
        json.loads(hyperparameters_path.read_text(encoding="utf-8"))
        if hyperparameters_path.exists()
        else {}
    )
    with validation_path.open(newline="", encoding="utf-8-sig") as file:
        try:
            headers = next(csv.reader(file))
        except StopIteration as error:
            raise ReclassificationBlockedError(
                f"empty validation_metrics.csv for {run_root}"
            ) from error

    train_dress_types = hyperparameters.get("train_dress_types")
    category_evidence = _classify_historical_categories(train_dress_types)
    metric_evidence = _classify_metric_headers(headers)
    requested_root = hyperparameters.get("fashioniq_root")
    root_evidence = _classify_historical_root(requested_root)

    if (
        category_evidence == "combined-fundus-cir"
        and hyperparameters_path.exists()
        and root_evidence != "combined-fundus-cir"
    ):
        category_evidence = None

    candidates = {
        value
        for value in (category_evidence, metric_evidence, root_evidence)
        if value is not None
    }
    if len(candidates) > 1:
        raise ReclassificationBlockedError(
            f"conflicting evidence for {run_root}: "
            f"categories={category_evidence}, metrics={metric_evidence}, "
            f"root={root_evidence}"
        )
    if not candidates:
        raise ReclassificationBlockedError(
            f"unresolved evidence for {run_root}: "
            f"categories={category_evidence}, metrics={metric_evidence}, "
            f"root={root_evidence}"
        )

    dataset_slug = candidates.pop()
    resolved_root = (
        str(Path(requested_root).expanduser().resolve())
        if requested_root
        else None
    )
    evidence = []
    if train_dress_types and category_evidence == dataset_slug:
        evidence.append(
            "training_hyperparameters:train_dress_types="
            + ",".join(str(value) for value in train_dress_types)
        )
    evidence.extend(
        f"validation_metrics:{header}"
        for header in _matching_metric_headers(headers, dataset_slug)
    )
    if requested_root and root_evidence == dataset_slug:
        evidence.append(
            f"training_hyperparameters:fashioniq_root={requested_root}"
        )
    return RunClassification(
        dataset_slug=dataset_slug,
        dataset_format="fashioniq",
        root_requested=requested_root,
        root_resolved=resolved_root,
        evidence=tuple(evidence),
    )


def _require_directory(path: Path, description: str) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ReclassificationBlockedError(
            f"symlink is not allowed for {description}: {path}"
        )
    if not stat.S_ISDIR(mode):
        raise ReclassificationBlockedError(
            f"expected directory for {description}: {path}"
        )


def _discover_exact_run_roots(source_root: Path) -> Tuple[Path, ...]:
    _require_directory(source_root, "source root")
    runs = []
    for stage_root in sorted(source_root.iterdir()):
        _require_directory(stage_root, "training stage")
        if stage_root.name not in VALID_STAGES:
            raise ReclassificationBlockedError(
                f"unexpected training stage: {stage_root}"
            )
        for model_root in sorted(stage_root.iterdir()):
            _require_directory(model_root, "model")
            for run_root in sorted(model_root.iterdir()):
                _require_directory(run_root, "run")
                if not (run_root / "run_manifest.json").is_file():
                    raise ReclassificationBlockedError(
                        f"run is missing run_manifest.json: {run_root}"
                    )
                runs.append(run_root)
    return tuple(runs)


def _reject_symlinks_and_special_files(run_root: Path) -> None:
    for path in (run_root, *sorted(run_root.rglob("*"))):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ReclassificationBlockedError(
                f"symlink inside run is not allowed: {path}"
            )
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ReclassificationBlockedError(
                f"special file inside run is not allowed: {path}"
            )


def _snapshot_file(path: Path, run_root: Path) -> FileSnapshot:
    before = path.lstat()
    digest = sha256_file(path)
    after = path.lstat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ReclassificationBlockedError(
            f"source changed while hashing: {path}"
        )
    return FileSnapshot(
        relative_path=path.relative_to(run_root),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def build_reclassification_plan(output_root: Path) -> ReclassificationPlan:
    output_root = output_root.resolve()
    source_root = output_root / "fashioniq"
    staging_root = output_root / ".dataset-reclassify-staging"
    if not source_root.exists():
        return ReclassificationPlan(output_root, (), (), ())

    run_roots = _discover_exact_run_roots(source_root)
    actions = []
    unresolved = []
    collisions = []
    for source in run_roots:
        try:
            _reject_symlinks_and_special_files(source)
            if _device_id(source) != _device_id(output_root):
                raise ReclassificationBlockedError(
                    f"cross-filesystem move is not allowed: {source}"
                )
            classification = classify_run(source)
        except ReclassificationBlockedError as error:
            unresolved.append(str(error))
            continue

        stage, model_slug, run_id = source.relative_to(source_root).parts
        destination = (
            output_root
            / classification.dataset_slug
            / stage
            / model_slug
            / run_id
        )
        staging = (
            staging_root
            / classification.dataset_slug
            / stage
            / model_slug
            / run_id
        )
        if destination.exists() or staging.exists():
            collisions.append(str(destination))
            continue
        files = tuple(
            _snapshot_file(path, source)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        )
        actions.append(
            ReclassificationAction(
                source=source,
                staging=staging,
                destination=destination,
                classification=classification,
                files=files,
            )
        )

    if unresolved or collisions:
        raise ReclassificationBlockedError(
            "reclassification plan blocked; unresolved="
            + repr(unresolved)
            + "; collisions="
            + repr(collisions)
        )
    return ReclassificationPlan(
        output_root=output_root,
        actions=tuple(actions),
        unresolved=(),
        collisions=(),
    )


TRAINING_SCRIPTS = frozenset({"clip_fine_tune.py", "combiner_train.py"})


def _argv_option(argv: Sequence[str], name: str) -> Optional[str]:
    for index, argument in enumerate(argv):
        if argument == name:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        prefix = name + "="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


def find_output_writer_pids(
    project_root: Path,
    output_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> Tuple[int, ...]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    matches = []
    try:
        process_directories = sorted(
            (
                path
                for path in proc_root.iterdir()
                if path.name.isdigit() and path.is_dir()
            ),
            key=lambda path: int(path.name),
        )
    except (FileNotFoundError, PermissionError):
        return ()

    for process in process_directories:
        try:
            decoded = [
                os.fsdecode(item)
                for item in (process / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            cwd = (process / "cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue

        scripts = [
            Path(argument)
            for argument in decoded
            if Path(argument).name in TRAINING_SCRIPTS
        ]
        if not scripts:
            continue
        script = scripts[0]
        if not script.is_absolute():
            script = cwd / script
        try:
            resolved_script = script.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved_script.parent != project_root / "src":
            continue

        dataset = _argv_option(decoded, "--dataset")
        if dataset is not None and dataset.casefold() == "cirr":
            continue
        requested_output = _argv_option(decoded, "--output-root")
        effective_output = resolve_output_root(
            project_root, requested_output
        )
        if effective_output == output_root:
            matches.append(int(process.name))
    return tuple(matches)


def _verify_snapshot(root: Path, snapshot: FileSnapshot) -> None:
    path = root / snapshot.relative_path
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise SourceChangedError(f"source file is missing: {path}") from error
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise SourceChangedError(f"source type changed: {path}")
    if (
        current.st_size != snapshot.size
        or current.st_mtime_ns != snapshot.mtime_ns
    ):
        raise SourceChangedError(f"source metadata changed: {path}")
    if sha256_file(path) != snapshot.sha256:
        raise SourceChangedError(f"source hash changed: {path}")


def _verify_action_snapshots(
    actions: Sequence[ReclassificationAction],
    location: str,
    *,
    include_run_manifest: bool = True,
) -> None:
    for action in actions:
        root = getattr(action, location)
        for snapshot in action.files:
            if (
                not include_run_manifest
                and snapshot.relative_path == Path("run_manifest.json")
            ):
                continue
            _verify_snapshot(root, snapshot)


def _replace_file(source: Path, destination: Path) -> Path:
    return source.replace(destination)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.dataset-reclassify.tmp"
    )
    with temporary.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    _replace_file(temporary, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_bytes_atomic(
        path,
        (
            json.dumps(payload, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    )


def _journal_actions(plan: ReclassificationPlan) -> List[dict]:
    return [
        {
            "source": str(action.source),
            "staging": str(action.staging),
            "destination": str(action.destination),
            "dataset": action.classification.dataset_slug,
        }
        for action in plan.actions
    ]


def _copy_metadata_backups(
    plan: ReclassificationPlan, backup_root: Path
) -> Tuple[Tuple[Path, Path], ...]:
    backups = []
    run_backup_root = backup_root / "run-manifests"
    run_backup_root.mkdir(parents=True)
    for index, action in enumerate(plan.actions):
        target = action.destination / "run_manifest.json"
        backup = run_backup_root / f"{index:04d}.json"
        backup.write_bytes(target.read_bytes())
        backups.append((backup, target))

    for name in ("migration_manifest.csv", "migration_report.json"):
        target = plan.output_root / name
        if not target.is_file():
            raise ReclassificationBlockedError(
                f"required audit file is missing: {target}"
            )
        backup = backup_root / name
        backup.write_bytes(target.read_bytes())
        backups.append((backup, target))
    return tuple(backups)


def _restore_metadata(
    backups: Sequence[Tuple[Path, Path]],
) -> Tuple[str, ...]:
    errors = []
    for backup, target in reversed(tuple(backups)):
        try:
            _write_bytes_atomic(target, backup.read_bytes())
        except Exception as error:
            errors.append(f"{target}: {error}")
    return tuple(errors)


def _translate_audit_path(
    raw_path: str, actions: Sequence[ReclassificationAction]
) -> str:
    if not raw_path:
        return raw_path
    path = Path(raw_path)
    for action in actions:
        try:
            relative = path.relative_to(action.source)
        except ValueError:
            continue
        return str(action.destination / relative)
    return raw_path


def _updated_run_manifest(action: ReclassificationAction) -> dict:
    path = action.destination / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "dataset": action.classification.dataset_slug,
            "dataset_slug": action.classification.dataset_slug,
            "dataset_format": action.classification.dataset_format,
            "dataset_root_requested": action.classification.root_requested,
            "dataset_root_resolved": action.classification.root_resolved,
            "dataset_classification_evidence": list(
                action.classification.evidence
            ),
        }
    )
    return payload


def _write_updated_metadata(plan: ReclassificationPlan) -> None:
    for action in plan.actions:
        _write_json_atomic(
            action.destination / "run_manifest.json",
            _updated_run_manifest(action),
        )

    csv_path = plan.output_root / "migration_manifest.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ReclassificationBlockedError(
            f"migration manifest has no header: {csv_path}"
        )
    for row in rows:
        original_path = row.get("new_path", "")
        translated = _translate_audit_path(original_path, plan.actions)
        if translated != original_path:
            row["new_path"] = translated
            translated_path = Path(translated)
            for action in plan.actions:
                if action.destination in translated_path.parents:
                    row["dataset"] = action.classification.dataset_slug
                    break
        canonical = row.get("canonical", "")
        if canonical:
            row["canonical"] = _translate_audit_path(
                canonical, plan.actions
            )
    csv_temporary = csv_path.with_name(
        f".{csv_path.name}.dataset-reclassify.tmp"
    )
    with csv_temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    _replace_file(csv_temporary, csv_path)

    report_path = plan.output_root / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["actual_dataset_run_counts"] = plan.dataset_counts
    report["reclassification_state"] = "applied"
    _write_json_atomic(report_path, report)


def _remove_empty_ancestors(path: Path, stop: Path) -> None:
    current = path
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _cleanup_rollback_artifacts(
    staging_root: Path,
    backup_paths: Sequence[Tuple[Path, Path]],
) -> Tuple[str, ...]:
    errors = []
    exact_files = [backup for backup, _target in backup_paths]
    exact_files.extend(
        [
            staging_root / "transaction.json",
            staging_root / ".transaction.json.dataset-reclassify.tmp",
        ]
    )
    for path in exact_files:
        try:
            if path.exists():
                path.unlink()
        except OSError as error:
            errors.append(f"{path}: {error}")
    if staging_root.exists():
        for directory in sorted(
            (
                path
                for path in staging_root.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            staging_root.rmdir()
        except OSError as error:
            errors.append(f"{staging_root}: {error}")
    return tuple(errors)


def _rollback_actions(
    plan: ReclassificationPlan,
) -> Tuple[str, ...]:
    errors = []
    for action in reversed(plan.actions):
        try:
            if action.destination.exists():
                action.source.parent.mkdir(parents=True, exist_ok=True)
                action.destination.replace(action.source)
            elif action.staging.exists():
                action.source.parent.mkdir(parents=True, exist_ok=True)
                action.staging.replace(action.source)
            elif not action.source.exists():
                errors.append(
                    f"run is missing from source, staging, and destination: "
                    f"{action.source}"
                )
        except OSError as error:
            errors.append(f"{action.source}: {error}")
    for action in reversed(plan.actions):
        _remove_empty_ancestors(
            action.destination.parent, plan.output_root
        )
    return tuple(errors)


def apply_reclassification(
    plan: ReclassificationPlan,
    *,
    project_root: Optional[Path] = None,
) -> VerificationResult:
    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else plan.output_root.parent.resolve()
    )
    writers = find_output_writer_pids(
        project_root, plan.output_root
    )
    if writers:
        raise ReclassificationBlockedError(
            "active FashionIQ output writers detected: "
            + ",".join(str(pid) for pid in writers)
        )
    for audit_name in (
        "migration_manifest.csv",
        "migration_report.json",
    ):
        audit_path = plan.output_root / audit_name
        if not audit_path.is_file():
            raise ReclassificationBlockedError(
                f"required audit file is missing: {audit_path}"
            )
    _verify_action_snapshots(plan.actions, "source")

    staging_root = plan.output_root / ".dataset-reclassify-staging"
    if staging_root.exists():
        raise ReclassificationBlockedError(
            f"staging path already exists: {staging_root}"
        )

    backups: Tuple[Tuple[Path, Path], ...] = ()
    journal_path = staging_root / "transaction.json"
    try:
        staging_root.mkdir()
        _write_json_atomic(
            journal_path,
            {
                "state": "prepared",
                "actions": _journal_actions(plan),
            },
        )
        for action in plan.actions:
            action.staging.parent.mkdir(parents=True, exist_ok=True)
            action.source.replace(action.staging)
        _verify_action_snapshots(plan.actions, "staging")
        _write_json_atomic(
            journal_path,
            {
                "state": "staged",
                "actions": _journal_actions(plan),
            },
        )

        for action in plan.actions:
            action.destination.parent.mkdir(parents=True, exist_ok=True)
            action.staging.replace(action.destination)
        _verify_action_snapshots(plan.actions, "destination")

        backup_root = staging_root / "metadata-backup"
        backup_root.mkdir()
        backups = _copy_metadata_backups(plan, backup_root)
        _write_updated_metadata(plan)
        result = verify_reclassification(
            plan.output_root, expected_plan=plan, finalized=False
        )
        if not result.ok:
            raise TransactionError(
                "applied-state verification failed: "
                + "; ".join(result.errors)
            )
        _write_json_atomic(
            journal_path,
            {
                "state": "applied",
                "actions": _journal_actions(plan),
            },
        )
        return result
    except Exception as error:
        restore_errors = _restore_metadata(backups)
        rollback_errors = _rollback_actions(plan)
        cleanup_errors = _cleanup_rollback_artifacts(
            staging_root, backups
        )
        all_errors = restore_errors + rollback_errors + cleanup_errors
        if all_errors:
            raise TransactionError(
                "apply failed and rollback was incomplete; preserve "
                f"{staging_root}: " + "; ".join(all_errors)
            ) from error
        raise TransactionError(
            "apply failed and was rolled back"
        ) from error


def _manifest_run_paths(output_root: Path) -> Tuple[Path, ...]:
    paths = []
    for dataset_slug in (
        "idrid",
        "uwf",
        "combined-fundus-cir",
        "fashioniq",
    ):
        dataset_root = output_root / dataset_slug
        if dataset_root.exists():
            paths.extend(
                dataset_root.glob("*/*/*/run_manifest.json")
            )
    return tuple(sorted(paths))


def _verify_run_manifests(
    output_root: Path, errors: List[str]
) -> Dict[str, int]:
    counts = Counter()
    for manifest_path in _manifest_run_paths(output_root):
        relative = manifest_path.relative_to(output_root)
        dataset_slug, stage, model_slug, run_id, _name = relative.parts
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid run manifest {manifest_path}: {error}")
            continue
        expected = {
            "dataset": dataset_slug,
            "dataset_slug": dataset_slug,
            "training_stage": stage,
            "model_slug": model_slug,
            "run_id": run_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                errors.append(
                    f"run manifest mismatch for {manifest_path}: "
                    f"{key}={payload.get(key)!r}, expected {value!r}"
                )
        if payload.get("dataset_format") not in {"fashioniq", "cirr"}:
            errors.append(
                f"run manifest missing dataset_format: {manifest_path}"
            )
        if not isinstance(
            payload.get("dataset_classification_evidence"), list
        ):
            errors.append(
                "run manifest missing dataset_classification_evidence: "
                f"{manifest_path}"
            )
        counts[dataset_slug] += 1
    return dict(sorted(counts.items()))


def _verify_audit_rows(
    output_root: Path, errors: List[str]
) -> int:
    manifest_path = output_root / "migration_manifest.csv"
    if not manifest_path.is_file():
        errors.append(f"migration manifest is missing: {manifest_path}")
        return 0
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    checked = 0
    for row in rows:
        status_value = row.get("status", "")
        destination_text = row.get("new_path", "")
        if status_value in {"moved", "deduplicated"}:
            destination = Path(destination_text)
            if not destination.is_file():
                errors.append(
                    f"migrated file is missing: {destination}"
                )
                continue
            expected_size = int(row["size"])
            if destination.stat().st_size != expected_size:
                errors.append(f"size mismatch for {destination}")
                continue
            if sha256_file(destination) != row["sha256"]:
                errors.append(f"sha256 mismatch for {destination}")
                continue
            if status_value == "deduplicated":
                canonical = Path(row["canonical"])
                if not canonical.is_file():
                    errors.append(
                        f"canonical checkpoint is missing: {canonical}"
                    )
                    continue
                canonical_stat = canonical.stat()
                destination_stat = destination.stat()
                if (
                    canonical_stat.st_dev != destination_stat.st_dev
                    or canonical_stat.st_ino != destination_stat.st_ino
                ):
                    errors.append(
                        f"duplicate inode mismatch for {destination}"
                    )
                    continue
            checked += 1
        elif status_value == "deleted-approved-report":
            if destination_text and Path(destination_text).exists():
                errors.append(
                    "approved deleted report still exists: "
                    f"{destination_text}"
                )
    return checked


def verify_reclassification(
    output_root: Path,
    *,
    expected_plan: Optional[ReclassificationPlan] = None,
    finalized: bool = False,
) -> VerificationResult:
    output_root = Path(output_root).resolve()
    errors: List[str] = []
    if expected_plan is not None:
        for action in expected_plan.actions:
            if not action.destination.is_dir():
                errors.append(
                    f"destination run is missing: {action.destination}"
                )
                continue
            for snapshot in action.files:
                if snapshot.relative_path == Path("run_manifest.json"):
                    continue
                path = action.destination / snapshot.relative_path
                if not path.is_file():
                    errors.append(f"destination file is missing: {path}")
                    continue
                if path.stat().st_size != snapshot.size:
                    errors.append(f"size mismatch for {path}")
                    continue
                if sha256_file(path) != snapshot.sha256:
                    errors.append(f"sha256 mismatch for {path}")

    run_counts = _verify_run_manifests(output_root, errors)
    retained_audit_files = _verify_audit_rows(output_root, errors)
    report_path = output_root / "migration_report.json"
    if not report_path.is_file():
        errors.append(f"migration report is missing: {report_path}")
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"invalid migration report: {error}")
        else:
            if report.get("actual_dataset_run_counts") != run_counts:
                errors.append(
                    "migration report dataset counts do not match runs"
                )

    if finalized:
        for path in (
            output_root / "reports",
            output_root / "fashioniq",
            output_root / ".dataset-reclassify-staging",
        ):
            if path.exists():
                errors.append(
                    f"finalized path should be absent: {path}"
                )
    checkpoint_count = sum(
        1
        for dataset_slug in (
            "idrid",
            "uwf",
            "combined-fundus-cir",
            "fashioniq",
        )
        for _path in (output_root / dataset_slug).glob(
            "*/*/*/checkpoints/*.pt"
        )
    )
    return VerificationResult(
        run_counts=run_counts,
        checkpoint_count=checkpoint_count,
        retained_audit_files=retained_audit_files,
        errors=tuple(errors),
    )


APPROVED_REPORT_STATUS = "deleted-approved-report"
APPROVED_REPORT_REASON = "user-approved-obsolete-legacy-summary"


def _read_audit_rows(
    output_root: Path,
) -> Tuple[List[str], List[dict]]:
    manifest_path = output_root / "migration_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ReclassificationBlockedError(
            f"migration manifest has no header: {manifest_path}"
        )
    return list(fieldnames), rows


def _approved_report_rows(
    output_root: Path, rows: Sequence[dict]
) -> Tuple[Tuple[dict, Path], ...]:
    legacy_root = (output_root / "reports" / "legacy").resolve()
    selected = []
    for row in rows:
        raw_path = row.get("new_path", "")
        if (
            row.get("status") != "moved"
            or not raw_path.lower().endswith(".xlsx")
        ):
            continue
        path = Path(raw_path)
        try:
            path.resolve().relative_to(legacy_root)
        except ValueError:
            continue
        selected.append((row, path))
    if len(selected) != 5:
        raise ReclassificationBlockedError(
            "expected exactly 5 approved legacy reports, found "
            f"{len(selected)}"
        )
    return tuple(selected)


def _verify_approved_report_set(
    output_root: Path,
    selected: Sequence[Tuple[dict, Path]],
) -> None:
    legacy_root = output_root / "reports" / "legacy"
    expected = {path.resolve() for _row, path in selected}
    actual = set()
    if legacy_root.exists():
        for path in legacy_root.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ReclassificationBlockedError(
                    f"symlink in legacy reports is not allowed: {path}"
                )
            if stat.S_ISREG(mode):
                actual.add(path.resolve())
            elif not stat.S_ISDIR(mode):
                raise ReclassificationBlockedError(
                    f"special file in legacy reports is not allowed: {path}"
                )
    unexpected = actual - expected
    missing = expected - actual
    if unexpected:
        raise ReclassificationBlockedError(
            "unplanned report files block finalize: "
            + ", ".join(str(path) for path in sorted(unexpected))
        )
    if missing:
        raise ReclassificationBlockedError(
            "approved report files are missing: "
            + ", ".join(str(path) for path in sorted(missing))
        )
    for row, path in selected:
        if path.stat().st_size != int(row["size"]):
            raise ReclassificationBlockedError(
                f"approved report size mismatch: {path}"
            )
        if sha256_file(path) != row["sha256"]:
            raise ReclassificationBlockedError(
                f"approved report sha256 mismatch: {path}"
            )


def _assert_tree_contains_only_directories(path: Path) -> None:
    if not path.exists():
        return
    for descendant in path.rglob("*"):
        mode = descendant.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ReclassificationBlockedError(
                f"cleanup tree is not empty: {descendant}"
            )


def _write_finalized_audits(
    output_root: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict],
    deleted_count: int,
) -> None:
    csv_path = output_root / "migration_manifest.csv"
    temporary = csv_path.with_name(
        f".{csv_path.name}.dataset-reclassify.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    _replace_file(temporary, csv_path)

    report_path = output_root / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status_counts = dict(report.get("status_counts", {}))
    moved_count = int(status_counts.get("moved", 0)) - deleted_count
    if moved_count < 0:
        raise ReclassificationBlockedError(
            "migration report moved count is smaller than report deletions"
        )
    if moved_count:
        status_counts["moved"] = moved_count
    else:
        status_counts.pop("moved", None)
    status_counts[APPROVED_REPORT_STATUS] = (
        int(status_counts.get(APPROVED_REPORT_STATUS, 0))
        + deleted_count
    )
    report["status_counts"] = dict(sorted(status_counts.items()))
    report["reclassification_state"] = "finalized"
    _write_json_atomic(report_path, report)


def _restore_finalize_state(
    audit_backups: Sequence[Tuple[Path, Path]],
    quarantined: Sequence[Tuple[Path, Path]],
) -> Tuple[str, ...]:
    errors = list(_restore_metadata(audit_backups))
    for original, quarantine in reversed(tuple(quarantined)):
        try:
            if quarantine.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                quarantine.replace(original)
        except OSError as error:
            errors.append(f"{original}: {error}")
    return tuple(errors)


def _remove_empty_tree(path: Path) -> None:
    if not path.exists():
        return
    for directory in sorted(
        (
            descendant
            for descendant in path.rglob("*")
            if descendant.is_dir() and not descendant.is_symlink()
        ),
        key=lambda descendant: len(descendant.parts),
        reverse=True,
    ):
        directory.rmdir()
    path.rmdir()


def _transaction_artifact_files(
    staging_root: Path,
    quarantined: Sequence[Tuple[Path, Path]],
    finalize_backups: Sequence[Tuple[Path, Path]],
) -> Tuple[Path, ...]:
    journal_path = staging_root / "transaction.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    action_count = len(journal.get("actions", ()))
    expected = {
        journal_path,
        staging_root / "metadata-backup/migration_manifest.csv",
        staging_root / "metadata-backup/migration_report.json",
    }
    expected.update(
        staging_root
        / "metadata-backup/run-manifests"
        / f"{index:04d}.json"
        for index in range(action_count)
    )
    expected.update(
        quarantine for _original, quarantine in quarantined
    )
    expected.update(backup for backup, _target in finalize_backups)
    actual = {
        path
        for path in staging_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        unexpected = actual - expected
        missing = expected - actual
        raise ReclassificationBlockedError(
            "unexpected transaction artifacts; unexpected="
            + repr(sorted(str(path) for path in unexpected))
            + "; missing="
            + repr(sorted(str(path) for path in missing))
        )
    for path in staging_root.rglob("*"):
        if path.is_symlink():
            raise ReclassificationBlockedError(
                f"symlink in transaction staging is not allowed: {path}"
            )
    return tuple(sorted(expected))


def finalize_reclassification(
    output_root: Path,
    *,
    expected_plan: Optional[ReclassificationPlan] = None,
) -> VerificationResult:
    output_root = Path(output_root).resolve()
    applied_verification = verify_reclassification(
        output_root,
        expected_plan=expected_plan,
        finalized=False,
    )
    if not applied_verification.ok:
        raise ReclassificationBlockedError(
            "applied-state verification failed: "
            + "; ".join(applied_verification.errors)
        )

    fieldnames, rows = _read_audit_rows(output_root)
    selected = _approved_report_rows(output_root, rows)
    _verify_approved_report_set(output_root, selected)
    _assert_tree_contains_only_directories(output_root / "fashioniq")

    staging_root = output_root / ".dataset-reclassify-staging"
    journal_path = staging_root / "transaction.json"
    if not journal_path.is_file():
        raise ReclassificationBlockedError(
            f"applied transaction journal is missing: {journal_path}"
        )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("state") != "applied":
        raise ReclassificationBlockedError(
            f"transaction is not in applied state: {journal_path}"
        )

    finalize_backup_root = staging_root / "finalize-backup"
    if finalize_backup_root.exists():
        raise ReclassificationBlockedError(
            f"finalize backup already exists: {finalize_backup_root}"
        )
    finalize_backup_root.mkdir()
    audit_backups = []
    for name in ("migration_manifest.csv", "migration_report.json"):
        target = output_root / name
        backup = finalize_backup_root / name
        backup.write_bytes(target.read_bytes())
        audit_backups.append((backup, target))

    legacy_root = output_root / "reports" / "legacy"
    quarantine_root = staging_root / "approved-report-deletions"
    quarantined = []
    try:
        for row, original in selected:
            relative = original.resolve().relative_to(
                legacy_root.resolve()
            )
            quarantine = quarantine_root / relative
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            original.replace(quarantine)
            quarantined.append((original, quarantine))
            row["status"] = APPROVED_REPORT_STATUS
            row["reason"] = APPROVED_REPORT_REASON

        _write_finalized_audits(
            output_root, fieldnames, rows, len(selected)
        )
        intermediate = verify_reclassification(
            output_root,
            expected_plan=expected_plan,
            finalized=False,
        )
        if not intermediate.ok:
            raise TransactionError(
                "finalized audit verification failed: "
                + "; ".join(intermediate.errors)
            )
        exact_artifacts = _transaction_artifact_files(
            staging_root, quarantined, audit_backups
        )
    except Exception as error:
        restore_errors = _restore_finalize_state(
            audit_backups, quarantined
        )
        try:
            _remove_empty_tree(finalize_backup_root)
        except OSError:
            pass
        try:
            _remove_empty_tree(quarantine_root)
        except OSError:
            pass
        if restore_errors:
            raise TransactionError(
                "finalize rollback was incomplete: "
                + "; ".join(restore_errors)
            ) from error
        raise TransactionError("finalize rolled back") from error

    for artifact in exact_artifacts:
        artifact.unlink()
    _remove_empty_tree(staging_root)
    _remove_empty_tree(output_root / "reports")
    _remove_empty_tree(output_root / "fashioniq")

    result = verify_reclassification(
        output_root,
        expected_plan=expected_plan,
        finalized=True,
    )
    if not result.ok:
        raise TransactionError(
            "final verification failed: " + "; ".join(result.errors)
        )
    return result
