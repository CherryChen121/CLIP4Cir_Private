import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import stat
from typing import Iterable, Optional, Sequence, Tuple

from dataset_identity import (
    FASHIONIQ_TYPES,
    IDRID_TYPES,
    ROOT_NAME_TO_DATASET,
    UWF_TYPES,
)
from model_output_migration import sha256_file
from output_paths import VALID_STAGES


class ReclassificationBlockedError(RuntimeError):
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
