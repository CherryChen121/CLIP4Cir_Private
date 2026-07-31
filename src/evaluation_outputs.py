import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Union,
)

from dataset_identity import DatasetIdentity
from output_paths import build_run_id, resolve_output_root, slugify_component


@dataclass(frozen=True)
class EvaluationLayout:
    root: Path
    manifest: Path
    metrics_json: Path
    metrics_csv: Path
    log: Path
    run_id: str


def validate_metrics_csv_filename(value: str) -> str:
    candidate = Path(value)
    if not value or candidate.is_absolute() or candidate.name != value:
        raise ValueError(
            "--output-csv must be a filename inside the evaluation run"
        )
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(json_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def create_evaluation_layout(
    *,
    project_root: Path,
    output_root: Optional[Union[str, Path]],
    identity: DatasetIdentity,
    evaluation_script: str,
    evaluation_name: str,
    model_name: str,
    split: str,
    categories: Sequence[str],
    cli_args: Mapping[str, Any],
    input_paths: Mapping[str, Any],
    metrics_csv_filename: str = "evaluation_metrics.csv",
    started_at: Optional[datetime] = None,
    pid: Optional[int] = None,
) -> EvaluationLayout:
    metrics_csv_filename = validate_metrics_csv_filename(
        metrics_csv_filename
    )
    started_at = started_at or datetime.now()
    pid = os.getpid() if pid is None else pid
    dataset_slug = slugify_component(identity.dataset_slug)
    model_slug = slugify_component(model_name)
    run_id = build_run_id(started_at, pid)
    root = (
        resolve_output_root(project_root, output_root)
        / dataset_slug
        / "evaluation"
        / model_slug
        / run_id
    )
    root.mkdir(parents=True, exist_ok=False)

    layout = EvaluationLayout(
        root=root,
        manifest=root / "evaluation_manifest.json",
        metrics_json=root / "evaluation_metrics.json",
        metrics_csv=root / metrics_csv_filename,
        log=root / "evaluation.log",
        run_id=run_id,
    )
    layout.log.touch(exist_ok=False)
    payload = {
        "schema_version": 1,
        "status": "running",
        "dataset": dataset_slug,
        "dataset_slug": dataset_slug,
        "dataset_format": identity.dataset_format,
        "dataset_root_requested": identity.root_requested,
        "dataset_root_resolved": identity.root_resolved,
        "dataset_classification_evidence": list(
            identity.classification_evidence
        ),
        "evaluation_script": evaluation_script,
        "evaluation_name": evaluation_name,
        "model_name": model_name,
        "model_slug": model_slug,
        "split": split,
        "categories": list(categories),
        "cli_args": json_safe(cli_args),
        "input_paths": json_safe(input_paths),
        "run_id": run_id,
        "pid": pid,
        "started_at": started_at.isoformat(),
        "log_file": layout.log.name,
        "metric_files": {
            "json": layout.metrics_json.name,
            "csv": layout.metrics_csv.name,
        },
    }
    _atomic_write_json(layout.manifest, payload)
    return layout


class _Tee:
    def __init__(self, original: TextIO, log_handle: TextIO):
        self._original = original
        self._log_handle = log_handle

    def write(self, value: str) -> int:
        result = self._original.write(value)
        self._log_handle.write(value)
        return result

    def flush(self) -> None:
        self._original.flush()
        self._log_handle.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@contextmanager
def tee_evaluation_output(log_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_handle:
        sys.stdout = _Tee(original_stdout, log_handle)
        sys.stderr = _Tee(original_stderr, log_handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def publish_evaluation_metrics(
    layout: EvaluationLayout,
    document: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary_json = layout.metrics_json.with_name(
        f".{layout.metrics_json.name}.tmp"
    )
    temporary_csv = layout.metrics_csv.with_name(
        f".{layout.metrics_csv.name}.tmp"
    )
    temporary_paths = (temporary_json, temporary_csv)
    final_paths = (layout.metrics_json, layout.metrics_csv)

    try:
        temporary_json.write_text(
            json.dumps(json_safe(document), sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        with temporary_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(json_safe(row))
        temporary_json.replace(layout.metrics_json)
        temporary_csv.replace(layout.metrics_csv)
    except BaseException:
        for path in temporary_paths + final_paths:
            path.unlink(missing_ok=True)
        raise


def discard_evaluation_metrics(layout: EvaluationLayout) -> None:
    layout.metrics_json.unlink(missing_ok=True)
    layout.metrics_csv.unlink(missing_ok=True)


def finalize_evaluation(
    layout: EvaluationLayout,
    status: str,
    *,
    error: Optional[BaseException] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    if status not in {"succeeded", "failed"}:
        raise ValueError(
            "evaluation status must be either 'succeeded' or 'failed'"
        )

    payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["completed_at"] = (
        completed_at or datetime.now()
    ).isoformat()
    if status == "failed" and error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    else:
        payload.pop("error", None)
    _atomic_write_json(layout.manifest, payload)
