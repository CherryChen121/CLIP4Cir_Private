from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Optional, Sequence, Union


VALID_STAGES = frozenset({"clip-finetune", "combiner"})


@dataclass(frozen=True)
class RunLayout:
    root: Path
    checkpoints: Path
    manifest: Path
    run_id: str


def slugify_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"empty slug for path component: {value!r}")
    return slug


def resolve_output_root(
    project_root: Path,
    requested: Optional[Union[str, Path]],
) -> Path:
    project_root = project_root.resolve()
    if requested is None:
        return project_root / "outputs"
    requested_path = Path(requested).expanduser()
    if requested_path.is_absolute():
        return requested_path.resolve()
    return (project_root / requested_path).resolve()


def build_run_id(started_at: datetime, pid: int) -> str:
    return f"{started_at:%Y%m%d-%H%M%S-%f}-p{pid}"


def create_run_layout(
    *,
    project_root: Path,
    output_root: Optional[Union[str, Path]],
    dataset: str,
    dataset_format: str,
    stage: str,
    model_name: str,
    dataset_root_requested: Optional[str] = None,
    dataset_root_resolved: Optional[str] = None,
    dataset_classification_evidence: Sequence[str] = (),
    started_at: Optional[datetime] = None,
    pid: Optional[int] = None,
) -> RunLayout:
    if stage not in VALID_STAGES:
        raise ValueError(
            f"invalid training stage {stage!r}; expected one of "
            f"{sorted(VALID_STAGES)}"
        )

    started_at = started_at or datetime.now()
    pid = os.getpid() if pid is None else pid
    dataset_slug = slugify_component(dataset)
    model_slug = slugify_component(model_name)
    run_id = build_run_id(started_at, pid)
    root = (
        resolve_output_root(project_root, output_root)
        / dataset_slug
        / stage
        / model_slug
        / run_id
    )
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    manifest = root / "run_manifest.json"
    temporary_manifest = root / ".run_manifest.json.tmp"
    payload = {
        "dataset": dataset_slug,
        "dataset_slug": dataset_slug,
        "dataset_format": slugify_component(dataset_format),
        "dataset_root_requested": dataset_root_requested,
        "dataset_root_resolved": dataset_root_resolved,
        "dataset_classification_evidence": list(
            dataset_classification_evidence
        ),
        "training_stage": stage,
        "model_name": model_name,
        "model_slug": model_slug,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "pid": pid,
        "checkpoint_dir": checkpoints.name,
    }
    temporary_manifest.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return RunLayout(
        root=root,
        checkpoints=checkpoints,
        manifest=manifest,
        run_id=run_id,
    )
