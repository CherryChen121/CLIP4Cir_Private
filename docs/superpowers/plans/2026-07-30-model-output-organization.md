# Model Output Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route all future training runs into a safe `outputs/` hierarchy, migrate and physically deduplicate valid legacy runs, remove strictly failed runs, and delete the old `models/` path only after complete verification.

**Architecture:** A dependency-light `output_paths` module owns run naming and directory creation for both training entry points. A separate `model_output_migration` module scans and snapshots legacy data, plans deterministic destinations and duplicate groups, then performs same-filesystem staged moves, hard-link deduplication, reporting, verification, and final empty-directory removal through a thin CLI.

**Tech Stack:** Python 3.9, Python standard library, PyTorch, argparse, pytest, Linux `/proc`, POSIX rename and hard links.

## Global Constraints

- Default future output root is `<project>/outputs`; `clip_fine_tune.py` and `combiner_train.py` must never construct `<project>/models`.
- Layout is `outputs/<dataset-slug>/<training-stage>/<model-slug>/<YYYYMMDD-HHMMSS-microseconds-pPID>/`.
- `training-stage` is exactly `clip-finetune` or `combiner`.
- Slugs are lowercase; each non-alphanumeric run becomes one `-`; leading and trailing `-` are removed.
- Checkpoints live in `checkpoints/`, not `saved_models/`.
- Only equal size plus equal SHA-256 constitutes a duplicate checkpoint.
- Equal-size checkpoints with different SHA-256 values remain separate.
- Duplicate historical checkpoints retain every logical path but share one inode through hard links.
- A failed run must simultaneously have no nonempty checkpoint, no nonempty metrics CSV, no live writer, and no modification within 24 hours.
- Unknown files, active writers, target collisions, cross-filesystem destinations, changed source snapshots, or verification failures block finalization.
- `models/` is removed only with guarded, bottom-up `rmdir`; recursive forced removal is forbidden.
- Preserve unrelated worktree changes in `命令.sh`, `IDRiD平均召回率汇总.xlsx`, and `Related_Work_组合示例查询与医学跨模态检索.md`.
- Use `PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest` for tests.

---

## File Map

- Create `src/output_paths.py`: slugging, output-root resolution, run IDs, run directory creation, and run manifest writing.
- Modify `src/clip_fine_tune.py`: use `output_paths`, expose `--output-root`, and write hyperparameters for both datasets.
- Modify `src/combiner_train.py`: use `output_paths` and expose `--output-root`.
- Modify `src/utils.py`: save weights below `checkpoints/`.
- Create `src/model_output_migration.py`: legacy discovery, strict classification, hashing, planning, transactional apply, deduplication, reports, verification, and source finalization.
- Create `scripts/organize_model_outputs.py`: CLI modes for dry run, apply, verify, and finalize.
- Create `tests/test_output_paths.py`: path component unit tests.
- Create `tests/test_training_output_paths.py`: checkpoint directory and training CLI integration tests.
- Create `tests/test_model_output_migration.py`: scanner, planner, apply, deduplication, safety, and finalization tests.
- Modify `.gitignore`: ignore `outputs/` and stop silently accepting a recreated legacy `models/`.
- Modify `README.md`: document the new layout, CLI override, and checkpoint examples.

---

### Task 1: Central Output Path Component

**Files:**
- Create: `src/output_paths.py`
- Create: `tests/test_output_paths.py`

**Interfaces:**
- Produces: `RunLayout(root: Path, checkpoints: Path, manifest: Path, run_id: str)`
- Produces: `slugify_component(value: str) -> str`
- Produces: `resolve_output_root(project_root: Path, requested: Optional[Union[str, Path]]) -> Path`
- Produces: `build_run_id(started_at: datetime, pid: int) -> str`
- Produces: `create_run_layout(*, project_root: Path, output_root: Optional[Union[str, Path]], dataset: str, stage: str, model_name: str, started_at: Optional[datetime] = None, pid: Optional[int] = None) -> RunLayout`

- [ ] **Step 1: Write failing slug, root, run ID, manifest, and collision tests**

Create `tests/test_output_paths.py` with concrete cases:

```python
import json
from datetime import datetime
from pathlib import Path

import pytest

from output_paths import (
    build_run_id,
    create_run_layout,
    resolve_output_root,
    slugify_component,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FashionIQ", "fashioniq"),
        ("ViT-B/32", "vit-b-32"),
        ("  BLIP  ITM_Base  ", "blip-itm-base"),
        ("RN50x4", "rn50x4"),
    ],
)
def test_slugify_component(raw, expected):
    assert slugify_component(raw) == expected


def test_slugify_component_rejects_empty_result():
    with pytest.raises(ValueError, match="empty slug"):
        slugify_component("///")


def test_relative_output_root_is_resolved_from_project(tmp_path):
    assert resolve_output_root(tmp_path, "artifacts") == tmp_path / "artifacts"
    assert resolve_output_root(tmp_path, None) == tmp_path / "outputs"


def test_run_id_is_sortable_and_contains_pid():
    started = datetime(2026, 7, 30, 9, 50, 58, 123456)
    assert build_run_id(started, 2384293) == "20260730-095058-123456-p2384293"


def test_create_run_layout_writes_manifest_and_sanitizes_model(tmp_path):
    layout = create_run_layout(
        project_root=tmp_path,
        output_root=None,
        dataset="FashionIQ",
        stage="combiner",
        model_name="ViT-B/32",
        started_at=datetime(2026, 7, 30, 9, 50, 58, 123456),
        pid=2384293,
    )
    assert layout.root == (
        tmp_path / "outputs/fashioniq/combiner/vit-b-32/"
        "20260730-095058-123456-p2384293"
    )
    assert layout.checkpoints == layout.root / "checkpoints"
    payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert payload["model_name"] == "ViT-B/32"
    assert payload["model_slug"] == "vit-b-32"
    assert payload["checkpoint_dir"] == "checkpoints"


def test_create_run_layout_never_reuses_existing_run(tmp_path):
    kwargs = {
        "project_root": tmp_path,
        "output_root": None,
        "dataset": "CIRR",
        "stage": "clip-finetune",
        "model_name": "RN50x4",
        "started_at": datetime(2026, 7, 30, 9, 50, 58, 123456),
        "pid": 123,
    }
    create_run_layout(**kwargs)
    with pytest.raises(FileExistsError):
        create_run_layout(**kwargs)
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_output_paths.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'output_paths'`.

- [ ] **Step 3: Implement the complete path component**

Create `src/output_paths.py` with this public shape and behavior:

```python
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Optional, Union


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
```

Implement `create_run_layout()` so it validates `stage`, computes both slugs and the run ID, calls `root.mkdir(parents=True, exist_ok=False)`, creates `checkpoints/`, and atomically writes `run_manifest.json` through a sibling temporary file followed by `Path.replace()`. The JSON keys are exactly `dataset`, `dataset_slug`, `training_stage`, `model_name`, `model_slug`, `run_id`, `started_at`, `pid`, and `checkpoint_dir`.

- [ ] **Step 4: Run the path tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_output_paths.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the path component**

```bash
git add src/output_paths.py tests/test_output_paths.py
git commit -m "feat: centralize training output paths"
```

---

### Task 2: Route Both Training Entrypoints to `outputs/`

**Files:**
- Modify: `src/clip_fine_tune.py:389-393,658-665,861-959`
- Modify: `src/combiner_train.py:250-258,784-791,1024-1127`
- Modify: `src/utils.py:146-173`
- Create: `tests/test_training_output_paths.py`

**Interfaces:**
- Consumes: `create_run_layout(...) -> RunLayout`
- Produces: both CLIs accept `--output-root PATH`
- Produces: `save_model(...)` writes `training_path / "checkpoints" / f"{name}.pt"`

- [ ] **Step 1: Write failing CLI and checkpoint-directory tests**

Create `tests/test_training_output_paths.py`:

```python
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from utils import save_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_save_model_uses_checkpoints_directory(tmp_path):
    model = nn.Linear(2, 1)
    save_model("tiny", 3, model, tmp_path)
    payload = torch.load(
        tmp_path / "checkpoints/tiny.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["epoch"] == 3
    assert not (tmp_path / "saved_models").exists()


def test_training_sources_do_not_construct_legacy_output_root():
    for relative in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert 'f"models/' not in source
        assert "/saved_models" not in source


def test_training_clis_expose_output_root():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    for script in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--output-root" in result.stdout
```

- [ ] **Step 2: Run the focused tests and confirm failures**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_training_output_paths.py -q
```

Expected: the checkpoint assertion fails because `save_model()` still uses `saved_models`, and the source/CLI assertions fail because both entrypoints still construct `models/`.

- [ ] **Step 3: Change checkpoint saving to `checkpoints/`**

In `src/utils.py`, change only the destination directory:

```python
def save_model(
    name: str,
    cur_epoch: int,
    model_to_save: nn.Module,
    training_path: Path,
):
    models_path = training_path / "checkpoints"
    models_path.mkdir(exist_ok=True, parents=True)
```

Preserve the existing state-dict selection and `torch.save()` payload.

- [ ] **Step 4: Replace all four hard-coded training paths**

Import `create_run_layout` in each training entrypoint. At the beginning of FashionIQ and CIRR training functions, replace timestamp/path concatenation with:

```python
layout = create_run_layout(
    project_root=base_path,
    output_root=kwargs.get("output_root"),
    dataset="fashioniq",
    stage="clip-finetune",
    model_name=clip_model_name,
)
training_path = layout.root
```

Use `dataset="cirr"` in the CIRR function and `stage="combiner"` in both Combiner functions. Remove the superseded `_safe_model_tag()` use from output construction, but retain the helper if another call site still needs it.

Write `training_hyperparameters.json` after directory creation in all four functions:

```python
with (training_path / "training_hyperparameters.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(training_hyper_params, file, sort_keys=True, indent=4)
```

- [ ] **Step 5: Add and propagate `--output-root`**

Add the same argument to both parsers:

```python
parser.add_argument(
    "--output-root",
    type=str,
    default=None,
    help="Training output root; relative paths are resolved from the project root",
)
```

Add `"output_root": args.output_root` to both `training_hyper_params` dictionaries so the existing `**training_hyper_params` calls pass it to all four training functions and persist it in the hyperparameter JSON.

- [ ] **Step 6: Run focused and existing CLI tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_output_paths.py \
            tests/test_training_output_paths.py \
            tests/test_training_cli_dataset_root.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Compile the modified modules**

Run:

```bash
/data0/qrchen/miniconda3/envs/clip4cir/bin/python -m py_compile \
  src/output_paths.py src/utils.py src/clip_fine_tune.py src/combiner_train.py
```

Expected: exit code 0.

- [ ] **Step 8: Commit training integration**

```bash
git add src/clip_fine_tune.py src/combiner_train.py src/utils.py \
  tests/test_training_output_paths.py
git commit -m "feat: route training runs to outputs"
```

---

### Task 3: Legacy Scanner and Strict Classifier

**Files:**
- Create: `src/model_output_migration.py`
- Create: `tests/test_model_output_migration.py`

**Interfaces:**
- Produces: `LegacyRun(source, destination, dataset_slug, stage, model_name, model_slug, run_id, checkpoints, metrics, newest_mtime_ns, pid, classification, reasons)`
- Produces: `ScanResult(runs, reports, unknown_paths, source_root, output_root)`
- Produces: `parse_legacy_run(source_root: Path, run_root: Path, output_root: Path) -> LegacyRun`
- Produces: `scan_legacy_outputs(source_root: Path, output_root: Path, *, now: datetime, pid_is_alive: Callable[[int], bool], legacy_writer_pids: Tuple[int, ...]) -> ScanResult`

- [ ] **Step 1: Write failing parser and classification tests**

Add fixtures and tests to `tests/test_model_output_migration.py`:

```python
from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import zipfile

from model_output_migration import scan_legacy_outputs


NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _checkpoint(path: Path, payload: bytes) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(
            "archive/data.pkl",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        archive.writestr(info, payload)
    return _write(path, buffer.getvalue())


def _age_tree(path: Path, age: timedelta) -> None:
    timestamp = (NOW - age).timestamp()
    for item in [path, *path.rglob("*")]:
        os.utime(item, (timestamp, timestamp))


def test_scanner_reconstructs_model_name_split_by_slash(tmp_path):
    source = tmp_path / "models"
    run = (
        source
        / "combiner_trained_on_fiq_ViT-B"
        / "32_2026-03-30_13:06:57_pid1462419"
    )
    _checkpoint(run / "saved_models/combiner.pt", b"weights")
    _write(run / "validation_metrics.csv", b"epoch,recall\n1,0.5\n")
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    record = result.runs[0]
    assert record.model_name == "ViT-B/32"
    assert record.model_slug == "vit-b-32"
    assert record.stage == "combiner"
    assert record.destination.parts[-4:-1] == (
        "fashioniq",
        "combiner",
        "vit-b-32",
    )


def test_failed_run_requires_all_four_conditions(tmp_path):
    source = tmp_path / "models"
    run = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid7"
    _write(run / "training_hyperparameters.json", b"{}")
    (run / "saved_models").mkdir()
    _age_tree(run, timedelta(hours=25))
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    assert result.runs[0].classification == "failed"


def test_recent_empty_run_is_not_failed(tmp_path):
    source = tmp_path / "models"
    run = source / "combiner_trained_on_fiq_RN50x4_2026-07-30_11:00:00_pid8"
    _write(run / "training_hyperparameters.json", b"{}")
    (run / "saved_models").mkdir()
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    assert result.runs[0].classification == "unresolved"
    assert "modified-within-24-hours" in result.runs[0].reasons


def test_live_pid_and_nonempty_metrics_each_prevent_failure(tmp_path):
    source = tmp_path / "models"
    live = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid9"
    metrics = source / "combiner_trained_on_fiq_RN50x4_2026-01-02_00:00:00_pid10"
    _write(live / "training_hyperparameters.json", b"{}")
    _write(metrics / "validation_metrics.csv", b"epoch,recall\n1,0.1\n")
    (live / "saved_models").mkdir()
    (metrics / "saved_models").mkdir()
    _age_tree(source, timedelta(hours=25))
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: pid == 9,
        legacy_writer_pids=(),
    )
    by_pid = {run.pid: run for run in result.runs}
    assert by_pid[9].classification == "active"
    assert by_pid[10].classification == "unresolved"


def test_unknown_top_level_file_blocks_clean_scan(tmp_path):
    source = tmp_path / "models"
    _write(source / "mystery.bin", b"unknown")
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    assert result.unknown_paths == (source / "mystery.bin",)


def test_nonempty_corrupt_checkpoint_is_unresolved(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    _write(run / "saved_models/tuned.pt", b"not-a-checkpoint")
    _age_tree(run, timedelta(hours=25))
    result = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    assert result.runs[0].classification == "unresolved"
    assert "checkpoint-format-invalid" in result.runs[0].reasons
```

- [ ] **Step 2: Run the migration tests and confirm the missing-module failure**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'model_output_migration'`.

- [ ] **Step 3: Implement immutable scanner records and legacy parsing**

Create `src/model_output_migration.py` using frozen dataclasses. Match a run against its source-relative POSIX path with:

```python
LEGACY_RUN_RE = re.compile(
    r"^(clip_finetuned|combiner_trained)_on_(fiq|cirr)_(.+)_"
    r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"
    r"(?:_\d+)?(?:_pid\d+)?)$"
)
```

Discover candidate run roots as the parent of every directory named `saved_models`, plus directories containing `training_hyperparameters.json`. Deduplicate roots before parsing. Map `fiq` to `fashioniq`, map the prefix to the exact stage, preserve the original model text including `/`, and normalize it through `slugify_component()`.

Validate checkpoint containers before classification. For PyTorch ZIP serialization, require `zipfile.is_zipfile()` and a clean `ZipFile.testzip()` result. For `.safetensors`, parse the little-endian 8-byte header length and JSON header, then require every tensor data offset to stay within the file. A nonempty legacy pickle checkpoint that cannot be structurally verified without loading tensor storage is `unresolved`, not `valid` or `failed`.

Classify a run as:

- `valid` when it contains a nonempty checkpoint candidate that passes structural format validation;
- `failed` only when all four global conditions hold;
- `active` when its parsed PID is alive or `legacy_writer_pids` is nonempty;
- `unresolved` for recent empty runs, nonempty metrics without a checkpoint, malformed paths, or ambiguous content.

Treat `.xlsx` files outside runs as legacy reports. Every other file not owned by a recognized run is an unknown path.

- [ ] **Step 4: Run scanner tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
```

Expected: all scanner tests pass.

- [ ] **Step 5: Commit scanner behavior**

```bash
git add src/model_output_migration.py tests/test_model_output_migration.py
git commit -m "feat: classify legacy model outputs"
```

---

### Task 4: Hash Plan, Duplicate Groups, and Reports

**Files:**
- Modify: `src/model_output_migration.py`
- Modify: `tests/test_model_output_migration.py`

**Interfaces:**
- Produces: `FileSnapshot(path, size, mtime_ns, sha256)`
- Produces: `MigrationAction(source, destination, kind, status, size, sha256, duplicate_group, canonical, reason)`
- Produces: `MigrationPlan(scan, actions, duplicate_groups, logical_bytes, physical_bytes_before)`
- Produces: `sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str`
- Produces: `build_migration_plan(scan: ScanResult) -> MigrationPlan`
- Produces: `write_migration_reports(plan: MigrationPlan, output_root: Path) -> Tuple[Path, Path]`

- [ ] **Step 1: Add failing hashing, grouping, and report tests**

Append:

```python
import csv
import hashlib
import json

from model_output_migration import (
    build_migration_plan,
    sha256_file,
    write_migration_reports,
)


def test_sha256_file_matches_hashlib(tmp_path):
    checkpoint = _write(tmp_path / "model.pt", b"abc" * 1000)
    assert sha256_file(checkpoint) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_plan_groups_only_byte_identical_checkpoints(tmp_path):
    source = tmp_path / "models"
    first = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    second = source / "clip_finetuned_on_fiq_RN50x4_2026-01-02_00:00:00_000002"
    third = source / "clip_finetuned_on_fiq_RN50x4_2026-01-03_00:00:00_000003"
    _checkpoint(first / "saved_models/tuned.pt", b"same")
    _checkpoint(second / "saved_models/tuned.pt", b"same")
    _checkpoint(third / "saved_models/tuned.pt", b"diff")
    scan = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    plan = build_migration_plan(scan)
    assert len(plan.duplicate_groups) == 1
    group = next(iter(plan.duplicate_groups.values()))
    assert len(group) == 2
    assert all(path.name == "tuned.pt" for path in group)


def test_reports_contain_path_hash_status_and_canonical(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    scan = scan_legacy_outputs(
        source,
        tmp_path / "outputs",
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    plan = build_migration_plan(scan)
    csv_path, json_path = write_migration_reports(plan, tmp_path / "outputs")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["old_path"].endswith("saved_models/tuned.pt")
    assert rows[0]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["valid_runs"] == 1
    assert report["logical_bytes"] >= len(b"weights")
```

- [ ] **Step 2: Run the new tests and confirm missing-interface failures**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py \
  -k "sha256 or groups or reports" -q
```

Expected: import fails for the new interfaces.

- [ ] **Step 3: Implement streaming snapshots and deterministic groups**

Hash regular files in 8 MiB chunks. For each file, capture `st_size` and `st_mtime_ns` before hashing, hash it, restat it, and raise `SourceChangedError` if either field changed.

Build destinations by replacing the run-relative `saved_models/` prefix with `checkpoints/`. Group checkpoint snapshots by `(size, sha256)` and retain only groups with at least two members. Sort every group by normalized destination path; the first member is the canonical copy.

Create actions for every file and failed directory. Set action statuses to `planned-move`, `planned-deduplicate`, `planned-delete-failed`, `skipped-active`, `unresolved`, or `error`. `MigrationPlan.has_blockers` is true for active, unresolved, unknown, collision, or error actions.

- [ ] **Step 4: Implement atomic CSV and JSON report writing**

Write reports first to `.migration_manifest.csv.tmp` and `.migration_report.json.tmp`, then replace the final files. CSV columns are exactly:

```python
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
```

The JSON report includes `total_runs`, `valid_runs`, `failed_runs`, `active_runs`, `unresolved_runs`, `total_files`, `logical_bytes`, `physical_bytes_before`, `duplicate_groups`, `duplicate_files`, and a `status_counts` mapping.

- [ ] **Step 5: Run the full migration module tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit planning and reports**

```bash
git add src/model_output_migration.py tests/test_model_output_migration.py
git commit -m "feat: plan and report model migrations"
```

---

### Task 5: Transactional Apply, Hard-Link Deduplication, Verification, and Finalization

**Files:**
- Modify: `src/model_output_migration.py`
- Modify: `tests/test_model_output_migration.py`

**Interfaces:**
- Produces: `apply_migration(plan: MigrationPlan) -> MigrationPlan`
- Produces: `verify_migration(manifest_path: Path) -> VerificationResult`
- Produces: `finalize_source(source_root: Path, verification: VerificationResult) -> None`
- Produces: `find_legacy_writer_pids(project_root: Path) -> Tuple[int, ...]`

- [ ] **Step 1: Add failing end-to-end temporary-directory tests**

Append an integration test that creates two valid duplicate runs, one strict failed run, and one `.xlsx` report, then asserts:

```python
from model_output_migration import (
    apply_migration,
    finalize_source,
    verify_migration,
)


def test_apply_deduplicates_verifies_and_finalize_removes_source(tmp_path):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    first = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    second = source / "clip_finetuned_on_fiq_RN50x4_2026-01-02_00:00:00_000002"
    failed = source / "combiner_trained_on_fiq_RN50x4_2026-01-03_00:00:00_pid999"
    _checkpoint(first / "saved_models/tuned.pt", b"same weights")
    _write(first / "validation_metrics.csv", b"epoch,r\n1,0.5\n")
    _checkpoint(second / "saved_models/tuned.pt", b"same weights")
    _write(second / "validation_metrics.csv", b"epoch,r\n1,0.5\n")
    _write(failed / "training_hyperparameters.json", b"{}")
    (failed / "saved_models").mkdir()
    _write(source / "validation_metrics_summary.xlsx", b"xlsx")
    _age_tree(failed, timedelta(hours=25))

    scan = scan_legacy_outputs(
        source,
        output,
        now=NOW,
        pid_is_alive=lambda pid: False,
        legacy_writer_pids=(),
    )
    applied = apply_migration(build_migration_plan(scan))
    manifest = output / "migration_manifest.csv"
    verification = verify_migration(manifest)
    checkpoint_paths = sorted(output.glob("fashioniq/clip-finetune/rn50x4/*/checkpoints/tuned.pt"))
    assert len(checkpoint_paths) == 2
    assert checkpoint_paths[0].stat().st_ino == checkpoint_paths[1].stat().st_ino
    assert not failed.exists()
    assert (output / "reports/legacy/validation_metrics_summary.xlsx").exists()
    assert source.exists()
    assert verification.ok
    assert not applied.has_blockers

    finalize_source(source, verification)
    assert not source.exists()
```

Add separate tests asserting that apply raises a specific exception and changes no files for:

- a live writer PID;
- an unknown file;
- an existing destination collision;
- mocked source and output `st_dev` mismatch;
- a source file whose size or mtime changes after planning;
- a symlink inside a run;
- a failed verification result passed to `finalize_source()`;
- a nonempty source containing an unexpected file.

- [ ] **Step 2: Run apply tests and confirm missing-interface failures**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py \
  -k "apply or finalize or writer or collision or symlink" -q
```

Expected: imports fail for the new interfaces.

- [ ] **Step 3: Implement preflight and staged same-filesystem moves**

`apply_migration()` must reject `plan.has_blockers`, any symlink under the source, a changed file snapshot, an existing destination, or differing source/output `st_dev`.

Create staging roots below `output_root / ".staging"`. Move every valid run root with `os.replace()`, rename its `saved_models/` directory to `checkpoints/`, and verify each staged file against its planned size and SHA-256. If any staged verification fails, restore every staged run to its exact old path before raising.

After all staged runs verify, move them to final destinations with `os.replace()`. Write each migrated `run_manifest.json` using the parsed legacy metadata and include `legacy_source`.

- [ ] **Step 4: Implement physical deduplication and failed-run deletion**

For each noncanonical duplicate:

1. Verify canonical and duplicate size and SHA-256 again.
2. Create `duplicate.parent / f".{duplicate.name}.dedupe-link"` with `os.link(canonical, temporary)`.
3. Atomically replace the duplicate with `os.replace(temporary, duplicate)`.
4. Verify canonical and duplicate `st_dev` and `st_ino` match.

Write a pre-deletion report containing `planned-delete-failed`, then delete only resolved failed run paths that are descendants of the resolved source root. Use `shutil.rmtree()` on each exact failed run, never on the source root. Move recognized Excel reports to `outputs/reports/legacy/`. Rewrite reports with final statuses `moved`, `deduplicated`, and `deleted-failed`.

- [ ] **Step 5: Implement `/proc` writer detection**

Scan numeric `/proc/<pid>/cmdline` entries. Resolve relative script arguments against `/proc/<pid>/cwd`, then return PIDs whose arguments identify this project’s `src/clip_fine_tune.py` or `src/combiner_train.py`; this must catch both absolute and project-relative invocations. Ignore vanished or permission-denied process entries. The CLI passes this tuple into the scanner; any nonempty tuple blocks apply.

- [ ] **Step 6: Implement manifest verification and guarded finalization**

`verify_migration()` rehashes every `moved` or `deduplicated` new path, verifies expected sizes and hashes, verifies duplicate canonical paths share an inode, confirms every `deleted-failed` old path is absent, and returns a frozen result containing `ok`, `checked_files`, `checked_bytes`, and `errors`.

`finalize_source()` requires `verification.ok`. It walks the source without following symlinks, rejects any file or symlink, sorts directories deepest first, calls `Path.rmdir()` on each, then calls `source_root.rmdir()`. It must never call `shutil.rmtree()` for finalization.

- [ ] **Step 7: Run all migration tests**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
```

Expected: all tests pass, including inode equality and source removal.

- [ ] **Step 8: Commit transactional migration**

```bash
git add src/model_output_migration.py tests/test_model_output_migration.py
git commit -m "feat: migrate and deduplicate model outputs"
```

---

### Task 6: Migration CLI, Documentation, and Repository Guards

**Files:**
- Create: `scripts/organize_model_outputs.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `tests/test_model_output_migration.py`

**Interfaces:**
- Consumes: scanner, planner, apply, verify, and finalize APIs.
- Produces: default dry run plus mutually exclusive `--apply`, `--verify`, and `--finalize` modes.

- [ ] **Step 1: Add failing CLI tests**

Append subprocess tests that use temporary source and output roots:

```python
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SCRIPT = PROJECT_ROOT / "scripts/organize_model_outputs.py"


def test_cli_dry_run_does_not_modify_source_or_output(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            "--source",
            str(source),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert checkpoint.exists()
    assert not (tmp_path / "outputs").exists()
    assert '"valid_runs": 1' in result.stdout


def test_cli_modes_are_mutually_exclusive(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            "--source",
            str(tmp_path / "models"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--apply",
            "--finalize",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
```

- [ ] **Step 2: Run CLI tests and confirm the missing-script failure**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -k "cli" -q
```

Expected: dry-run test fails because the CLI script does not exist.

- [ ] **Step 3: Implement the thin CLI**

The parser accepts:

```text
--source PATH          default: <project>/models
--output-root PATH     default: <project>/outputs
--apply                scan, hash, migrate, deduplicate, and delete strict failures
--verify               verify <output-root>/migration_manifest.csv
--finalize             verify the manifest, then remove only the empty source tree
```

Make `--apply`, `--verify`, and `--finalize` mutually exclusive. Default mode scans and hashes, prints the same JSON summary that apply would write, and creates no directory or file. Apply refuses to start when `find_legacy_writer_pids(PROJECT_ROOT)` is nonempty. Verify exits 0 only when `VerificationResult.ok` is true. Finalize internally runs verification before calling `finalize_source()`.

- [ ] **Step 4: Update `.gitignore`**

Replace the legacy generated-output entry:

```gitignore
models/
```

with:

```gitignore
outputs/
```

Keeping `models/` unignored ensures any future regression that recreates it is visible in `git status`.

- [ ] **Step 5: Update README paths and usage**

Change the repository tree comment from `models/` to `outputs/`, document the canonical hierarchy, document `--output-root`, replace `saved_models` with `checkpoints`, and update evaluation examples from:

```text
models/<combiner_run>/best_combiner.pth
```

to:

```text
outputs/fashioniq/combiner/<model>/<run-id>/checkpoints/combiner.pt
```

Add a “Legacy output migration” section containing the four commands:

```bash
PYTHONPATH=src python scripts/organize_model_outputs.py
PYTHONPATH=src python scripts/organize_model_outputs.py --apply
PYTHONPATH=src python scripts/organize_model_outputs.py --verify
PYTHONPATH=src python scripts/organize_model_outputs.py --finalize
```

Explain that apply is blocked by old training writers and finalize only removes an empty, fully verified source tree.

- [ ] **Step 6: Run focused tests and source-reference checks**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_output_paths.py \
            tests/test_training_output_paths.py \
            tests/test_model_output_migration.py \
            tests/test_training_cli_dataset_root.py -q
rg -n 'base_path / f?"models|saved_models|models/<combiner_run>' \
  src README.md .gitignore
```

Expected: tests pass and `rg` returns no matches.

- [ ] **Step 7: Run the complete test suite**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit CLI and documentation**

```bash
git add scripts/organize_model_outputs.py .gitignore README.md \
  tests/test_model_output_migration.py
git commit -m "docs: add safe model output migration workflow"
```

---

### Task 7: Audit and Apply the Real 99 GB Migration

**Files:**
- Generated, ignored: `outputs/migration_manifest.csv`
- Generated, ignored: `outputs/migration_report.json`
- Move: `models/**` to `outputs/**`
- Delete only after verification: `models/`

**Interfaces:**
- Consumes: the committed migration CLI and all safety gates.
- Produces: a verified `outputs/` tree and no legacy `models` path.

- [ ] **Step 1: Record the pre-migration repository and storage state**

Run:

```bash
git status --short --branch
du -sh models
find models -type f | wc -l
find models -type d -name saved_models | wc -l
```

Expected baseline: the known unrelated worktree changes remain, `models` is about 99 GB, and run/file counts are recorded in the execution log.

- [ ] **Step 2: Run the full suite immediately before touching artifacts**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Confirm there are no legacy training writers**

Run:

```bash
pgrep -af 'src/(clip_fine_tune|combiner_train)\.py' || true
```

Expected: no matching process. If a process is listed, do not run apply; repeat this check after that training process exits because moving an active run is forbidden.

- [ ] **Step 4: Run and save a real dry-run audit outside the source and destination**

Run:

```bash
audit_dir=$(mktemp -d /data0/qrchen/clip4cir-model-migration-audit.XXXXXX)
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  scripts/organize_model_outputs.py \
  --source /data0/qrchen/projects/CLIP4Cir/models \
  --output-root /data0/qrchen/projects/CLIP4Cir/outputs \
  > "$audit_dir/dry-run.json"
/data0/qrchen/miniconda3/envs/clip4cir/bin/python -m json.tool \
  "$audit_dir/dry-run.json"
```

Expected: exit code 0; `active_runs`, `unknown_paths`, `collisions`, and `errors` are all zero. The JSON reports the exact valid-run, strict-failed-run, duplicate-group, file, logical-byte, and reclaimable-byte counts. If `unresolved_runs` is nonzero only because an empty run is younger than 24 hours, preserve the audit, do not apply, and rerun this step after that run crosses the agreed 24-hour boundary.

- [ ] **Step 5: Apply staged moves, physical deduplication, report migration, and strict failure deletion**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  scripts/organize_model_outputs.py \
  --source /data0/qrchen/projects/CLIP4Cir/models \
  --output-root /data0/qrchen/projects/CLIP4Cir/outputs \
  --apply
```

Expected: exit code 0. `outputs/migration_manifest.csv` and `outputs/migration_report.json` exist. The physical output size is smaller than the pre-migration logical size when duplicate groups exist. The top-level `models/` path still exists until explicit finalization.

- [ ] **Step 6: Independently verify every migrated hash and duplicate inode**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  scripts/organize_model_outputs.py \
  --source /data0/qrchen/projects/CLIP4Cir/models \
  --output-root /data0/qrchen/projects/CLIP4Cir/outputs \
  --verify
```

Expected: exit code 0, zero verification errors, and checked file/byte counts equal the final migration report.

- [ ] **Step 7: Check the source tree before final deletion**

Run:

```bash
find models -mindepth 1 \( -type f -o -type l \) -print
find models -mindepth 1 -type d -name saved_models -print
```

Expected: both commands print nothing. Empty legacy parent directories may remain for bottom-up `rmdir`; any file, symlink, or `saved_models` directory blocks finalization and must be reconciled against the manifest without recursive deletion.

- [ ] **Step 8: Finalize and remove the empty legacy path**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  scripts/organize_model_outputs.py \
  --source /data0/qrchen/projects/CLIP4Cir/models \
  --output-root /data0/qrchen/projects/CLIP4Cir/outputs \
  --finalize
test ! -e /data0/qrchen/projects/CLIP4Cir/models
```

Expected: both commands exit 0 and the legacy path no longer exists.

- [ ] **Step 9: Verify the final repository and artifact state**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest -q
rg -n 'base_path / f?"models|saved_models|models/<combiner_run>' \
  src README.md .gitignore
git status --short --branch
du -sh outputs
```

Expected: all tests pass; the reference scan is empty; `models/` is absent; `outputs/` is ignored; only the user’s pre-existing unrelated worktree changes remain visible; and output storage matches the verified report.

No Git commit is required for Task 7 because every migrated artifact and report is intentionally ignored. Preserve the external dry-run audit directory until the user has accepted the migration result.
