# Unified Evaluation Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both validation entry points publish manifests, JSON metrics, CSV metrics, and logs beneath `outputs/<actual-dataset>/evaluation/`.

**Architecture:** A new `evaluation_outputs.py` module owns run allocation, manifest state, log teeing, and atomic metric publication. Dataset-root classification remains in `dataset_identity.py`; each validation script resolves identity before loading a model, converts its native metrics to one canonical result shape, and delegates persistence to the shared module.

**Tech Stack:** Python 3, `argparse`, `dataclasses`, `pathlib`, standard-library JSON/CSV/context managers, pytest, Bash command templates

## Global Constraints

- The output path is `outputs/<actual-dataset>/evaluation/<model-slug>/<timestamp-and-pid-run-id>/`.
- A successful run contains `evaluation_manifest.json`, `evaluation_metrics.json`, `evaluation_metrics.csv`, and `evaluation.log`.
- An allocated failed run contains a failed manifest and log but no final metric files.
- FashionIQ-compatible loading does not imply an actual dataset named `fashioniq`.
- `--output-csv` accepts only a basename inside the allocated evaluation run.
- Existing training runs and historical validation files are not moved or deleted.
- Preserve unrelated working-tree changes; stage only the hunks named by each task.

---

## File Structure

- Create `src/evaluation_outputs.py`: evaluation run layout, manifest lifecycle, log tee, atomic JSON/CSV publication.
- Modify `src/dataset_identity.py`: resolve one physical FashionIQ-format root for evaluation categories and split.
- Modify `src/validate.py`: expose output CLI, resolve identity before model loading, return structured metrics, persist the run.
- Modify `src/validate_retizero_lora.py`: expose explicit dataset CLI and use the same structured output lifecycle.
- Create `tests/test_evaluation_outputs.py`: pure filesystem and lifecycle tests.
- Modify `tests/test_dataset_identity.py`: evaluation identity tests.
- Modify `tests/test_validate_cli.py`: CLI and mocked single-model orchestration tests.
- Create `tests/test_validate_retizero_lora.py`: RetiZero CLI, dataset wiring, and mocked multi-checkpoint output tests.
- Modify `tests/test_combined_commands.py`: validation-template routing assertions.
- Modify `命令.sh`: three Combined Fundus validation templates.
- Modify `README.md`: document actual-dataset evaluation outputs and updated examples.

### Task 1: Shared Evaluation Output Lifecycle

**Files:**
- Create: `src/evaluation_outputs.py`
- Create: `tests/test_evaluation_outputs.py`

**Interfaces:**
- Consumes: `DatasetIdentity`, `build_run_id()`, `resolve_output_root()`, and `slugify_component()`.
- Produces:
  - `EvaluationLayout(root, manifest, metrics_json, metrics_csv, log, run_id)`
  - `validate_metrics_csv_filename(value: str) -> str`
  - `json_safe(value: Any) -> Any`
  - `create_evaluation_layout(...) -> EvaluationLayout`
  - `tee_evaluation_output(log_path: Path) -> ContextManager[None]`
  - `publish_evaluation_metrics(layout, document, rows, fieldnames) -> None`
  - `discard_evaluation_metrics(layout) -> None`
  - `finalize_evaluation(layout, status, error=None, completed_at=None) -> None`

- [ ] **Step 1: Write failing layout and filename tests**

Create `tests/test_evaluation_outputs.py` with deterministic path and safety
coverage:

```python
import json
from datetime import datetime

import pytest

from dataset_identity import DatasetIdentity
from evaluation_outputs import (
    create_evaluation_layout,
    validate_metrics_csv_filename,
)


def _identity(dataset="idrid"):
    return DatasetIdentity(
        dataset_slug=dataset,
        dataset_format="fashioniq",
        root_requested="/datasets/link",
        root_resolved="/datasets/IDRiD_CIR_Dataset_cold",
        classification_evidence=("resolved-root:idrid",),
    )


def test_create_evaluation_layout_writes_running_manifest(tmp_path):
    layout = create_evaluation_layout(
        project_root=tmp_path,
        output_root=None,
        identity=_identity(),
        evaluation_script="src/validate.py",
        evaluation_name="idrid-val",
        model_name="ViT-B/32",
        split="val",
        categories=["IDRiD"],
        cli_args={"clip_model_path": tmp_path / "clip.pt"},
        input_paths={"clip_model_path": tmp_path / "clip.pt"},
        started_at=datetime(2026, 7, 30, 16, 5, 6, 123456),
        pid=42,
    )

    assert layout.root == (
        tmp_path / "outputs/idrid/evaluation/vit-b-32/"
        "20260730-160506-123456-p42"
    )
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["dataset"] == "idrid"
    assert manifest["dataset_format"] == "fashioniq"
    assert manifest["evaluation_name"] == "idrid-val"
    assert manifest["categories"] == ["IDRiD"]
    assert manifest["cli_args"]["clip_model_path"] == str(tmp_path / "clip.pt")
    assert manifest["metric_files"] == {
        "json": "evaluation_metrics.json",
        "csv": "evaluation_metrics.csv",
    }
    assert layout.log.exists()


@pytest.mark.parametrize("value", ["/tmp/results.csv", "../results.csv", "sub/results.csv"])
def test_metrics_csv_filename_rejects_paths(value):
    with pytest.raises(ValueError, match="filename"):
        validate_metrics_csv_filename(value)


def test_metrics_csv_filename_accepts_basename():
    assert validate_metrics_csv_filename("retizero.csv") == "retizero.csv"
```

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_evaluation_outputs.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'evaluation_outputs'`.

- [ ] **Step 3: Implement layout creation and JSON-safe manifest writes**

Create `src/evaluation_outputs.py` with these public types and signatures:

```python
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TextIO, Union

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
        raise ValueError("--output-csv must be a filename inside the evaluation run")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
```

Implement `create_evaluation_layout()` with this exact interface:

```python
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
```

Use `root.mkdir(parents=True, exist_ok=False)`. Write the initial manifest
through a private `_atomic_write_json(path, payload)` helper. Include
`schema_version=1`, all `DatasetIdentity` fields, `status="running"`, script,
name, model name/slug, split, categories, JSON-safe CLI/input mappings, run ID,
PID, ISO start time, `log_file`, and both metric filenames. Create an empty
UTF-8 `evaluation.log` immediately after allocating the run so every allocated
failed run retains a log; `tee_evaluation_output()` must reopen it in append
mode.

- [ ] **Step 4: Run layout tests and verify they pass**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_evaluation_outputs.py
```

Expected: all current tests pass.

- [ ] **Step 5: Add failing lifecycle, atomic publication, and tee tests**

Append tests that:

```python
import csv
import sys

from evaluation_outputs import (
    discard_evaluation_metrics,
    finalize_evaluation,
    publish_evaluation_metrics,
    tee_evaluation_output,
)


def test_publish_metrics_and_finalize_success(tmp_path):
    layout = _make_layout(tmp_path)
    publish_evaluation_metrics(
        layout,
        {"schema_version": 1, "results": [{"epoch": 3, "average_recall": 8.5}]},
        [{"epoch": 3, "average_recall": 8.5}],
        ["epoch", "average_recall"],
    )
    finalize_evaluation(
        layout,
        "succeeded",
        completed_at=datetime(2026, 7, 30, 16, 6, 7),
    )

    assert json.loads(layout.metrics_json.read_text())["results"][0]["epoch"] == 3
    with layout.metrics_csv.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"epoch": "3", "average_recall": "8.5"}]
    assert json.loads(layout.manifest.read_text())["status"] == "succeeded"
    assert not list(layout.root.glob(".*.tmp"))


def test_failed_run_discards_metrics_and_records_error(tmp_path):
    layout = _make_layout(tmp_path)
    publish_evaluation_metrics(layout, {"results": []}, [], ["epoch"])
    discard_evaluation_metrics(layout)
    error = RuntimeError("checkpoint mismatch")
    finalize_evaluation(layout, "failed", error=error)

    manifest = json.loads(layout.manifest.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "checkpoint mismatch",
    }
    assert not layout.metrics_json.exists()
    assert not layout.metrics_csv.exists()


def test_tee_copies_stdout_and_stderr(tmp_path, capsys):
    layout = _make_layout(tmp_path)
    with tee_evaluation_output(layout.log):
        print("metric line")
        print("warning line", file=sys.stderr)

    captured = capsys.readouterr()
    assert "metric line" in captured.out
    assert "warning line" in captured.err
    log = layout.log.read_text(encoding="utf-8")
    assert "metric line" in log
    assert "warning line" in log
```

Define `_make_layout()` in the test using `_identity()` and fixed timestamp/PID.

- [ ] **Step 6: Implement lifecycle functions**

Implement:

```python
def publish_evaluation_metrics(
    layout: EvaluationLayout,
    document: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
```

Write JSON and CSV temporary siblings first. Flush and close both files before
renaming. If writing or renaming fails, remove temporary files and both final
metric files in this unique run, then re-raise.

Implement `discard_evaluation_metrics()` by unlinking the two final metric
files with `missing_ok=True`.

Implement:

```python
def finalize_evaluation(
    layout: EvaluationLayout,
    status: str,
    *,
    error: Optional[BaseException] = None,
    completed_at: Optional[datetime] = None,
) -> None:
```

Reject statuses outside `{"succeeded", "failed"}`. Reload the running
manifest, set completion time and status, add `error={"type": ..., "message":
...}` only for failure, and atomically replace the manifest.

Implement a private `_Tee(TextIO)` that forwards `write()` and `flush()` to the
original stream and UTF-8 log handle. `tee_evaluation_output()` temporarily
replaces both `sys.stdout` and `sys.stderr` and restores them in `finally`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_evaluation_outputs.py tests/test_output_paths.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit the shared component**

```bash
git add src/evaluation_outputs.py tests/test_evaluation_outputs.py
git commit -m "feat: add evaluation output lifecycle"
```

### Task 2: Evaluation Dataset Identity

**Files:**
- Modify: `src/dataset_identity.py`
- Modify: `tests/test_dataset_identity.py`

**Interfaces:**
- Consumes: existing `resolve_dataset_identity()` and a
  `root_resolver(category, split, dataset_root) -> Path`.
- Produces:
  `resolve_fashioniq_evaluation_identity(*, project_root, dress_types, split, dataset_root, output_dataset, root_resolver) -> DatasetIdentity`.

- [ ] **Step 1: Write failing evaluation identity tests**

Append:

```python
from dataset_identity import resolve_fashioniq_evaluation_identity


def test_evaluation_identity_uses_requested_root_and_split(tmp_path):
    root = tmp_path / "Combined_Fundus_CIR_Dataset"
    root.mkdir()
    calls = []

    identity = resolve_fashioniq_evaluation_identity(
        project_root=tmp_path,
        dress_types=["ODIR5K", "GRAPE"],
        split="test",
        dataset_root=str(root),
        output_dataset="combined-fundus-cir",
        root_resolver=lambda category, split, requested: (
            calls.append((category, split, requested)) or root
        ),
    )

    assert calls == [
        ("ODIR5K", "test", str(root)),
        ("GRAPE", "test", str(root)),
    ]
    assert identity.dataset_slug == "combined-fundus-cir"
    assert identity.root_requested == str(root)
    assert identity.root_resolved == str(root.resolve())


def test_evaluation_identity_rejects_categories_on_multiple_roots(tmp_path):
    first = tmp_path / "IDRiD_CIR_Dataset_cold"
    second = tmp_path / "UWF_CIR_Dataset_cold"
    first.mkdir()
    second.mkdir()

    with pytest.raises(DatasetIdentityError, match="multiple dataset roots"):
        resolve_fashioniq_evaluation_identity(
            project_root=tmp_path,
            dress_types=["IDRiD", "CH"],
            split="val",
            dataset_root=None,
            output_dataset=None,
            root_resolver=lambda category, *_: (
                first if category == "IDRiD" else second
            ),
        )
```

- [ ] **Step 2: Run the tests and verify the missing-symbol failure**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_dataset_identity.py
```

Expected: collection fails because
`resolve_fashioniq_evaluation_identity` is not defined.

- [ ] **Step 3: Implement the evaluation resolver**

Add:

```python
def resolve_fashioniq_evaluation_identity(
    *,
    project_root: Path,
    dress_types: Sequence[str],
    split: str,
    dataset_root: Optional[Union[str, Path]],
    output_dataset: Optional[str],
    root_resolver: Callable[
        [str, str, Optional[Union[str, Path]]], Path
    ],
) -> DatasetIdentity:
```

Resolve every `(category, split, dataset_root)` tuple, call `.resolve()`, and
require exactly one physical root. Preserve an explicit requested root. When
no root was requested, preserve the project `fashionIQ_dataset` symlink if it
resolves to that same root. Delegate final classification to
`resolve_dataset_identity(dataset_format="fashioniq", ...)`.

- [ ] **Step 4: Run identity regression tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_dataset_identity.py tests/test_training_output_paths.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the identity helper**

```bash
git add src/dataset_identity.py tests/test_dataset_identity.py
git commit -m "feat: resolve evaluation dataset identity"
```

### Task 3: Structured Outputs for `validate.py`

**Files:**
- Modify: `src/validate.py`
- Modify: `tests/test_validate_cli.py`

**Interfaces:**
- Consumes: Task 1 output lifecycle and Task 2 evaluation identity.
- Produces:
  - `build_parser() -> ArgumentParser`
  - `_run_evaluation(args, categories) -> list[dict]`
  - `_metrics_document(identity, args, results) -> dict`
  - `_csv_projection(args, categories, results) -> tuple[list[dict], list[str]]`
  - `main(argv=None) -> None`

- [ ] **Step 1: Extend the CLI help test**

Update the existing help assertion:

```python
def test_validate_cli_exposes_combined_dataset_options():
    result = _run_validate("--help")

    assert result.returncode == 0, result.stderr
    for option in (
        "--fashioniq-root",
        "--dress-types",
        "--fashioniq-split",
        "--output-root",
        "--output-dataset",
        "--evaluation-name",
    ):
        assert option in result.stdout
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_cli.py::test_validate_cli_exposes_combined_dataset_options
```

Expected: failure because the three output options are absent.

- [ ] **Step 3: Extract the parser and add output arguments**

Move parser construction to `build_parser()`. Add:

```python
parser.add_argument("--output-root", default=None)
parser.add_argument("--output-dataset", default=None)
parser.add_argument("--evaluation-name", default="validation")
```

Change `main()` to `main(argv=None)` and parse with
`args = build_parser().parse_args(argv)`. Do not move model loading yet.

- [ ] **Step 4: Run the CLI tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_cli.py
```

Expected: all current CLI tests pass.

- [ ] **Step 5: Write failing mocked success and failure tests**

Import `src/validate.py` through `importlib` with `src` on `sys.path`. Create a
minimal FashionIQ-format root containing empty
`captions/cap.Internal.val.json` and
`image_splits/split.Internal.val.json` files plus an `images/` directory.
Monkeypatch
`validate._run_evaluation` so no GPU or model is loaded.

The success test calls:

```python
validate.main([
    "--dataset", "FashionIQ",
    "--fashioniq-root", str(dataset_root),
    "--dress-types", "Internal",
    "--fashioniq-split", "val",
    "--combining-function", "sum",
    "--clip-model-name", "ViT-B/32",
    "--output-root", str(tmp_path / "artifacts"),
    "--output-dataset", "combined-fundus-cir",
    "--evaluation-name", "internal-val",
])
```

Return one canonical result:

```python
[{
    "model_path": None,
    "epoch": None,
    "per_category": {
        "Internal": {
            "recall_at1": 10.0,
            "recall_at5": 20.0,
            "recall_at10": 30.0,
        }
    },
    "aggregate": {
        "average_recall_at1": 10.0,
        "average_recall_at5": 20.0,
        "average_recall_at10": 30.0,
    },
}]
```

Assert one run exists under
`artifacts/combined-fundus-cir/evaluation/vit-b-32/`, the manifest succeeded,
all four files exist, and CSV contains the three category columns and three
average columns.

The failure test patches `_run_evaluation` to raise
`RuntimeError("shape mismatch")`. Assert `main()` re-raises, the manifest is
failed, the log contains `shape mismatch`, and neither metric file exists.

Add a CIRR projection unit test using a result whose `aggregate` contains:

```python
{
    "group_recall_at1": 1.0,
    "group_recall_at2": 2.0,
    "group_recall_at3": 3.0,
    "global_recall_at1": 4.0,
    "global_recall_at5": 5.0,
    "global_recall_at10": 6.0,
    "global_recall_at50": 7.0,
}
```

Assert those exact names are present in the CSV row and fieldnames.

- [ ] **Step 6: Run the new tests and verify structured output is missing**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_cli.py
```

Expected: success/failure orchestration tests fail because no evaluation run is
created.

- [ ] **Step 7: Refactor evaluation and add lifecycle orchestration**

Before model loading:

1. For FashionIQ, determine `categories` from `args.dress_types` or
   `list_fashioniq_categories(args.fashioniq_split, args.fashioniq_root)`.
   Raise `FileNotFoundError` when none exist.
2. Resolve identity with `resolve_fashioniq_evaluation_identity()` and
   `resolve_fashioniq_root`.
3. For CIRR, use `resolve_dataset_identity(dataset_format="cirr", ...)` and an
   empty category list.
4. Create the layout using project root
   `Path(__file__).resolve().parents[1]`.

Move all existing model loading and validation into
`_run_evaluation(args, categories)`. Return the canonical result list instead
of only printing values. FashionIQ returns one result with `per_category` and
the three average recall fields. CIRR returns one result with an empty
`per_category` and the seven explicit aggregate names from Step 5.

Build canonical JSON:

```python
{
    "schema_version": 1,
    "dataset": identity.dataset_slug,
    "dataset_format": identity.dataset_format,
    "evaluation_name": args.evaluation_name,
    "split": args.fashioniq_split if args.dataset.lower() == "fashioniq" else "val",
    "results": results,
}
```

Build a one-row wide CSV. FashionIQ field order is `model_path`, `epoch`, then
three recall columns per requested category, then the three average columns.
CIRR field order is `model_path`, `epoch`, then the seven aggregate fields.

Orchestrate with nested exception logging so the traceback is written before
the tee closes:

```python
try:
    with tee_evaluation_output(layout.log):
        try:
            results = _run_evaluation(args, categories)
            document = _metrics_document(identity, args, results)
            rows, fieldnames = _csv_projection(args, categories, results)
            publish_evaluation_metrics(layout, document, rows, fieldnames)
        except BaseException:
            traceback.print_exc()
            raise
except BaseException as error:
    discard_evaluation_metrics(layout)
    finalize_evaluation(layout, "failed", error=error)
    raise
else:
    finalize_evaluation(layout, "succeeded")
```

Pass `vars(args)` as CLI metadata and `clip_model_path`/`combiner_path` as input
paths.

- [ ] **Step 8: Run validate and lifecycle regressions**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/test_validate_cli.py \
  tests/test_evaluation_outputs.py \
  tests/test_fashioniq_evaluation.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit `validate.py` integration**

```bash
git add src/validate.py tests/test_validate_cli.py
git commit -m "feat: persist structured validation outputs"
```

### Task 4: Structured Outputs for RetiZero LoRA Validation

**Files:**
- Modify: `src/validate_retizero_lora.py`
- Create: `tests/test_validate_retizero_lora.py`

**Interfaces:**
- Consumes: Task 1 lifecycle, Task 2 identity helper, and the canonical result
  format introduced by Task 3.
- Produces:
  - `build_parser() -> ArgumentParser`
  - `validate_single_model(model, categories, preprocess, combining_function, *, split, dataset_root) -> dict`
  - `_run_evaluation(args, categories) -> list[dict]`
  - `main(argv=None) -> None`

- [ ] **Step 1: Write failing CLI and filename validation tests**

Create `tests/test_validate_retizero_lora.py` with a subprocess helper and help
assertions:

```python
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "validate_retizero_lora.py"


def _run(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_exposes_dataset_and_output_options():
    result = _run("--help")
    assert result.returncode == 0, result.stderr

    for option in (
        "--fashioniq-root",
        "--dress-types",
        "--fashioniq-split",
        "--output-root",
        "--output-dataset",
        "--evaluation-name",
        "--output-csv",
    ):
        assert option in result.stdout


def test_cli_rejects_output_csv_outside_run():
    result = _run(
        "--model-paths", "/checkpoints/model.pth",
        "--base-weight-path", "/weights/RetiZero.pth",
        "--output-csv", "../outside.csv",
    )
    assert result.returncode == 2
    assert "filename inside the evaluation run" in result.stderr
```

Argument parsing must fail before any checkpoint is loaded.

- [ ] **Step 2: Run tests and verify the new options are missing**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_retizero_lora.py
```

Expected: CLI assertions fail for the dataset/output options.

- [ ] **Step 3: Extract parser and add explicit dataset/output arguments**

Add `build_parser()` and `main(argv=None)`. Define:

```python
parser.add_argument("--fashioniq-root", default=None)
parser.add_argument("--dress-types", nargs="+", default=None)
parser.add_argument("--fashioniq-split", choices=("val", "test"), default="val")
parser.add_argument("--output-root", default=None)
parser.add_argument("--output-dataset", default=None)
parser.add_argument("--evaluation-name", default="retizero-lora-validation")
parser.add_argument(
    "--output-csv",
    type=validate_metrics_csv_filename,
    default="evaluation_metrics.csv",
)
```

Remove import-time global category discovery. Resolve categories after parsing
from `args.dress_types` or
`list_fashioniq_categories(args.fashioniq_split, args.fashioniq_root)`, and
raise `FileNotFoundError` if the result is empty.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_retizero_lora.py
```

Expected: help and unsafe filename tests pass.

- [ ] **Step 5: Write failing dataset wiring and multi-checkpoint output tests**

Test `validate_single_model()` with monkeypatched `FashionIQDataset`,
`extract_index_features()`, and `compute_fiq_val_metrics()`. Call it with
`split="test"` and `dataset_root="/datasets/combined"`. Assert both classic and
relative dataset calls receive that split and root, and relative mode receives
`return_target=True`.

For orchestration, create a minimal Combined root for `Internal`, patch
`_run_evaluation` to return two canonical results. The root must contain
`captions/`, `image_splits/`, and `images/`, with matching
`cap.Internal.val.json` and `split.Internal.val.json` files:

```python
[
    {
        "model_path": "/checkpoints/a.pth",
        "epoch": 4,
        "classification_val_accuracy": 0.80,
        "per_category": {
            "Internal": {
                "recall_at1": 11.0,
                "recall_at5": 21.0,
                "recall_at10": 31.0,
            }
        },
        "aggregate": {
            "average_recall_at1": 11.0,
            "average_recall_at5": 21.0,
            "average_recall_at10": 31.0,
            "average_recall": 21.0,
        },
    },
    {
        "model_path": "/checkpoints/b.pth",
        "epoch": 9,
        "classification_val_accuracy": 0.84,
        "per_category": {
            "Internal": {
                "recall_at1": 12.0,
                "recall_at5": 22.0,
                "recall_at10": 32.0,
            }
        },
        "aggregate": {
            "average_recall_at1": 12.0,
            "average_recall_at5": 22.0,
            "average_recall_at10": 32.0,
            "average_recall": 22.0,
        },
    },
]
```

Call `main()` with explicit root, category, output root, output dataset, two
model paths, and a base weight path. Assert the run is under
`combined-fundus-cir/evaluation/retizero-lora/`, JSON has two results, CSV has
two rows, and the manifest lists both checkpoints as input paths.

Add a failure test mirroring Task 3: patched `_run_evaluation` raises, manifest
is failed, log is retained, and metrics are absent.

- [ ] **Step 6: Run tests and verify wiring/output failures**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_retizero_lora.py
```

Expected: dataset constructor and structured output assertions fail.

- [ ] **Step 7: Implement dataset wiring, canonical results, and persistence**

Change `validate_single_model()` to accept keyword-only `split` and
`dataset_root`. Pass both into classic and relative `FashionIQDataset`
instances; pass `return_target=(split == "test")` for relative mode.

Move the checkpoint loop into `_run_evaluation(args, categories)`. Convert each
wide internal result to the canonical shape while retaining checkpoint epoch
and classification validation accuracy.

Resolve identity before loading the base model. Create the evaluation layout
with model name `RetiZero LoRA`, the validated CSV filename, both
`model_paths`, and `base_weight_path`. Use the same try/tee/publish/finalize
sequence as Task 3.

CSV field order is:

```python
[
    "model_path",
    "epoch",
    "classification_val_accuracy",
    *category_recall_columns,
    "average_recall_at1",
    "average_recall_at5",
    "average_recall_at10",
    "average_recall",
]
```

Continue printing per-checkpoint and cross-checkpoint summaries so they appear
in `evaluation.log`.

- [ ] **Step 8: Run RetiZero and shared regressions**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/test_validate_retizero_lora.py \
  tests/test_retizero_adapter.py \
  tests/test_retizero_training_integration.py \
  tests/test_evaluation_outputs.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit RetiZero validation integration**

```bash
git add src/validate_retizero_lora.py tests/test_validate_retizero_lora.py
git commit -m "feat: unify RetiZero evaluation outputs"
```

### Task 5: Command Templates, Documentation, and Full Verification

**Files:**
- Modify: `命令.sh`
- Modify: `tests/test_combined_commands.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the CLI options implemented in Tasks 3 and 4.
- Produces: three safe Combined Fundus validation templates and documented
  output examples.

- [ ] **Step 1: Write failing validation-template assertions**

Add a helper that selects the three commented lines containing
`python src/validate.py`. Assert:

```python
assert len(validation_commands) == 3
assert all("--output-dataset combined-fundus-cir" in line for line in validation_commands)
assert {
    re.search(r"--evaluation-name\\s+(\\S+)", line).group(1)
    for line in validation_commands
} == {"internal-test", "odir5k-test", "grape-test"}
assert all("eval_combined_" not in line for line in validation_commands)
assert all(line.endswith("> /dev/null 2>&1 &") for line in validation_commands)
```

- [ ] **Step 2: Run the command test and verify failure**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_combined_commands.py
```

Expected: the new validation-template test fails because output dataset and
evaluation names are absent and project-root log names remain.

- [ ] **Step 3: Update the three templates**

Set the category-specific names:

- `Internal` → `--evaluation-name internal-test`
- `ODIR5K` → `--evaluation-name odir5k-test`
- `GRAPE` → `--evaluation-name grape-test`

Add `--output-dataset combined-fundus-cir` once per line. Replace each
`> eval_combined_*.log 2>&1 &` suffix with `> /dev/null 2>&1 &`.

- [ ] **Step 4: Update validation documentation**

In `README.md`:

- explain the actual-dataset evaluation tree;
- add `--fashioniq-root`, `--output-dataset uwf`, and
  `--evaluation-name uwf-val` to the standard validation example;
- add `--fashioniq-root`, `--dress-types`, `--fashioniq-split`,
  `--output-dataset uwf`, and an evaluation name to the RetiZero example;
- change `--output-csv results_retizero.csv` to
  `--output-csv evaluation_metrics.csv`;
- state that `--output-csv` is a filename within the run, not an external path;
- show the four generated filenames.

- [ ] **Step 5: Run focused command and CLI checks**

Run:

```bash
bash -n 命令.sh
PYTHONPATH=src pytest -q \
  tests/test_combined_commands.py \
  tests/test_validate_cli.py \
  tests/test_validate_retizero_lora.py
PYTHONPATH=src python src/validate.py --help
PYTHONPATH=src python src/validate_retizero_lora.py --help
```

Expected: shell syntax is valid, all tests pass, and both help commands list
the shared output options.

- [ ] **Step 6: Run the complete regression suite**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass; warnings are acceptable only when they already exist
on the baseline and are unrelated to evaluation output persistence.

- [ ] **Step 7: Inspect output hygiene and the final diff**

Run:

```bash
find . -maxdepth 1 -type f \( -name 'eval_*.log' -o -name 'nohup.out' \) -print
git diff --check
git status --short
git diff -- src/evaluation_outputs.py src/dataset_identity.py src/validate.py \
  src/validate_retizero_lora.py tests README.md 命令.sh
```

Expected: the root-file search prints nothing, `git diff --check` is clean,
and no existing output/model files are deleted.

- [ ] **Step 8: Commit only the documentation and validation-template hunks**

`命令.sh` and `tests/test_combined_commands.py` already contain user-approved
working-tree edits from the earlier bulk command update. Inspect each patch and
stage only the new validation-related hunks:

```bash
git add README.md
git add -p -- 命令.sh tests/test_combined_commands.py
git diff --cached --check
git diff --cached
git commit -m "docs: route validation templates to outputs"
```

Leave unrelated pre-existing hunks unstaged.

- [ ] **Step 9: Record final verification evidence**

Run:

```bash
git log -5 --oneline
git status --short
PYTHONPATH=src pytest -q
bash -n 命令.sh
```

Report the exact test summary, the commits created by Tasks 1–5, and any
preserved dirty files. Do not claim the work complete unless this fresh run
passes.
